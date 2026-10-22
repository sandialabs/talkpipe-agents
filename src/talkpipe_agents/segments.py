"""The ``llmAgent`` talkpipe segment — a Strands agent as a pipeline
step.

Each incoming item (or ``field`` of it) becomes the agent's prompt; the
agent may call its tools any number of times before answering, and the
final answer flows on (or lands in ``set_as``). ``source``/``model``
fall back to the configured defaults exactly as in ``llmPrompt``;
``source`` is one of the agent model sources (``ollama``, ``openai``,
``openai_responses``, ``openai_chat``, ``anthropic``, ``litellm``, or
one added with ``register_agent_model`` — see ``models``), and
``model_params`` (JSON text, as chatterlang has no dict literal) passes
provider request parameters such as ``reasoning_effort`` through to it.
``tools`` names come from the ``register_agent_tool`` registry (or are Strands tool paths), so
unknown names fail at chatterlang compile time; ``skills`` names an
agent-skills folder (or skill, ``SKILL.md`` path, or URL) that Strands'
``AgentSkills`` plugin lists in the system prompt and loads on demand
through its ``skills`` tool, and a missing path fails the same way. With
``multi_turn=True`` one agent (one conversation) spans the whole
stream; otherwise each item gets a fresh agent, keeping the step
stateless. Either way the segment's agents share one Strands
``AgentState``, so state-keeping tools (the notepad) carry across the
items of a stream while the conversation itself does not.

The step is bounded by Strands' per-invocation ``limits``: ``max_turns``
(model calls per item; a turn may hold several tool calls) plus the
soft token budgets ``max_output_tokens`` and ``max_total_tokens``.
The Strands ``AgentResult`` of the most recent item is kept as
``last_result`` (its ``stop_reason`` tells a capped run from one the
agent ended itself).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Annotated, Any

from strands import Agent
from strands.agent.state import AgentState
from talkpipe.chatterlang.registry import register_segment
from talkpipe.pipe.core import AbstractSegment
from talkpipe.util.data_manipulation import assign_property, extract_property

from .hooks import AgentHooks, AgentRetryStrategy
from .models import build_agent_model
from .models import model_params as parse_model_params
from .replies import agent_limits, ask_and_clean
from .support import teller
from .tools import resolve_agent_tools, skills_plugin

logger = logging.getLogger(__name__)


@register_segment("llmAgent")
class LLMAgent(AbstractSegment):
    """Run each item through a tool-using agent and yield its answer.

    After each item, ``last_result`` holds the Strands ``AgentResult``:
    ``stop_reason`` is ``"end_turn"`` when the agent finished on its own,
    ``"limit_turns"`` (etc.) when a budget cut it off, ``"interrupt"``
    when a tool paused for a human; ``metrics`` has the token usage.
    """

    def __init__(
        self,
        model: Annotated[
            str | None,
            "The model name, as in llmPrompt (falls back to the configured default)",
        ] = None,
        source: Annotated[
            str | None,
            "The model source: ollama, openai (= openai_responses), openai_chat, "
            "anthropic, litellm, or one added with register_agent_model (falls "
            "back to the configured default)",
        ] = None,
        system_prompt: Annotated[
            str, "The agent's system prompt"
        ] = "You are a helpful assistant.",
        tools: Annotated[
            list | None,
            "Tool names from the agent tool registry, Strands tool paths "
            "(module:function), or tool objects from Python",
        ] = None,
        skills: Annotated[
            str | list | None,
            "Agent skills folder (subdirectories holding a SKILL.md), a skill "
            "directory, a SKILL.md path, or an https:// URL, singly or as a "
            "list; loaded by Strands' AgentSkills plugin",
        ] = None,
        multi_turn: Annotated[
            bool,
            "Keep one conversation across the whole stream instead of a fresh agent per item",
        ] = False,
        max_turns: Annotated[
            int | None,
            "Cap on model calls per item; a turn may hold several tool calls (None = no cap)",
        ] = 8,
        max_output_tokens: Annotated[
            int | None,
            "Soft cap on tokens the model generates per item (None = no cap)",
        ] = None,
        max_total_tokens: Annotated[
            int | None,
            "Soft cap on input + output tokens per item (None = no cap)",
        ] = None,
        temperature: Annotated[float | None, "The sampling temperature"] = None,
        model_params: Annotated[
            dict | str | None,
            "Provider request parameters as a JSON object (a dict from Python), "
            'e.g. \'{"reasoning_effort": "none"}\'; Ollama takes them as options',
        ] = None,
        field: Annotated[
            str | None,
            "Property of the item to use as the prompt (default: the item itself)",
        ] = None,
        set_as: Annotated[
            str | None,
            "Set the answer as this property and yield the item, instead of yielding the answer",
        ] = None,
        progress: Annotated[
            Callable[[str], None] | None,
            "Callable given each one-line progress message (agent → tool(args), "
            "tool errors, limits); default: this module's logger at INFO",
        ] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.system_prompt = system_prompt
        self.tools = resolve_agent_tools(tools or [])
        self.skills = skills_plugin(skills)
        self.multi_turn = multi_turn
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.limits = agent_limits(max_turns, max_output_tokens, max_total_tokens)
        self.temperature = temperature
        # parsed here so malformed JSON fails when the script compiles
        self.model_params = parse_model_params(model_params)
        self.field = field
        self.set_as = set_as
        self.tell = teller(progress, logger)
        self.state = AgentState()
        self.last_result: Any = None
        self._agent: Agent | None = None

    def _make_agent(self) -> Agent:
        for plugin in self.skills:
            # A fresh agent starts from the plain system prompt, so the
            # plugin's record (in the shared state) of what it injected into
            # the previous agent's prompt must not carry over.
            self.state.delete(plugin.name)
        return Agent(
            model=build_agent_model(
                self.source,
                self.model,
                temperature=self.temperature,
                params=self.model_params,
            ),
            tools=list(self.tools),
            plugins=list(self.skills),
            system_prompt=self.system_prompt,
            # the default handler prints to stdout — never in a pipeline
            callback_handler=None,
            hooks=[AgentHooks(self.tell)],
            retry_strategy=AgentRetryStrategy(),
            state=self.state,
        )

    def _ask(self, agent: Agent, prompt: str) -> str:
        self.last_result, answer = ask_and_clean(
            agent, prompt, limits=self.limits, tell=self.tell
        )
        return answer

    def _prompt_of(self, item: Any) -> Any:
        if not self.field:
            return item
        # A missing property must not be prompted as the text "None" and
        # answered with a hallucination; name the field instead.
        try:
            return extract_property(item, self.field, fail_on_missing=True)
        except AttributeError as exc:
            raise ValueError(
                f"llmAgent: field={self.field!r} not found on the item "
                f"({type(item).__name__}); is the property name right?"
            ) from exc

    def transform(self, input_iter: Iterable) -> Iterator:
        for item in input_iter:
            prompt = self._prompt_of(item)
            if self.multi_turn:
                self._agent = self._agent or self._make_agent()
                agent = self._agent
            else:
                agent = self._make_agent()
            answer = self._ask(agent, str(prompt))
            if self.set_as:
                assign_property(item, self.set_as, answer)
                yield item
            else:
                yield answer
