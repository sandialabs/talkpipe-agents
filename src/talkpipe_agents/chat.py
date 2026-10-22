"""``AgentChat`` — one tool-using conversation, for interactive use.

The thin Python-side counterpart of the ``llmAgent`` segment: a chat UI
creates one ``AgentChat`` per conversation and calls ``ask()`` per user
message; the Strands agent keeps the history and may call its tools on
every turn. Memory management is Strands' (sliding window by default;
pass ``summarizing=True`` for LLM-summarized compaction of long
conversations, akin to talkpipe's ``memory_mode="summary_llm"``).
Each ``ask()`` is bounded by the same Strands ``limits`` as the segment:
``max_turns`` plus the soft ``max_output_tokens`` / ``max_total_tokens``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager

from .hooks import AgentHooks, AgentRetryStrategy
from .models import build_agent_model
from .replies import agent_limits, ask_and_clean
from .support import teller
from .tools import resolve_agent_tools, skills_plugin

logger = logging.getLogger(__name__)

# Shown when even the plain-prose retry yielded no answer text; an
# interactive chat must never come back with silence.
NO_ANSWER = "(the model produced no answer text — try asking again or rephrasing)"
# Shown when a turn/token limit ended the exchange before any answer text:
# the cause is the configured budget, not the model, so say so.
LIMIT_REACHED = (
    "(the agent stopped at its {limit} limit ({value}) before answering — "
    "raise the limit or ask a smaller question)"
)
# Shown when a tool raised a Strands interrupt (it wants a human's input)
# and the exchange ended on it: the next ``ask`` answers it.
INTERRUPTED = (
    "(the agent paused for human input by {name}: {reason} — your next "
    "message answers it; or leave that tool out)"
)


class AgentChat:
    """One conversation. Create a new instance to start over.

    Keyword arguments:

    - ``source`` / ``model``: the provider (``ollama``, ``openai``,
      ``openai_responses``, ``openai_chat``, ``anthropic``, ``litellm``,
      or one added with
      ``register_agent_model``) and the model on it. As for talkpipe's
      ``llmPrompt``, both fall back to the configured defaults
      (``TALKPIPE_DEFAULT_MODEL_SOURCE`` / ``TALKPIPE_DEFAULT_MODEL_NAME``).
    - ``system_prompt``: the agent's system prompt.
    - ``tools``: names from the ``register_agent_tool`` registry, Strands
      tool paths (``"module:function"``), ``@strands.tool`` objects, or
      plain functions (wrapped as tools).
    - ``skills``: agent skills, loaded by Strands' ``AgentSkills`` plugin:
      a folder of skill subdirectories (each holding a ``SKILL.md``), one
      skill directory, a ``SKILL.md`` path, or an ``https://`` URL, singly
      or as a list. The skills are listed in the system prompt and a
      ``skills`` tool loads one's full instructions on demand.
    - ``temperature``: sampling temperature, when the provider supports it.
    - ``model_params``: provider request parameters, a dict (or its JSON
      text) such as ``{"reasoning_effort": "none"}``; the provider's API
      defines the keys, and Ollama takes them as its ``options``.
    - ``max_turns``: model calls per ``ask()`` (a turn may hold several
      tool calls; ``None`` = no cap; default 8); ``max_output_tokens`` /
      ``max_total_tokens``: soft token budgets per ``ask()`` (``None`` =
      no cap). Hitting one ends the exchange with a visible notice, and
      ``last_result.stop_reason`` says which (``"limit_turns"``, …). A
      tool that raises a Strands interrupt (human input required) also
      ends the exchange with a notice; ``last_result.interrupts`` holds it,
      and the next ``ask()`` resumes the agent with that message as the
      human's answer.
    - ``summarizing``: LLM-summarized compaction of long conversations
      instead of Strands' default sliding window.
    - ``progress``: a one-line-message callable for ``agent → tool(args)``
      progress lines; defaults to this module's logger at INFO.
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Iterable[Any] = (),
        skills: str | Iterable[str] | None = None,
        temperature: float | None = None,
        model_params: dict[str, Any] | str | None = None,
        max_turns: int | None = 8,
        max_output_tokens: int | None = None,
        max_total_tokens: int | None = None,
        summarizing: bool = False,
        progress: Callable[[str], None] | None = None,
    ):
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.limits = agent_limits(max_turns, max_output_tokens, max_total_tokens)
        self.tell = teller(progress, logger)
        self.last_result: Any = None
        manager = SummarizingConversationManager() if summarizing else None
        self._agent = Agent(
            model=build_agent_model(
                source, model, temperature=temperature, params=model_params
            ),
            tools=resolve_agent_tools(tools),
            plugins=skills_plugin(skills),
            system_prompt=system_prompt,
            callback_handler=None,
            conversation_manager=manager,
            hooks=[AgentHooks(self.tell)],
            retry_strategy=AgentRetryStrategy(),
        )

    @property
    def agent(self) -> Agent:
        """The underlying Strands agent (its ``state``, ``messages``,
        ``tool_registry`` …) for hosts that need more than ``ask``."""
        return self._agent

    @property
    def messages(self) -> list:
        """The conversation so far, as Strands messages."""
        return self._agent.messages

    @property
    def tool_names(self) -> list[str]:
        """The names of the tools this conversation was given, sorted."""
        return sorted(self._agent.tool_names)

    def _resume_or(self, question: str) -> Any:
        """*question* as the agent's input: when the previous exchange
        ended in a tool's interrupt that is still pending, the responses
        that resume it with *question* as the human's answer; otherwise
        *question* itself."""
        state = getattr(self._agent, "_interrupt_state", None)
        pending = getattr(self.last_result, "interrupts", None) or ()
        if not (pending and getattr(state, "activated", False)):
            return question
        return [
            {"interruptResponse": {"interruptId": i.id, "response": question}}
            for i in pending
        ]

    def ask(self, question: str) -> str:
        """Send one user message and return the agent's reply text.

        Blocks while the agent runs (calling tools as it sees fit, within
        the limits). The reply is cleaned of reasoning blocks and leaked
        tool-call syntax; when a limit ended the turn before any answer,
        or the model produced none, a short notice is returned instead
        of an empty string. ``last_result`` holds the Strands
        ``AgentResult`` of this exchange afterwards. A blank question
        returns ``""`` without calling the model. When the previous
        exchange ended in a tool's interrupt (human input required), the
        message is delivered as the human's answer and the agent resumes
        where it paused instead of starting a new turn.
        """
        if not isinstance(question, str):
            raise TypeError(
                "AgentChat.ask() takes the user's message as a str, got "
                f"{type(question).__name__}"
            )
        question = question.strip()
        if not question:
            return ""
        self.last_result, answer = ask_and_clean(
            self._agent, self._resume_or(question), limits=self.limits, tell=self.tell
        )
        if answer:
            return answer
        stop_reason = str(self.last_result.stop_reason)
        if stop_reason.startswith("limit_"):
            key = stop_reason.removeprefix("limit_")
            return LIMIT_REACHED.format(limit=key, value=(self.limits or {}).get(key))
        interrupts = getattr(self.last_result, "interrupts", None) or []
        if interrupts:
            return "\n".join(
                INTERRUPTED.format(name=i.name, reason=i.reason) for i in interrupts
            )
        return NO_ANSWER
