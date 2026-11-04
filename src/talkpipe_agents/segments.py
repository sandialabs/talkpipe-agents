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

With ``judge=True`` the step runs the findings loop (see ``findings``):
each item gets a ``FindingsSession`` for its question, every finding
tool's results are judged by the same model before the agent sees them,
and the agent stops by calling ``enough_information``; ``findings=True``
runs the session without a judge, the agent keeping and ignoring by
hand. ``first_tool`` names a finding tool to call with the question
before the agent's first turn: its findings are judged first, and when
the judge finds them enough the agent only answers, taking no search
turn at all. The run's ``FindingsOutcome`` (kept findings, how it ended, the
trace) is kept as ``last_outcome`` and, with ``findings_as``, set on
the item.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Annotated, Any

from strands import Agent
from strands.agent.state import AgentState
from talkpipe.chatterlang.registry import register_segment
from talkpipe.pipe.core import AbstractSegment
from talkpipe.util.data_manipulation import assign_property, extract_property

from .findings import (
    DEFAULT_MAX_KEPT,
    FINDINGS_PROTOCOL,
    FINDINGS_PROTOCOL_UNJUDGED,
    FindingsJudge,
    FindingsSession,
    JudgeFindings,
    finding_function,
)
from .hooks import AgentHooks, AgentRetryStrategy, Deadline
from .memory import memory_manager
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
        max_seconds: Annotated[
            float | None,
            "Wall-clock budget per item in seconds: once spent, further tool "
            "calls are refused and the agent is told to answer from what it "
            "has (None = no budget; max_turns stays the hard cap)",
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
        judge: Annotated[
            bool | FindingsJudge,
            "Run the findings loop: judge every finding tool's results with "
            "the agent's model and accumulate them per item (from Python, a "
            "FindingsJudge to judge with instead)",
        ] = False,
        findings: Annotated[
            bool,
            "Run the findings loop without a judge: the agent keeps and "
            "ignores findings itself (keep_finding, ignore_finding)",
        ] = False,
        max_kept: Annotated[
            int, "With judge or findings: the most findings one item may keep"
        ] = DEFAULT_MAX_KEPT,
        question_field: Annotated[
            str | None,
            "With judge or findings: property of the item holding the question "
            "findings are judged against (default: the prompt)",
        ] = None,
        findings_as: Annotated[
            str | None,
            "With judge or findings: set the run's FindingsOutcome (kept "
            "findings, how the run ended, the trace) as this property of the item",
        ] = None,
        first_tool: Annotated[
            str | None,
            "With judge or findings: a finding tool to call with the prompt, "
            "as its first argument, before the agent's first turn. Its "
            "findings are judged like any call's; when the judge finds them "
            "enough to answer, the agent is told to answer from them at once "
            "instead of searching, otherwise it starts from them and is told "
            "what is missing",
        ] = None,
        hooks: Annotated[
            list | None,
            "From Python: more Strands hook providers for every agent",
        ] = None,
        memory: Annotated[
            Any,
            "From Python: Strands memory stores (a SearchMemoryStore over a "
            "talkpipe search, say), singly or as a list, whose best matches for "
            "each item are put before the model without a tool call; or a "
            "MemoryManager configured by hand",
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
        # Validates max_seconds now, so a bad value fails when the script compiles.
        self._deadline = (
            Deadline(max_seconds, self.tell) if max_seconds is not None else None
        )
        self.max_seconds = max_seconds
        self.temperature = temperature
        # parsed here so malformed JSON fails when the script compiles
        self.model_params = parse_model_params(model_params)
        self.field = field
        self.set_as = set_as
        self._progress = teller(progress, logger)
        self._default_progress = progress is None
        self.state = AgentState()
        self.last_result: Any = None
        self._agent: Agent | None = None
        self.hooks = list(hooks or [])
        self.memory = memory_manager(memory)
        self.max_kept = max_kept
        self.question_field = question_field
        self.findings_as = findings_as
        self.first_tool = first_tool
        self.last_outcome: Any = None
        self._session: FindingsSession | None = None
        self._loop: JudgeFindings | None = None
        self._first: tuple[Any, Callable[..., Any], str] | None = None
        if judge is not False or findings:
            if judge is True:
                judge = FindingsJudge(
                    model=model,
                    source=source,
                    temperature=temperature,
                    model_params=self.model_params,
                )
            self._loop = JudgeFindings(judge=judge or None)
            self.system_prompt += "\n\n" + (
                FINDINGS_PROTOCOL if self._loop.judge else FINDINGS_PROTOCOL_UNJUDGED
            )
            # The session tools the protocol names, unless already given.
            names = {getattr(t, "tool_name", None) for t in self.tools}
            wanted = ["enough_information"]
            if self._loop.judge is None:
                wanted += ["keep_finding", "ignore_finding"]
            self.tools += resolve_agent_tools([n for n in wanted if n not in names])
            self._first = self._find_first_tool(first_tool) if first_tool else None
        elif first_tool:
            raise ValueError(
                f"llmAgent: first_tool={first_tool!r} needs the findings loop: "
                "set judge=True or findings=True"
            )

    def _find_first_tool(self, name: str) -> tuple[Any, Callable[..., Any], str]:
        """The tool named *name* among this agent's tools (resolved and
        added if it is not there yet), the finding function behind it, and
        the name of its first parameter, which the prompt is passed as.
        Fails now, when the segment is built, for a name that is not a
        finding tool."""
        tool = next(
            (t for t in self.tools if getattr(t, "tool_name", None) == name), None
        )
        if tool is None:
            [tool] = resolve_agent_tools([name])
            self.tools.append(tool)
        func = finding_function(tool)
        if func is None:
            raise ValueError(
                f"llmAgent: first_tool={name!r} is not a finding tool (one made "
                "with @finding_tool or findingTool), so it returns no findings to judge"
            )
        params = [
            p
            for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.name != "tool_context"
        ]
        if not params:
            raise ValueError(
                f"llmAgent: first_tool={name!r} takes no argument to pass the prompt as"
            )
        return tool, func, params[0].name

    def _seed(self, loop: JudgeFindings, session: FindingsSession, prompt: str) -> str:
        """Call the first tool with the question and judge its findings
        before the agent starts; the agent's opening prompt. A tool that
        fails is traced, and the agent starts as if nothing had been
        called."""
        if self._first is None:
            return prompt
        tool, func, param = self._first
        name = tool.tool_name
        try:
            found = func(**{param: session.question})
        except Exception as exc:
            session.tell(f"agent ← {name}: error: {exc}")
            return prompt
        reply = loop.seed(name, found, args={param: session.question})
        return session.opening_prompt(reply, tool=name)

    def tell(self, message: str) -> None:
        """Report *message*: to the running findings session's trace as
        well, when there is one."""
        if self._session is not None:
            self._session.tell(message)
        else:
            self._progress(message)

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
            hooks=[
                AgentHooks(self.tell),
                *([self._deadline] if self._deadline else []),
                *([self._loop] if self._loop else []),
                *self.hooks,
            ],
            retry_strategy=AgentRetryStrategy(),
            state=self.state,
            memory_manager=self.memory,
        )

    def _ask(self, agent: Agent, prompt: str) -> str:
        self.last_result, answer = ask_and_clean(
            agent, prompt, limits=self.limits, tell=self.tell
        )
        stop_reason = str(getattr(self.last_result, "stop_reason", ""))
        if not answer and stop_reason.startswith("limit_") and self._default_progress:
            # An item that yields nothing must not pass unnoticed when no
            # progress channel is shown (the INFO lines are silent by default).
            key = stop_reason.removeprefix("limit_")
            logger.warning(
                "llmAgent: an item yielded an empty answer: the agent stopped at "
                "its %s limit (%s) before answering; raise the limit or ask a "
                "smaller question",
                key,
                (self.limits or {}).get(key),
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

    def _ask_with_findings(
        self, loop: JudgeFindings, agent: Agent, prompt: str, item: Any
    ) -> str:
        """Ask *prompt* with a fresh findings session for this item."""
        question = prompt
        if self.question_field:
            question = extract_property(item, self.question_field, fail_on_missing=True)
        session = self._session = loop.session = FindingsSession(
            str(question),
            max_kept=self.max_kept,
            # The segment's own caps, so the countdown matches the limit
            # and the outcome names the budget the Deadline hook enforces.
            max_turns=self.max_turns,
            max_seconds=self.max_seconds,
            tell=self._progress,
        )
        try:
            answer = self._ask(agent, self._seed(loop, session, prompt))
        except Exception as exc:
            self.last_outcome = session.outcome(error=str(exc) or repr(exc))
            raise
        else:
            stop_reason = str(getattr(self.last_result, "stop_reason", ""))
            self.last_outcome = session.outcome(stop_reason)
        finally:
            self._session = loop.session = None
        if self.findings_as:
            assign_property(item, self.findings_as, self.last_outcome)
        return answer

    def _check_assignable(self, item: Any) -> None:
        # Fail before the model runs, not after: a plain value has nowhere
        # to put set_as / findings_as.
        if not isinstance(item, (str, bytes, int, float, tuple)) and item is not None:
            return
        for name in ("set_as", "findings_as"):
            value = getattr(self, name)
            # findings_as is only used with the findings loop on
            if value and (name == "set_as" or self._loop is not None):
                raise ValueError(
                    f"llmAgent: {name}={value!r} sets a property on the item, "
                    f"but the item is a {type(item).__name__}; pass dict items "
                    '(in chatterlang, e.g. toDict[field_list="_:question"] '
                    'before llmAgent[field="question", ...])'
                )

    def transform(self, input_iter: Iterable) -> Iterator:
        for item in input_iter:
            self._check_assignable(item)
            prompt = self._prompt_of(item)
            if self.multi_turn:
                self._agent = self._agent or self._make_agent()
                agent = self._agent
            else:
                agent = self._make_agent()
            if self._loop is not None:
                answer = self._ask_with_findings(self._loop, agent, str(prompt), item)
            else:
                answer = self._ask(agent, str(prompt))
            if self.set_as:
                assign_property(item, self.set_as, answer)
                yield item
            else:
                yield answer
