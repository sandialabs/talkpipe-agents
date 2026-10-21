"""Test an agent offline: a scripted Strands ``Model`` as a model source.

An application that hands its own tools to ``llmAgent`` or ``AgentChat``
wants to test that wiring without a model server. ``ScriptedModel`` is
a Strands ``Model`` that replays turns you script — tool calls and text
answers — through the real agent loop, so the real tool functions run,
the real hygiene and limits apply, and only the model is fake.

``scripted_source`` registers one instance as an agent model source (by
default ``"scripted"``) for the duration of a ``with`` block, so a
script or segment that says ``source="scripted"`` gets it::

    from talkpipe_agents import LLMAgent
    from talkpipe_agents.testing import scripted_source

    with scripted_source() as model:
        model.call("lookup_ticket", ticket_id="T-1").say("Not urgent.")
        step = LLMAgent(source="scripted", model="any", tools=["lookup_ticket"])
        assert list(step(["How urgent is T-1?"])) == ["Not urgent."]
        assert model.tool_results[0][:2] == ("lookup_ticket", "success")

Under pytest the ``scripted_model`` fixture (installed with this
package as a pytest plugin) does the ``with`` for you, one fresh model
per test.

The same instance is handed to every agent built while the source is
registered, so a segment that builds a fresh agent per item still
consumes one script in order. ``tool_results`` records what each
scripted call came back with, so a test proves the real tool ran;
``calls`` records the message count seen by each model turn and
``prompts`` the full messages, for asserting on what the agent
actually sent; ``remaining`` is the script not yet played.

Two mistakes a test author makes are caught rather than masked by the
scripted answer that follows: a scripted call to a tool the agent was
not given fails the turn outright, and a call the agent reports back as
an error (wrong argument names, a tool that raised) is emitted as a
``ScriptedToolErrorWarning`` — pytest shows it even when the test
passes; a test of error recovery silences it with ``pytest.warns``.
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any, Literal, NamedTuple

from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from . import models

TextTurn = tuple[Literal["text"], str]
ToolTurn = tuple[Literal["tool"], str, dict[str, Any]]
Turn = TextTurn | ToolTurn
"""One scripted model turn: ``("text", answer)`` or ``("tool", name, args)``."""


class ToolResult(NamedTuple):
    """What one scripted tool call came back with, as the agent reported
    it to the model: the tool's name, ``"success"`` or ``"error"``, and
    the result text. A plain tuple compares equal, so
    ``assert model.tool_results == [("add", "success", "5.0")]`` reads well."""

    name: str
    status: str
    text: str


class ScriptedToolErrorWarning(UserWarning):
    """A scripted tool call came back to the model as an error — usually
    the script named the wrong arguments, or the tool raised. The agent
    carries on with the next scripted turn, so without this a test would
    pass with its tool never having run."""


class ScriptedModel(Model):
    """A Strands ``Model`` that replays scripted turns instead of calling
    a model server. Build the script with :meth:`say` and :meth:`call`
    (or pass turns to the constructor); each model turn pops the next
    one. Running out of script raises ``IndexError`` — the agent asked
    for more turns than the test planned — and a scripted call to a tool
    the agent does not have raises ``ValueError`` naming the tools it
    does have. Strands lets an error from the first model call through
    as-is and wraps one from a later turn in an ``EventLoopException``;
    the message is the same either way.

    ``tool_results`` holds a :class:`ToolResult` for each scripted call
    once the agent has run it; a call that came back as an error is
    also a :class:`ScriptedToolErrorWarning`. ``calls`` holds the number
    of messages the agent sent on each turn and ``prompts`` the messages
    themselves, so a test can assert on conversation shape (a tool call
    followed by an answer is two turns with message counts ``[1, 3]``:
    the question; then the question, the tool use, and the tool result).
    """

    def __init__(self, *turns: Turn) -> None:
        self.script: deque[Turn] = deque(turns)
        self.calls: list[int] = []
        self.prompts: list[Messages] = []
        self.tool_results: list[ToolResult] = []
        self._awaiting: dict[str, str] = {}  # toolUseId -> tool name, until its result

    def say(self, text: str) -> ScriptedModel:
        """Script a text reply (the model answers and the turn ends)."""
        self.script.append(("text", text))
        return self

    def call(self, tool_name: str, /, **arguments: Any) -> ScriptedModel:
        """Script a tool call; the agent runs the real tool and feeds the
        result back for the next scripted turn."""
        self.script.append(("tool", tool_name, dict(arguments)))
        return self

    @property
    def remaining(self) -> tuple[Turn, ...]:
        """The scripted turns not yet played, in order (empty when the
        agent used every turn — ``assert not model.remaining``)."""
        return tuple(self.script)

    def reset(self) -> None:
        """Drop the remaining script and everything recorded."""
        self.script.clear()
        self.calls.clear()
        self.prompts.clear()
        self.tool_results.clear()
        self._awaiting.clear()

    def _record_tool_results(self, messages: Messages) -> list[ToolResult]:
        """Note the outcome of each scripted call whose result the agent
        has now sent back; return the ones new since the last turn."""
        new: list[ToolResult] = []
        for message in messages:
            for block in message.get("content", []):
                result = block.get("toolResult")
                if result is None:
                    continue
                name = self._awaiting.pop(result.get("toolUseId", ""), None)
                if name is None:
                    continue
                text = "\n".join(
                    part["text"] if "text" in part else json.dumps(part.get("json"))
                    for part in result.get("content", [])
                    if "text" in part or "json" in part
                )
                outcome = ToolResult(name, result.get("status", "success"), text)
                self.tool_results.append(outcome)
                new.append(outcome)
                if outcome.status == "error":
                    warnings.warn(
                        f"scripted call to tool {name!r} came back as an error: {text}",
                        ScriptedToolErrorWarning,
                        stacklevel=2,
                    )
        return new

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(len(messages))
        self.prompts.append(messages)
        just_returned = self._record_tool_results(messages)
        try:
            turn = self.script.popleft()
        except IndexError:
            where = (
                f" after {just_returned[-1].name!r} returned" if just_returned else ""
            )
            raise IndexError(
                f"scripted model has no turn left for model call {len(self.calls)}"
                f"{where} (script more turns with .say()/.call())"
            ) from None
        if turn[0] == "tool":
            # Checked before any event is streamed, so the failure is this
            # error and not a half-built message. Strands passes None, not
            # [], for an agent with no tools.
            available = sorted(spec["name"] for spec in tool_specs or [])
            if turn[1] not in available:
                has = (
                    f"has only: {', '.join(available)}" if available else "has no tools"
                )
                raise ValueError(
                    f"scripted call to tool {turn[1]!r}, but the agent {has} "
                    "(misspelled, or missing from its tools=[...]?)"
                )
        yield {"messageStart": {"role": "assistant"}}
        if turn[0] == "text":
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": turn[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            _, name, args = turn
            tool_use_id = f"scripted-{len(self.calls)}"
            self._awaiting[tool_use_id] = name
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"name": name, "toolUseId": tool_use_id}}
                }
            }
            yield {
                "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}

    async def structured_output(
        self,
        output_model: type[Any],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError("the scripted model is chat-only")
        # A yield, even unreachable, is what makes this an async generator
        # like the interface it implements.
        yield  # type: ignore[unreachable]

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}


@contextmanager
def scripted_source(
    name: str = "scripted", model: ScriptedModel | None = None
) -> Iterator[ScriptedModel]:
    """Register *model* (a fresh ``ScriptedModel`` by default) as agent
    model source *name* for the block, then restore whatever *name*
    meant before. Every agent built with ``source=name`` inside the
    block shares the one instance, whatever ``model=`` string it names.
    """
    scripted = model if model is not None else ScriptedModel()
    previous = models._factories.get(name)
    models.register_agent_model(name, lambda _model, **_kwargs: scripted)
    try:
        yield scripted
    finally:
        if previous is None:
            models._factories.pop(name, None)
        else:
            models._factories[name] = previous


__all__ = [
    "ScriptedModel",
    "ScriptedToolErrorWarning",
    "ToolResult",
    "Turn",
    "scripted_source",
]
