"""``AgentHooks`` and ``AgentRetryStrategy`` — the Strands hook providers
both entry points install.

Three concerns that belong at the agent loop rather than around it:

- **progress**: one line per tool call (``agent → name(args…)``), and one
  more when a tool call came back as an error (``agent ← name: error: …``),
  through the ``teller`` convention the surrounding pipelines use, so a
  UI or log shows what the agent is doing — and what failed on it — while
  it works;
- **history hygiene**: raw tool-call syntax that a server failed to parse
  (as seen with gpt-oss calls Ollama could not decode) is scrubbed from
  every assistant message as Strands stores it, so it is never replayed
  into the model's context — where it derails later turns — and never
  reaches the reply either, since the stored message is what the result
  is read from;
- **retrying a tool call the server could not parse**: Ollama rejects a
  model turn whose tool-call JSON is malformed (gpt-oss sometimes stops
  mid-object) with ``error parsing tool call``. That is a sampling
  accident, not a fault in the request, so the model call is retried the
  way Strands retries throttling, instead of ending the whole run.

:class:`Deadline` is the wall-clock budget (``max_seconds``) Strands'
``limits`` lack: once a prompt has run that long, every further tool call
is cancelled with a message telling the model to answer from what it has.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from strands.agent import ModelRetryStrategy
from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

from .support import strip_tool_leak

logger = logging.getLogger(__name__)

_ARGS_CHARS = 120
_ERROR_CHARS = 200
# What Ollama's error says when it cannot decode the model's tool call.
_UNPARSED_TOOL_CALL = "error parsing tool call"


class AgentHooks(HookProvider):
    """Progress lines and history hygiene for a talkpipe-hosted agent."""

    def __init__(self, tell: Callable[[str], None] = logger.info) -> None:
        self._tell = tell

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(MessageAddedEvent, self._message_added)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        args = str(event.tool_use.get("input", ""))
        if len(args) > _ARGS_CHARS:
            args = args[:_ARGS_CHARS] + "…"
        self._tell(f"agent → {event.tool_use.get('name')}({args})")

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        # The model sees the error and usually talks around it; without
        # this line the person watching sees only the confused answer.
        result = event.result
        if result.get("status") != "error":
            return
        text = " ".join(
            str(block.get("text", "")) for block in result.get("content", ())
        ).strip()
        text = text.removeprefix("Error: ").splitlines()[0] if text else "(no detail)"
        if len(text) > _ERROR_CHARS:
            text = text[:_ERROR_CHARS] + "…"
        self._tell(f"agent ← {event.tool_use.get('name')}: error: {text}")

    def _message_added(self, event: MessageAddedEvent) -> None:
        if event.message.get("role") != "assistant":
            return
        for block in event.message.get("content", ()):
            text = block.get("text") if isinstance(block, dict) else None
            if not text:
                continue
            cleaned = strip_tool_leak(text)
            if cleaned != text:
                block["text"] = cleaned
                self._tell("agent: scrubbed leaked tool-call syntax from a reply")


class Deadline(HookProvider):
    """A wall-clock budget per prompt: *max_seconds* after an invocation
    starts, every tool call is cancelled before it runs, and the model
    reads a tool result saying the budget is spent and to answer now
    from what it has. Soft on purpose — the model's turns are still
    bounded by ``max_turns``, the hard cap — so a model that keeps
    calling tools after the deadline gets the same answer each time
    and nothing more is fetched. ``timed_out`` says whether the latest
    invocation hit the budget. *clock* is for tests."""

    def __init__(
        self,
        max_seconds: float,
        tell: Callable[[str], None] = logger.info,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
            raise ValueError(f"max_seconds must be a number, got {max_seconds!r}")
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds!r}")
        self.max_seconds = max_seconds
        self._tell = tell
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self.timed_out = False

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._start)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    def _start(self, event: BeforeInvocationEvent) -> None:
        self._started = self._clock()
        self.timed_out = False

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        elapsed = self.elapsed
        if elapsed < self.max_seconds:
            return
        if not self.timed_out:
            self.timed_out = True
            self._tell(f"agent: stopped at the time budget ({self.max_seconds:g} s)")
        event.cancel_tool = (
            f"The {self.max_seconds:g}-second time budget is spent ({elapsed:.0f} s "
            "used): stop calling tools and answer now from what you have."
        )


class AgentRetryStrategy(ModelRetryStrategy):
    """Strands' default model-call retry (throttling, same backoff and
    attempt count), extended to a tool call the server failed to parse."""

    def is_retryable(self, exception: Exception) -> bool:
        return super().is_retryable(exception) or _UNPARSED_TOOL_CALL in str(exception)
