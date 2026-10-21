"""``AgentHooks`` — the Strands hook provider both entry points install.

Two concerns that belong at the agent loop rather than around it:

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
  is read from.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

from .support import strip_tool_leak

logger = logging.getLogger(__name__)

_ARGS_CHARS = 120
_ERROR_CHARS = 200


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
