"""``ThinkingOllamaModel`` — Strands' ``OllamaModel`` without its lossy
handling of reasoning and tool calls.

Strands' provider (as of strands-agents 1.61) drops what a model reasons
between tool calls and splits each turn apart on the way back:

- it never reads ``message.thinking`` from the stream, so the reasoning
  is neither stored on the assistant message nor sent back — a model that
  decides on a plan in its reasoning has forgotten the plan by its next
  turn (minimax kept searching until the turn limit instead of keeping
  what it had found);
- it sends each ``toolUse`` block as an assistant message of its own, so
  a turn with text and two parallel calls goes back as three messages;
- it sends tool results without ``tool_name``, so they pair with their
  calls by position only.

This subclass streams the reasoning as Strands ``reasoningContent``
(Strands stores it on the assistant message), sends a turn back as one
assistant message carrying its text, ``thinking`` and every tool call,
and names each tool result — Ollama's documented message format. The
replay of ``thinking`` can be switched off (``replay_thinking=False``,
or ``"replay_thinking": false`` in the ``ollama`` source's
``model_params``) for a model that reacts badly to it; a model that does
not think returns no reasoning, so there is nothing to replay. Worth
offering to Strands upstream; this package carries it until then.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import ollama
from strands.models._validation import warn_on_tool_choice_not_supported
from strands.models.ollama import OllamaModel
from strands.types.content import ContentBlock, Messages
from strands.types.exceptions import ContextWindowOverflowException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


class ThinkingOllamaModel(OllamaModel):
    """Strands' ``OllamaModel`` that keeps a turn's reasoning and sends
    the turn back whole (see the module docstring)."""

    def __init__(
        self, host: str | None, *, replay_thinking: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(host, **kwargs)
        self.replay_thinking = replay_thinking

    def _format_request_messages(
        self, messages: Messages, system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        formatted = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        tool_names: dict[str, str] = {}  # toolUseId → name, for the results
        for message in messages:
            if message["role"] == "assistant":
                formatted.extend(self._format_turn(message["content"], tool_names))
                continue
            for content in message["content"]:
                parts = super()._format_request_messages(
                    [{"role": message["role"], "content": [content]}]
                )
                name = tool_names.get(
                    content.get("toolResult", {}).get("toolUseId", "")
                )
                for part in parts:
                    if name and part["role"] == "tool":
                        part["tool_name"] = name
                formatted.extend(parts)
        return formatted

    def _format_turn(
        self, contents: list[ContentBlock], tool_names: dict[str, str]
    ) -> list[dict[str, Any]]:
        """One assistant turn as a single Ollama message (plus, as before,
        a message per block of any other kind, such as an image)."""
        texts: list[str] = []
        thinking: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for content in contents:
            if "reasoningContent" in content:
                text = content["reasoningContent"].get("reasoningText", {}).get("text")
                if text and self.replay_thinking:
                    thinking.append(text)
            elif "toolUse" in content:
                tool_use = content["toolUse"]
                tool_names[tool_use["toolUseId"]] = tool_use["name"]
                tool_calls.append(
                    {
                        "function": {
                            "name": tool_use["name"],
                            "arguments": tool_use["input"],
                        }
                    }
                )
            elif "text" in content:
                texts.append(content["text"])
            else:
                rest.extend(
                    super()._format_request_messages(
                        [{"role": "assistant", "content": [content]}]
                    )
                )
        turn: dict[str, Any] = {"role": "assistant"}
        if texts:
            turn["content"] = "\n".join(texts)
        if thinking:
            turn["thinking"] = "\n".join(thinking)
        if tool_calls:
            turn["tool_calls"] = tool_calls
        return ([turn] if len(turn) > 1 else []) + rest

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        # OllamaModel.stream, plus the reasoning as reasoningContent deltas.
        warn_on_tool_choice_not_supported(tool_choice)
        request = self.format_request(messages, tool_specs, system_prompt)
        tool_requested = False
        event = None
        client = ollama.AsyncClient(self.host, **self.client_args)
        try:
            response = await client.chat(**request)
            yield self.format_chunk({"chunk_type": "message_start"})
            yield self.format_chunk(
                {"chunk_type": "content_start", "data_type": "text"}
            )
            async for event in response:
                if event.message.thinking:
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "reasoningContent": {"text": event.message.thinking}
                            }
                        }
                    }
                for tool_call in event.message.tool_calls or []:
                    for chunk_type in (
                        "content_start",
                        "content_delta",
                        "content_stop",
                    ):
                        yield self.format_chunk(
                            {
                                "chunk_type": chunk_type,
                                "data_type": "tool",
                                "data": tool_call,
                            }
                        )
                    tool_requested = True
                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "text",
                        "data": event.message.content,
                    }
                )
        except ollama.ResponseError as error:
            if any(text in str(error).lower() for text in self.OVERFLOW_MESSAGES):
                raise ContextWindowOverflowException(str(error)) from error
            raise

        stop_reason = (
            "tool_use" if tool_requested else (event.done_reason if event else None)
        )
        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "message_stop", "data": stop_reason})
        if event is not None:
            yield self.format_chunk({"chunk_type": "metadata", "data": event})
