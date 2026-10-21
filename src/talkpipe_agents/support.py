"""Small shared helpers: progress convention and reply cleanup.

``teller`` is the one-line-progress-message convention shared with
talkpipe-style segments. ``strip_think`` removes reasoning-model
``<think>`` blocks and ``strip_tool_leak`` removes raw tool-call syntax
that escaped the server's parser into reply text — neither is ever
meant for display.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any


def teller(
    progress: Callable[[str], None] | None, logger: logging.Logger
) -> Callable[[str], None]:
    """The progress convention used across segments: a one-line-message
    callable, defaulting to the caller's logger."""
    return progress or (lambda msg: logger.info("%s", msg))


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning-model <think> blocks (never meant for display)."""
    cleaned = _THINK_RE.sub("", text)
    # an unterminated block means the model was cut off mid-thought
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


_HARMONY_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")
_TOOL_LEAK_RE = re.compile(
    # a stray glyph a mangled token leaves behind (not prose punctuation)
    r"[^\s\w.,;:!?()'\"\-]{0,2}[ \t]*"
    r"(?:<\|[a-z_]+\|>[ \t]*|\b(?:commentary|analysis|final|assistant)\b[ \t]*)*"
    r"to=functions(?:\.[\w\-]+)?"
    r"(?:[ \t]*<\|[a-z_]+\|>|[ \t]*\bjson\b)*[ \t]*"
)


_ARG_KEYS = ("parameters", "arguments", "args", "input")
_CALL_KEYS = {"name", *_ARG_KEYS}


def _is_args(value: Any) -> bool:
    """An argument object, decoded or still JSON-encoded as a string."""
    return isinstance(value, dict) or (
        isinstance(value, str) and value.lstrip().startswith("{")
    )


def _is_tool_call(obj: Any) -> bool:
    """Whether *obj* is a decoded JSON tool call rather than data.

    Deliberately narrow: a name, at least one recognized argument
    object, and nothing else — so an answer that merely happens to be
    JSON (``{"name": "Ada", "age": 3}``, ``{"name": "Ada"}``) is left
    alone.
    """
    if not isinstance(obj, dict):
        return False
    keys = set(obj)
    if keys <= {"type", "id", "function"} and "function" in keys:
        return _is_tool_call(obj["function"])  # OpenAI-style envelope
    return (
        keys <= _CALL_KEYS
        and isinstance(obj.get("name"), str)
        and any(_is_args(obj.get(k)) for k in _ARG_KEYS)
    )


def _strip_json_calls(text: str) -> str:
    """Remove JSON tool-call objects a model emitted as plain text."""
    decoder = json.JSONDecoder()
    at = 0
    while (start := text.find("{", at)) >= 0:
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except ValueError:
            at = start + 1
            continue
        if _is_tool_call(obj):
            text = text[:start] + text[start + consumed :]
            at = start
        else:
            at = start + consumed
    return text


def strip_tool_leak(text: str) -> str:
    """Remove raw tool-call syntax that leaked into reply text.

    Two shapes are recognized: the harmony format (gpt-oss family), and
    a bare JSON tool-call object emitted as prose (``{"name": ...,
    "parameters": {...}}``, as seen from the llama family), which a
    model produces when it means to call a tool but does not use the
    provider's tool-call channel. Other text passes through unchanged.
    Surrounding text is preserved exactly (no outer strip), so callers
    can compare against the input to detect that a leak was removed —
    and a reply that was nothing but a leak becomes empty, which the
    caller retries for plain prose.
    """
    decoder = json.JSONDecoder()
    while True:
        match = _TOOL_LEAK_RE.search(text)
        if not match:
            break
        end = match.end()
        # swallow the call's JSON arguments, when present
        if text[end : end + 1] in "[{":
            try:
                _, consumed = decoder.raw_decode(text[end:])
                end += consumed
            except ValueError:
                pass
        text = text[: match.start()] + text[end:]
    return _strip_json_calls(_HARMONY_TOKEN_RE.sub("", text))
