"""Running one prompt: budget, retry, and reply cleanup.

``ask_and_clean`` runs one prompt through a Strands agent under a
per-invocation budget (Strands ``limits``) and returns display-ready
text. Two failure modes observed with real local models are handled
here rather than shown to the user: reasoning-model
``<think>`` blocks are stripped, and a reply with no user-visible text
at all (only reasoning, or only a mangled call) is retried once with a
nudge to answer in plain prose instead of surfacing as an empty bubble.
Leaked tool-call syntax is scrubbed earlier, as Strands stores each
message, by :class:`~talkpipe_agents.hooks.AgentHooks`.

``agent_limits`` builds the budget from the three caps every entry
point exposes: ``max_turns`` (model calls per prompt — a turn may hold
several tool calls; the one hard, provider-independent cap) and the
soft token budgets ``max_output_tokens`` / ``max_total_tokens``, which
depend on the provider reporting usage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from strands.types.agent import Limits

from .support import strip_think, strip_tool_leak

logger = logging.getLogger(__name__)

RETRY_NUDGE = (
    "Your previous reply contained no user-visible text. Answer the "
    "user's last question now in plain prose, using the tool results "
    "you already have."
)


def _cap(name: str, value: Any) -> int | None:
    """*value* checked as a cap: ``None`` (no cap) or a positive integer,
    named as the caller wrote it in the error (not as Strands' key)."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return value


def agent_limits(
    max_turns: int | None = None,
    max_output_tokens: int | None = None,
    max_total_tokens: int | None = None,
) -> Limits | None:
    """The Strands ``limits`` for the caps given (``None`` = no cap);
    ``None`` when nothing is capped. A cap that is not ``None`` or a
    positive integer raises ``ValueError`` naming the parameter."""
    limits: Limits = {}
    if (turns := _cap("max_turns", max_turns)) is not None:
        limits["turns"] = turns
    if (output := _cap("max_output_tokens", max_output_tokens)) is not None:
        limits["output_tokens"] = output
    if (total := _cap("max_total_tokens", max_total_tokens)) is not None:
        limits["total_tokens"] = total
    return limits or None


def _text_of(result: Any) -> str:
    """The assistant text of *result*. ``str(result)`` is that text
    except when a tool raised a Strands interrupt (human input required):
    then it is the interrupt list itself, which must never be handed
    downstream as an answer."""
    if not getattr(result, "interrupts", None):
        return str(result)
    content = (getattr(result, "message", None) or {}).get("content", [])
    return "".join(block["text"] for block in content if "text" in block)


def ask_and_clean(
    agent: Any,
    prompt: Any,
    *,
    limits: Limits | None = None,
    tell: Callable[[str], None] = logger.info,
) -> tuple[Any, str]:
    """Run *prompt* through *agent*: the agent result and clean answer
    text (which is empty when even the retry produced nothing — the
    caller decides what an empty answer means in its context). *prompt*
    is the user's text, or whatever else the Strands agent accepts (the
    interrupt responses that resume a paused agent)."""

    def one_round(text: Any) -> tuple[Any, str]:
        result = agent(text, limits=limits)
        stop_reason = str(result.stop_reason)
        if stop_reason.startswith("limit_"):  # limit_turns, limit_total_tokens, …
            key = stop_reason.removeprefix("limit_")
            tell(f"agent: stopped at the {key} limit ({(limits or {}).get(key)})")
        for interrupt in getattr(result, "interrupts", None) or ():
            tell(
                f"agent: paused for human input by {interrupt.name} ({interrupt.reason})"
            )
        return result, strip_think(strip_tool_leak(_text_of(result)))

    result, answer = one_round(prompt)
    # No retry after a limit (the budget is spent) or an interrupt (the
    # agent is waiting for a human, not for another prompt).
    stop_reason = str(result.stop_reason)
    if (
        not answer
        and stop_reason != "interrupt"
        and not stop_reason.startswith("limit")
    ):
        tell(
            "agent: the reply had no user-visible text; asking for a plain-prose answer"
        )
        result, answer = one_round(RETRY_NUDGE)
    return result, answer
