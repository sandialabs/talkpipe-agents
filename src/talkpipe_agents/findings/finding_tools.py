"""Finding tools and the session tools.

:func:`finding_tool` makes any function returning findings an agent
tool; ``enough_information``, ``keep_finding`` and ``ignore_finding``
are the registered tools that act on the running session; the
``FINDINGS_PROTOCOL`` texts tell the model how the loop works.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from strands import tool as strands_tool
from strands.types.tools import ToolContext

from ..tools import register_agent_tool
from .finding import FoundFindings, as_findings, below_score_floor, render_findings
from .hook import FINDINGS_KEY, SESSION_KEY, session_of

logger = logging.getLogger(__name__)


# -- finding tools -----------------------------------------------------------


def finding_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    register: bool = True,
    min_score: float | None = None,
) -> Any:
    """Make *func* a finding tool: an agent tool whose result is findings.

    *func* is an ordinary function — its docstring describes the tool to
    the model, its type hints give the schema — that returns zero or
    more findings: :class:`Finding` objects, strings, dicts with
    ``text``/``label``/``source``, or search results (see
    :func:`as_findings`); plain values are labelled with the tool's
    name. It knows nothing of sessions or judges. Under
    :class:`JudgeFindings` (``llmAgent[judge=True]``) its findings are
    judged and accumulated; anywhere else the model reads them as text.
    Once the session's run is over the call is refused without running
    *func*. By convention a finding tool may take ``looking_for`` (what
    this call should find that the answer still lacks); the judge is
    told it.

    *name* is the tool's name (default: the function's), *description*
    overrides the docstring's. The tool is registered with
    ``register_agent_tool`` unless *register* is false. *min_score* is
    a floor on the findings' scores (:attr:`Finding.search_score`: a
    search hit's similarity, higher is better): results under it are left out
    before the judge sees them, and the model is told how many — so a
    call with nothing above the floor costs no judge call. Findings
    with no score always pass; the scale is the tool's, so the floor
    belongs with the tool. Works bare or with arguments::

        @finding_tool
        def search_notes(query: str, looking_for: str = "") -> list[Finding]:
            \"\"\"Search the notes ...\"\"\"
    """
    if func is None:

        def decorate(f: Callable[..., Any]) -> Any:
            return finding_tool(
                f,
                name=name,
                description=description,
                register=register,
                min_score=min_score,
            )

        return decorate
    tool_name = name or func.__name__
    signature = inspect.signature(func)
    takes_context = "tool_context" in signature.parameters

    def run(**kwargs: Any) -> str:
        context = (
            kwargs.get("tool_context")
            if takes_context
            else kwargs.pop("tool_context", None)
        )
        state = getattr(context, "invocation_state", None)
        session = state.get(SESSION_KEY) if isinstance(state, dict) else None
        if session is not None and (refusal := session.guard(tool_name)):
            return str(refusal)
        found, below_floor = below_score_floor(
            as_findings(func(**kwargs), label=tool_name), min_score
        )
        if isinstance(state, dict) and context is not None:
            stash = state.setdefault(FINDINGS_KEY, {})
            stash[str(context.tool_use.get("toolUseId"))] = FoundFindings(
                found, below_floor
            )
        return render_findings(found, below_floor=below_floor)

    # The schema Strands builds is func's, plus the context it injects.
    params = list(signature.parameters.values())
    if not takes_context:
        params.append(
            inspect.Parameter(
                "tool_context",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=ToolContext,
            )
        )
    run.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=params, return_annotation=str
    )
    run.__annotations__ = {
        **getattr(func, "__annotations__", {}),
        "tool_context": ToolContext,
        "return": str,
    }
    run.__name__ = func.__name__
    run.__qualname__ = func.__qualname__
    run.__module__ = func.__module__
    run.__doc__ = func.__doc__
    # Lets get_type_hints resolve func's string annotations in its module.
    run.__wrapped__ = func  # type: ignore[attr-defined]
    if not (description or func.__doc__):
        logger.warning(
            "finding tool %r has no docstring; the model is told nothing about "
            "what it does or when to call it",
            tool_name,
        )
    tool_obj = strands_tool(name=tool_name, description=description, context=True)(run)
    # The function itself, for a host that calls the tool outside the agent
    # loop (JudgeFindings.seed, llmAgent's first_tool) and wants findings,
    # not the rendered text the model reads.
    tool_obj.finding_function = func  # type: ignore[attr-defined]  # a plain attribute on the tool object
    if register:
        register_agent_tool(tool_obj)
    return tool_obj


def finding_function(tool: Any) -> Callable[..., Any] | None:
    """The function behind a finding tool — what :func:`finding_tool`
    wrapped, returning findings — or ``None`` for any other tool."""
    return getattr(tool, "finding_function", None)


# -- the session tools -------------------------------------------------------


def _no_session(name: str) -> str:
    return (
        f"{name} only works while a findings session runs "
        "(llmAgent with judge=True or findings=True)."
    )


@register_agent_tool
@strands_tool(context=True)
def enough_information(
    reason: str = "", tool_context: ToolContext | None = None
) -> str:
    """Declare that the findings kept so far would let you answer the
    question well, and end the search. This is how a run should end:
    call it as soon as that is true, however many findings that took.
    Call it also when repeated calls find nothing new and the question
    still cannot be fully answered, saying what is missing.

    Args:
        reason: One short phrase saying what the kept findings cover,
            or what is still missing.
    """
    session = session_of(tool_context)
    if session is None:
        return _no_session("enough_information")
    return session.guard("enough_information") or session.declare_enough(reason)


@register_agent_tool
@strands_tool(context=True)
def keep_finding(
    finding_id: str, reason: str = "", tool_context: ToolContext | None = None
) -> str:
    """Keep a finding: it holds information that helps answer the
    question, even if only part of it. Kept findings are what the
    question will be answered from.

    Args:
        finding_id: The id shown with the finding, e.g. f3.
        reason: One short phrase saying what it contributes.
    """
    session = session_of(tool_context)
    if session is None:
        return _no_session("keep_finding")
    return session.guard("keep_finding") or session.decide(
        finding_id, reason, keep=True
    )


@register_agent_tool
@strands_tool(context=True)
def ignore_finding(
    finding_id: str, reason: str = "", tool_context: ToolContext | None = None
) -> str:
    """Ignore a finding: it is about something other than the question,
    so nothing in it would help the answer.

    Args:
        finding_id: The id shown with the finding, e.g. f3.
        reason: One short phrase saying why it does not help.
    """
    session = session_of(tool_context)
    if session is None:
        return _no_session("ignore_finding")
    return session.guard("ignore_finding") or session.decide(
        finding_id, reason, keep=False
    )


# What llmAgent appends to the system prompt when it runs a session, so
# the model knows the protocol whatever its own instructions say.
FINDINGS_PROTOCOL = """How to gather information: call your tools to find what answers the question. Each call's findings are judged against the question as they arrive: the relevant ones are kept, the rest ignored, and the reply shows the kept findings in full, what is already known, and how many turns you have left.
When a tool takes looking_for, say in one phrase what this call should find that the answer still lacks, so a finding supplying just that is kept; leave it empty on the first call. After each reply, ask what the question still needs and look for that, with different terms if a call kept nothing. Do not repeat a call you already made.
Call enough_information as soon as the kept findings would let you answer the question well - that judgement, not the number of findings, is the reason to stop - then answer the question from the kept findings, citing their sources. If nothing relevant is found, say so instead of guessing. When a reply says the next turn is your last, call enough_information at once."""

FINDINGS_PROTOCOL_UNJUDGED = """How to gather information: call your tools to find what answers the question. Each call's findings are shown with ids (f1, f2, ...): keep the ones that help answer the question with keep_finding and drop the rest with ignore_finding; replies say how many turns you have left.
After each call, ask what the question still needs and look for that, with different terms if nothing helped. Do not repeat a call you already made.
Call enough_information as soon as the kept findings would let you answer the question well, then answer the question from the kept findings, citing their sources. If nothing relevant is found, say so instead of guessing. When a reply says the next turn is your last, call enough_information at once."""
