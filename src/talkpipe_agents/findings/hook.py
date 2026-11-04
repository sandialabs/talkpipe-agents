"""The findings loop as a Strands hook: :class:`JudgeFindings`.

Installed on an agent, it hands the running session to the tools
through Strands' invocation state, and after every finding tool's call
(before the model sees the result) runs the call's findings through the
session and the judge, replacing the result with the session's reply.
"""

from __future__ import annotations

import asyncio
from typing import Any

from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from .finding import Finding, FoundFindings, as_findings, render_findings
from .judge import FindingsJudge
from .session import JUDGE_ENOUGH_REASON, FindingsSession

# Where the loop keeps its per-invocation state in Strands'
# invocation_state, which the tools (through their ToolContext) and the
# hook events share: the running session, and each finding tool call's
# findings by toolUseId until the hook judges them.
SESSION_KEY = "talkpipe_agents.findings_session"
FINDINGS_KEY = "talkpipe_agents.findings"


def session_of(tool_context: Any) -> FindingsSession | None:
    """The findings session running the tool call made with
    *tool_context*, if any: the :class:`JudgeFindings` hook hands it to
    the tools through Strands' invocation state. For a tool that needs
    the session (to look a finding up by id, to trace a line, to check
    the stop) and is not a finding tool."""
    state = getattr(tool_context, "invocation_state", None)
    session = state.get(SESSION_KEY) if isinstance(state, dict) else None
    return session if isinstance(session, FindingsSession) else None


# -- the hook ----------------------------------------------------------------


class JudgeFindings(HookProvider):
    """The findings loop as a Strands hook provider.

    Before every tool call it hands the tools the running ``session``
    (through Strands' ``invocation_state``) and records how many turns
    the agent has used. After a :func:`finding_tool` call it takes the
    call's findings and, with a ``judge``, judges them — skipping those
    already decided, against the question, the call's ``looking_for``
    and what is already known — keeps and ignores them in order. At
    ``max_kept`` a finding graded above the weakest kept one displaces
    it (the displaced finding is ignored as ``displaced by <id>``); the
    rest of the relevant ones are passed over. The result the model
    sees is replaced with :meth:`FindingsSession.reply`. Without a
    judge the findings are registered and shown with their ids for the
    agent to keep or ignore by hand. Every other tool's reply gets the
    turn countdown appended. With ``session`` unset the hook does
    nothing, so one hook can serve an agent across several questions
    (``llmAgent`` sets a fresh session per item).
    """

    def __init__(
        self, session: FindingsSession | None = None, judge: FindingsJudge | None = None
    ) -> None:
        self.session = session
        self.judge = judge

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        session = self.session
        if session is None:
            return
        event.invocation_state[SESSION_KEY] = session
        session.note_turn_of(event.agent)

    async def _after_tool(self, event: AfterToolCallEvent) -> None:
        session = self.session
        if session is None:
            return
        stash = event.invocation_state.get(FINDINGS_KEY) or {}
        found = stash.pop(str(event.tool_use.get("toolUseId")), None)
        result = event.result
        if found is None or result.get("status") == "error":
            if result.get("status") != "error":
                event.result = _with_note(result, session.turns_note())
            return
        if not isinstance(found, FoundFindings):  # a bare list of findings
            found = FoundFindings(list(found))
        args = event.tool_use.get("input") or {}
        looking_for = (
            str(args.get("looking_for") or "") if isinstance(args, dict) else ""
        )
        # Off the event loop: the judge makes a blocking model call.
        text = await asyncio.to_thread(
            self.process,
            found.findings,
            looking_for,
            str(event.tool_use.get("name", "")),
            found.below_floor,
        )
        event.result = {**result, "content": [{"text": text}]}

    def seed(
        self,
        tool: str,
        findings: Any,
        *,
        args: dict[str, Any] | None = None,
        looking_for: str = "",
        reason: str = JUDGE_ENOUGH_REASON,
    ) -> str:
        """Judge a call made before the agent starts, and end the run on
        the judge's verdict when it can.

        *findings* is what the tool named *tool* returned when the host
        called it with *args* (traced as the agent's own call would be);
        they go through the session and the judge exactly as a call from
        inside the agent loop does — one judge call, the findings kept or
        ignored, the same reply text. When the judge's verdict is that
        what is kept answers the question, the session is marked as
        having enough (:meth:`FindingsSession.accept_verdict`, with
        *reason*) and the agent need never take a turn: a question the
        first call answers costs one call and one judgement. Returns the
        reply, for the agent's opening prompt
        (:meth:`FindingsSession.opening_prompt`) otherwise. Does nothing
        beyond rendering the findings without a session."""
        session = self.session
        if session is None:
            return render_findings(as_findings(findings, tool))
        shown = f"({args!r})" if args is not None else "()"
        session.tell(f"agent → {tool}{shown} — before the agent")
        reply = self.process(as_findings(findings, tool), looking_for, tool)
        session.accept_verdict(reason)
        return reply

    def process(
        self,
        findings: list[Finding],
        looking_for: str = "",
        tool: str = "",
        below_floor: int = 0,
    ) -> str:
        """Run one call's *findings* through the session (and the judge,
        if any) and return the reply the model reads. *below_floor* is
        how many results the tool's score floor left out: they are
        reported, not judged — with nothing above the floor the judge is
        not called at all."""
        session = self.session
        if session is None:
            return render_findings(findings, below_floor=below_floor)
        with session.lock:
            session.calls += 1
            floor = f", {below_floor} below the score floor" if below_floor else ""
            if self.judge is None:
                fids = list(dict.fromkeys(session.register(f) for f in findings))
                fresh = sum(1 for fid in fids if not session.is_decided(fid))
                session.tell(
                    f"agent ← {tool}: {len(fids)} finding(s), {fresh} undecided{floor}. "
                    + session.status_line()
                )
                return session.show_undecided(fids, what=tool, below_floor=below_floor)
            judgements = self.judge.judge(
                findings,
                session.question,
                looking_for,
                session.known_notes(),
                session.seen_keys(),
                # Passed only when there are any, so a judge subclass with
                # the older signature still works until a list exists.
                **({"needs": session.needs} if session.needs else {}),
            )
            kept: list[str] = []
            ignored: list[str] = []
            displaced: list[str] = []
            passed_over = 0
            # The judge's label for each finding -> the session's id, so a
            # need's "found" can be traced to the finding it names.
            labels: dict[str, str] = {}
            for judgement in judgements:
                weakest = None
                if judgement.keep and session.at_capacity:
                    weakest = session.displaceable_by(judgement.score)
                    if weakest is None:
                        # Not registered either, so it is not left undecided.
                        passed_over += 1
                        continue
                fid = session.register(judgement.finding)
                if judgement.label:
                    labels[judgement.label] = fid
                if session.is_decided(fid):
                    continue
                if weakest is not None:
                    session.apply(weakest, False, f"displaced by {fid}")
                    displaced.append(weakest)
                session.apply(fid, judgement.keep, judgement.reason, judgement.score)
                (kept if judgement.keep else ignored).append(fid)
            # The verdict: derived from the judge's list of needs when it gave
            # one, else as the judge said. A judge subclass may return a plain
            # list: then no verdict.
            needs = getattr(judgements, "needs", None)
            if needs is not None:
                session.note_needs(needs, labels)
            else:
                session.note_verdict(
                    getattr(judgements, "enough", None),
                    str(getattr(judgements, "missing", "") or ""),
                )
            enough = (
                session.sufficient
                if (needs is not None)
                else getattr(judgements, "enough", None)
            )
            summary = session.needs_summary()
            session.tell(
                f"agent ← {tool}: {len(kept) + len(ignored) + passed_over} "
                f"finding(s), kept {len(kept)}, ignored {len(ignored)}"
                + (f", {len(displaced)} displaced" if displaced else "")
                + (f", {passed_over} passed over" if passed_over else "")
                + f"{floor}. {session.status_line()}"
                + (f" Judge: {summary}." if summary else "")
                + (
                    " Judge: enough to answer."
                    if enough
                    else f" Judge: still missing {session.missing}."
                    if enough is False and session.missing
                    else ""
                )
            )
            return session.reply(
                kept,
                ignored,
                passed_over,
                what=tool,
                displaced=displaced,
                below_floor=below_floor,
            )


def _with_note(result: Any, note: str) -> Any:
    """*result* with *note* appended as a text block, unless already there."""
    if not note:
        return result
    content = list(result.get("content") or [])
    if any(note in str(block.get("text", "")) for block in content):
        return result
    return {**result, "content": [*content, {"text": note}]}
