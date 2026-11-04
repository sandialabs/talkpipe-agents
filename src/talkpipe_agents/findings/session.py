"""One question's state: :class:`FindingsSession` and its
:class:`FindingsOutcome`.

The session holds the findings seen, kept (graded, best first) and
ignored, why each was decided, the judge's verdict on what is kept, the
stop rule and the turn countdown, and writes every reply the model
reads from a judged tool call. It knows nothing about models, hooks or
Strands: the hook (:mod:`.hook`) and the session tools
(:mod:`.finding_tools`) drive it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .finding import DEFAULT_MAX_CHARS, Finding, floor_note, preview

logger = logging.getLogger(__name__)

DEFAULT_MAX_KEPT = 10
# The judge grades each finding's relevance 1..5. A finding kept without
# a grade (by hand, by a judge whose reply gave none, or as "not judged")
# ranks as the default, so an ungraded run keeps in arrival order.
MIN_SCORE, MAX_SCORE, DEFAULT_SCORE = 1, 5, 3

DEFAULT_STOP_INSTRUCTION = (
    "Stop calling tools and answer the question from the kept findings."
)

# The reason recorded when the judge's verdict, not the agent, ends a run.
JUDGE_ENOUGH_REASON = "the judge found the first call's findings enough to answer"

# How a run can end, as ``FindingsOutcome.ending`` names it, most to least
# satisfying, with the words ``FindingsOutcome.note`` uses. A host that
# words the endings its own way (chunks and searches, say) maps these
# codes instead of re-deriving the conditions.
ENDINGS = {
    "judge_enough": "found the first call enough: answered without an agent turn",
    "enough": "judged it had enough to answer",
    "capacity": "filled its {max_kept}-finding maximum",
    "timeout": "hit its {max_seconds:g}-second time budget",
    "turn_cap": "hit the{turn_cap} cap",
    "nothing_relevant": "found nothing relevant to keep",
    "undecided": "stopped without saying it had enough",
    "error": "stopped with an error ({error})",
}


# -- the session -------------------------------------------------------------


@dataclass
class Need:
    """One fact the answer needs, as the judge lists it: the fact, the
    label of the kept finding that states it (``None`` when none does;
    ``"known"`` when a finding kept earlier did), and the id the session
    gave it on an earlier call (``""`` for a fact named just now)."""

    fact: str
    found: str | None = None
    id: str = ""

    @property
    def covered(self) -> bool:
        return self.found is not None


@dataclass
class FindingsOutcome:
    """What a findings run produced: the kept findings (best first by
    the judge's grade, ties in keep order), a one-line note saying how
    the run ended, and the trace of what the agent did. ``error`` is
    the exception text when the run stopped with one; ``undecided``
    counts findings shown but neither kept nor ignored (only possible
    without a judge). ``sufficient`` and ``missing`` are the judge's
    last verdict on what is kept: whether it would answer the question
    (``None`` when the judge gave no verdict) and, in a phrase, what the
    answer still lacks. ``ending`` names how the run ended, one of the
    ``ENDINGS`` keys, for a host that words the note itself."""

    findings: list[Finding]
    note: str
    trace: list[str] = field(default_factory=list)
    error: str = ""
    undecided: int = 0
    ending: str = ""
    # The judge's last verdict on the kept findings: whether they would
    # answer the question (None when it gave none) and what is missing.
    sufficient: bool | None = None
    missing: str = ""


class FindingsSession:
    """One question's findings: seen, kept (graded) and ignored, why
    each was decided, the stop rule and the turn countdown.

    *question* is what findings are judged against. *max_kept* is a
    ceiling on how much one run may keep — a backstop, not a goal: the
    run should end when the agent calls ``enough_information``. Each
    kept finding carries the judge's relevance grade (``DEFAULT_SCORE``
    when it was kept without one); ``kept`` lists them best first, and
    at the ceiling a better-graded finding may displace the weakest.
    *max_turns* is the agent's turn cap, when it has one: every reply
    then says how many turns are left. *max_seconds* is the run's time
    budget, when it has one: once it is spent the run is over, like
    reaching *max_kept* (the :class:`~talkpipe_agents.hooks.Deadline`
    hook, which ``llmAgent`` installs with the same budget, cancels the
    tool calls themselves). *tell* receives each trace line (they are
    also collected in ``trace``). *stop_instruction* is what the agent
    is told to do once the run is over.

    Strands runs one turn's tool calls concurrently, in threads; ``lock``
    serializes registering and deciding, and the :class:`JudgeFindings`
    hook holds it for a whole judgement, so a second call's findings are
    judged against what the first one kept.
    """

    def __init__(
        self,
        question: str = "",
        *,
        max_kept: int = DEFAULT_MAX_KEPT,
        max_turns: int | None = None,
        tell: Callable[[str], None] | None = None,
        stop_instruction: str = DEFAULT_STOP_INSTRUCTION,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.question = str(question).strip()
        self.max_kept = max(1, max_kept)
        self.max_turns = max(1, max_turns) if max_turns is not None else None
        self.max_seconds = max_seconds
        self._clock = clock or time.monotonic
        self.started = self._clock()
        self.stop_instruction = stop_instruction
        self.max_chars = max(1, max_chars)
        self._tell = tell or logger.info
        self.trace: list[str] = []
        # Model turns the run has used, as last reported (see note_turn).
        self.turns_used = 0
        # Finding-tool calls whose findings reached the session.
        self.calls = 0
        self._findings: dict[str, Finding] = {}
        self._by_key: dict[str, str] = {}
        self.kept_ids: list[str] = []
        self.ignored_ids: set[str] = set()
        # Why each finding was kept or ignored, by id.
        self.reasons: dict[str, str] = {}
        # The judge's relevance grade, by id, for findings that got one.
        self.scores: dict[str, int] = {}
        # Set by enough_information: the agent's judgement that what is
        # kept would answer the question well — the intended stop. Or by
        # accept_verdict, when the judge's verdict ends the run before the
        # agent takes a turn; enough_by says which ("agent" or "judge").
        self.enough = False
        self.enough_reason = ""
        self.enough_by = ""
        # The judge's last verdict on what is kept (see note_verdict), and
        # the facts the answer needs as the judge has listed them, each with
        # the id of the kept finding that states it (see note_needs). With
        # a list, the verdict is derived from it: enough when every fact is
        # stated by a finding that is still kept.
        self.sufficient: bool | None = None
        self.missing = ""
        self.needs: list[Need] = []
        self.lock = threading.RLock()

    # -- bookkeeping ---------------------------------------------------------

    def tell(self, message: str) -> None:
        """Record *message* in the trace and pass it on."""
        self.trace.append(message)
        self._tell(message)

    def register(self, finding: Finding) -> str:
        """The short id (``f1``, ``f2``, …) of *finding*, registering it
        if its key is new; a finding seen before keeps its first id."""
        with self.lock:
            existing = self._by_key.get(finding.key)
            if existing is not None:
                return existing
            fid = f"f{len(self._findings) + 1}"
            self._findings[fid] = finding
            self._by_key[finding.key] = fid
            return fid

    def finding(self, fid: str) -> Finding | None:
        """The finding registered as *fid*, if any."""
        return self._findings.get(fid.strip())

    def is_decided(self, fid: str) -> bool:
        return fid in self.ignored_ids or fid in self.kept_ids

    def score_of(self, fid: str) -> int:
        """The grade *fid* ranks by: the judge's, else ``DEFAULT_SCORE``."""
        return self.scores.get(fid, DEFAULT_SCORE)

    @property
    def kept(self) -> list[Finding]:
        """The kept findings, best first: by grade, ties in keep order."""
        ranked = sorted(self.kept_ids, key=self.score_of, reverse=True)
        return [self._findings[fid] for fid in ranked]

    def displaceable_by(self, score: int | None) -> str | None:
        """At capacity, the id of the weakest kept finding when a new one
        graded *score* (``None``: the default grade) outranks it; else
        ``None``. Among equally weak findings, the one kept first."""
        if not self.at_capacity:
            return None
        weakest = min(self.kept_ids, key=self.score_of)
        if (DEFAULT_SCORE if score is None else score) > self.score_of(weakest):
            return weakest
        return None

    @property
    def ignored(self) -> list[Finding]:
        return [
            self._findings[fid] for fid in self._findings if fid in self.ignored_ids
        ]

    @property
    def notes(self) -> dict[str, str]:
        """What each kept finding contributes, by id."""
        return {fid: self.reasons.get(fid, "") for fid in self.kept_ids}

    @property
    def undecided(self) -> list[str]:
        """Ids of findings shown but neither kept nor ignored."""
        return [fid for fid in self._findings if not self.is_decided(fid)]

    def known_notes(self) -> list[str]:
        """What is already known, one line per kept finding: its label
        and what it contributes. The judge sees these, so a finding that
        repeats them is ignored and one that adds to them is kept."""
        notes = []
        for fid in self.kept_ids:
            parts = [self._findings[fid].label, self.reasons.get(fid, "")]
            notes.append(" — ".join(p for p in parts if p) or fid)
        return notes

    def seen_keys(self) -> set[str]:
        """The keys of every finding already kept or ignored, so a judge
        skips them instead of judging them again."""
        return {
            self._findings[fid].key for fid in self._findings if self.is_decided(fid)
        }

    def apply(
        self, fid: str, keep: bool, reason: str = "", score: int | None = None
    ) -> None:
        """Record a decision on *fid*, with the judge's grade *score* if
        it gave one, and trace it. Ignoring a kept finding un-keeps it
        (how a displaced finding leaves the kept list)."""
        reason = reason.strip().rstrip(".")
        with self.lock:
            if keep:
                self.ignored_ids.discard(fid)
                if fid not in self.kept_ids:
                    self.kept_ids.append(fid)
            else:
                self.ignored_ids.add(fid)
                if fid in self.kept_ids:
                    self.kept_ids.remove(fid)
            self.reasons[fid] = reason
            if score is not None:
                self.scores[fid] = score
        label = self._findings[fid].label
        # The trace is read by a person, who never saw which finding the
        # short id stands for; name it.
        named = f" ({label})" if label else ""
        graded = f" [{score}/{MAX_SCORE}]" if score is not None else ""
        why = f" — {reason}" if reason else ""
        verb = "keep" if keep else "ignore"
        self.tell(f"agent · {verb} {fid}{named}{graded}{why}. {self.status_line()}")

    def note_verdict(self, sufficient: bool | None, missing: str = "") -> None:
        """Record the judge's verdict on what is kept: whether it would
        answer the question, and what the answer still lacks. A verdict
        of ``None`` (the judge gave none) leaves the last one standing."""
        if sufficient is None:
            return
        self.sufficient = sufficient
        self.missing = "" if sufficient else missing.strip().rstrip(".")

    def note_needs(self, needs: list[Need], labels: dict[str, str]) -> None:
        """Record the judge's list of the facts the answer needs, and
        derive the verdict from it. *labels* maps the judge's labels for
        this call's findings (``f1``, …) to the session's ids. An entry
        with the id of a need recorded earlier updates it (a fact once
        found stays found, by its finding's id); any other entry is a new
        need. A need counts as covered while the finding that states it
        is kept. The verdict is then enough when every need is covered,
        and ``missing`` names the facts that are not."""
        by_id = {need.id: need for need in self.needs}
        for need in needs:
            found = self._fid_of(need.found, labels)
            # An earlier need, by the id it was shown under, or by its
            # wording when the judge wrote the same fact again without one.
            known = by_id.get(need.id) or next(
                (n for n in self.needs if n.fact.lower() == need.fact.lower()), None
            )
            if known is None:
                if not need.fact:
                    continue
                known = Need(need.fact, None, f"N{len(self.needs) + 1}")
                self.needs.append(known)
                by_id[known.id] = known
            if found is not None and not self._covers(known):
                known.found = found
        uncovered = [need.fact for need in self.needs if not self._covers(need)]
        self.sufficient = bool(self.needs) and not uncovered
        self.missing = "; ".join(uncovered)

    def _fid_of(self, found: str | None, labels: dict[str, str]) -> str | None:
        """The session id of the kept finding the judge cited as *found*:
        by its label for this call, or directly when the judge cited a
        session id; ``None`` for no citation or one that is not kept."""
        if not found:
            return None
        label = "".join(ch for ch in found.lower() if ch.isalnum())
        if label[:1] in ("f", "r") and label[1:].isdigit():
            label = label[1:]
        fid = labels.get(f"f{label}" if label.isdigit() else label, f"f{label}")
        return fid if fid in self.kept_ids else None

    def _covers(self, need: Need) -> bool:
        return need.found is not None and need.found in self.kept_ids

    def needs_summary(self) -> str:
        """The needs and their coverage in one phrase, for the trace;
        empty without a list."""
        if not self.needs:
            return ""
        covered = sum(1 for need in self.needs if self._covers(need))
        return f"needs {len(self.needs)}, {covered} found"

    def accept_verdict(self, reason: str = JUDGE_ENOUGH_REASON) -> bool:
        """End the run on the judge's verdict: when the judge found that
        what is kept would answer the question, and something is kept,
        the run has enough, as if the agent had said so, and *reason* is
        recorded as why. Returns whether the run ended. This is how a
        call judged before the agent starts (:meth:`JudgeFindings.seed`)
        can answer the question with no agent turn at all."""
        if not (self.sufficient and self.kept_ids):
            return False
        self.enough = True
        self.enough_by = "judge"
        self.enough_reason = reason.strip().rstrip(".")
        self.tell(f"agent · enough information, by the judge's verdict — {reason}")
        return True

    @property
    def verdict_note(self) -> str:
        """The judge's verdict, phrased for the model; empty without one."""
        if self.sufficient is None or not self.kept_ids:
            return ""
        if self.sufficient:
            return (
                "The judge finds that what is kept would answer the question: "
                "call enough_information now unless you see a gap it missed."
            )
        if self.missing:
            return f"The judge finds the answer still lacks: {self.missing}."
        return "The judge finds that what is kept does not yet answer the question."

    def needs_notes(self) -> list[str]:
        """The facts the answer needs, one line each with whether a kept
        finding states it, for the model's reply; empty without a list."""
        return [
            f"- {need.fact}: "
            + (f"stated by {need.found}" if self._covers(need) else "not yet found")
            for need in self.needs
        ]

    # -- stopping ------------------------------------------------------------

    @property
    def at_capacity(self) -> bool:
        """True once the run has kept as much as it may."""
        return len(self.kept_ids) >= self.max_kept

    @property
    def timed_out(self) -> bool:
        """True once the run's time budget (*max_seconds*) is spent."""
        return (
            self.max_seconds is not None
            and self._clock() - self.started >= self.max_seconds
        )

    @property
    def stop_note(self) -> str:
        """Why the run is over, phrased for the model; empty while it runs."""
        if self.enough:
            return "you already said you had enough"
        if self.at_capacity:
            return f"the {self.max_kept}-finding maximum is reached"
        if self.timed_out:
            return f"the {self.max_seconds:g}-second time budget is spent"
        return ""

    @property
    def should_stop(self) -> bool:
        return bool(self.stop_note)

    def guard(self, name: str) -> str:
        """The refusal a tool returns once the run is over, or "" while it
        runs. Finding tools and the session tools check it before doing
        anything, so a model that keeps calling after being told to stop
        gets the same answer every time instead of collecting more."""
        reason = self.stop_note
        if not reason:
            return ""
        self.tell(f"agent ✗ {name} refused: {reason}")
        return self.with_turns(
            f"The search is over — {reason}. {self.stop_instruction}"
        )

    def declare_enough(self, reason: str = "") -> str:
        """Record the agent's judgement that it can answer the question.
        Refused while nothing is kept: there would be nothing to answer
        from, so the agent is told to keep looking instead."""
        reason = reason.strip().rstrip(".")
        if not self.kept_ids:
            self.tell("agent ✗ enough_information refused: nothing kept yet")
            return self.with_turns(
                "Nothing has been kept yet, so there is nothing to answer from. "
                "Keep looking, with different terms, for what would answer the "
                "question, even if only in part."
            )
        self.enough = True
        self.enough_by = "agent"
        self.enough_reason = reason
        why = f" — {reason}" if reason else ""
        self.tell(f"agent · enough information{why}. {self.status_line()}")
        return self.with_turns(
            f"Noted: {len(self.kept_ids)} finding(s) kept. {self.stop_instruction}"
        )

    def decide(self, fid: str, reason: str, *, keep: bool) -> str:
        """The ``keep_finding`` / ``ignore_finding`` tools' reply."""
        fid = fid.strip()
        verb = "keep" if keep else "ignore"
        if fid not in self._findings:
            # Traced as well as returned: a model that guesses ids, or
            # decides in the same turn as the call that finds them, would
            # otherwise leave a trace full of keeps and nothing kept.
            self.tell(f"agent ✗ {verb}_finding refused: unknown finding id {fid!r}")
            return self.with_turns(
                f"Unknown finding id {fid!r}; use an id from a tool reply you "
                "have already received (a call made in the same step has not "
                "returned its ids yet)."
            )
        if fid in self.kept_ids:
            return self.with_turns(f"{fid} is already kept.")
        if fid in self.ignored_ids and not keep:
            return self.with_turns(f"{fid} is already ignored.")
        self.apply(fid, keep, reason)
        if self.at_capacity:
            return self.with_turns(
                f"{verb.capitalize()} {fid}. You have kept the maximum of "
                f"{self.max_kept} findings. {self.stop_instruction}"
            )
        return self.with_turns(
            f"{verb.capitalize()} {fid}. {self.status_line()} Call "
            "enough_information once what you have kept would answer the "
            "question well."
        )

    # -- the countdown -------------------------------------------------------

    def note_turn(self, turns_used: int) -> None:
        """Record that the run has used *turns_used* model turns (never
        goes backwards)."""
        self.turns_used = max(self.turns_used, turns_used)

    def note_turn_of(self, agent: Any) -> None:
        """Record the turns *agent* has used in its current invocation:
        Strands counts each model call as an event-loop cycle, which is
        what the turn cap counts. Does nothing without metrics."""
        metrics = getattr(agent, "event_loop_metrics", None)
        invocation = getattr(metrics, "latest_agent_invocation", None)
        cycles = getattr(invocation, "cycles", None)
        if cycles is not None:
            self.note_turn(len(cycles))

    @property
    def turns_left(self) -> int | None:
        """Turns the agent can still take after this reply, or ``None``
        without a cap. 1 means the next turn is its last."""
        if self.max_turns is None:
            return None
        return max(0, self.max_turns - self.turns_used)

    def turns_note(self) -> str:
        """How many turns are left, for the model; "" without a cap."""
        left = self.turns_left
        if left is None:
            return ""
        if left == 0:
            return "No turns left."
        if left == 1 and not self.should_stop:
            # The reply the model is reading is the last one it can act
            # on: another call now would end the run before it saw the
            # results.
            if not self.kept_ids:
                # ...and enough_information is refused with nothing kept.
                return (
                    "1 turn left: the next one is your last, and nothing is "
                    "kept yet, so try once more with different terms, or "
                    "answer that nothing relevant was found."
                )
            return (
                "1 turn left: the next one is your last, so call "
                "enough_information now rather than another tool."
            )
        return f"{left} of {self.max_turns} turns left."

    def with_turns(self, reply: str) -> str:
        """*reply* with the turn countdown appended, once."""
        note = self.turns_note()
        if note and note not in reply:
            return f"{reply} {note}"
        return reply

    def status_line(self) -> str:
        return self.with_turns(
            f"Kept {len(self.kept_ids)} finding(s), at most {self.max_kept}."
        )

    # -- replies -------------------------------------------------------------

    def _describe(self, fid: str) -> str:
        finding = self._findings[fid]
        head = f"[{fid}]" + (f" {finding.label}" if finding.label else "")
        if finding.source and finding.source != finding.label:
            head += f" (source: {finding.source})"
        return f"{head}\n{preview(finding.text, self.max_chars)}"

    def _known_summary(self) -> str:
        notes = self.known_notes()
        if not notes:
            return "Already known: nothing yet."
        return "Already known:\n" + "\n".join(f"- {note}" for note in notes)

    def _ignored_line(self, fid: str) -> str:
        label = self._findings[fid].label
        reason = self.reasons.get(fid, "")
        return (
            f"[{fid}]"
            + (f" {label}" if label else "")
            + (f" — {reason}" if reason else "")
        )

    def reply(
        self,
        kept: list[str],
        ignored: list[str],
        passed_over: int = 0,
        *,
        what: str = "",
        displaced: list[str] | None = None,
        below_floor: int = 0,
    ) -> str:
        """The standard reply to a judged call: the kept findings in full,
        one line per ignored one with its reason, which earlier findings
        the new ones displaced at the maximum and how many relevant ones
        it left out, how many results the tool's score floor left out
        unjudged, what is already known, and what to do next with the
        turns left. *what* names the call (a tool name)."""
        total = len(kept) + len(ignored) + passed_over
        source = f" from {what}" if what else ""
        if not total:
            return self.with_turns(
                " ".join(
                    filter(
                        None,
                        [
                            f"No new findings{source}.",
                            floor_note(below_floor),
                            f"Try different terms. {self.status_line()}",
                        ],
                    )
                )
            )
        lines = [
            f"{total} new finding(s){source}: kept {len(kept)}, ignored {len(ignored)}."
        ]
        if kept:
            lines.append("Kept:")
            lines.extend(self._describe(fid) for fid in kept)
        if ignored:
            lines.append("Ignored:")
            lines.extend(self._ignored_line(fid) for fid in ignored)
        if displaced:
            lines.append(
                f"No longer kept, outranked at the {self.max_kept}-finding maximum:"
            )
            lines.extend(self._ignored_line(fid) for fid in displaced)
        if passed_over:
            lines.append(
                f"{passed_over} more relevant finding(s) were not kept: the "
                f"{self.max_kept}-finding maximum is reached."
            )
        if below_floor:
            lines.append(floor_note(below_floor))
        lines.append(self._known_summary())
        if notes := self.needs_notes():
            lines.append("The answer needs:\n" + "\n".join(notes))
        if verdict := self.verdict_note:
            lines.append(verdict)
        if self.at_capacity:
            lines.append(
                f"You have kept the maximum of {self.max_kept} findings. "
                f"{self.stop_instruction}"
            )
        else:
            lines.append(
                f"{self.status_line()} Look for what the answer still lacks, "
                "or call enough_information once what is known would answer "
                "the question well."
            )
        return self.with_turns("\n\n".join(lines))

    def show_undecided(
        self, fids: list[str], *, what: str = "", below_floor: int = 0
    ) -> str:
        """The reply to a call when no judge runs: every finding with its
        id, for the agent to decide on with keep_finding / ignore_finding."""
        source = f" from {what}" if what else ""
        if not fids:
            return self.with_turns(
                " ".join(
                    filter(
                        None,
                        [
                            f"Nothing found{source}.",
                            floor_note(below_floor),
                            "Try different terms.",
                        ],
                    )
                )
            )
        blocks = []
        if below_floor:
            blocks.append(floor_note(below_floor))
        for fid in fids:
            block = self._describe(fid)
            if fid in self.kept_ids:
                block = block.replace(f"[{fid}]", f"[{fid}] (already kept)", 1)
            elif fid in self.ignored_ids:
                block = block.replace(f"[{fid}]", f"[{fid}] (already ignored)", 1)
            blocks.append(block)
        return self.with_turns(
            f"{len(fids)} finding(s){source}. Decide on each undecided one with "
            "keep_finding or ignore_finding, then look for what the question "
            f"still needs. {self.status_line()}\n\n" + "\n\n".join(blocks)
        )

    # -- the opening prompt --------------------------------------------------

    def budget_note(self) -> str:
        """The run's budget, for the agent's opening prompt: how many
        findings may be kept and how many turns it has."""
        turns = (
            f", and you have {self.max_turns} turns: every tool reply says how "
            "many are left, so call enough_information before they run out"
            if self.max_turns
            else ""
        )
        return f"At most {self.max_kept} findings are kept{turns}."

    def opening_prompt(self, reply: str | None = None, *, tool: str = "") -> str:
        """The agent's first message under the loop: the question, what has
        already happened, and the budget.

        Without *reply*, nothing has been called yet and the agent is told
        to start. With *reply* — what a call judged before the agent
        started returned (:meth:`JudgeFindings.seed`), *tool* naming the
        tool — the agent is told not to repeat that call: when the judge's
        verdict ended the run (:meth:`accept_verdict`), to answer now from
        what is kept; otherwise to look for what the answer still lacks,
        the judge's ``missing`` phrase included."""
        head = f"Question: {self.question}\n"
        budget = self.budget_note()
        if reply is None:
            return (
                f"{head}Nothing has been called yet. Call your tools to find what "
                "answers it; each call keeps its relevant findings. Then look for "
                "what the answer still lacks, and call enough_information once the "
                f"kept findings would answer this question well. {budget}"
            )
        call = f"A {tool} call" if tool else "A call"
        if self.enough:
            return (
                f"{head}{call} for the question itself has already been made and "
                "judged, and the judge finds that what it kept answers the "
                "question; its reply is below. Answer the question now from the "
                "kept findings, citing their sources, without calling tools "
                f"unless you see a gap the judge missed.\n\n{reply}"
            )
        missing = (
            f" The judge says it still lacks: {self.missing}." if self.missing else ""
        )
        return (
            f"{head}{call} for the question itself has already been made and "
            "judged; its reply is below. Do not repeat it. Search for what the "
            f"answer still lacks{missing} Use different terms or another tool, and "
            "call enough_information once the kept findings would answer this "
            f"question well. {budget}\n\n{reply}"
        )

    # -- the outcome ---------------------------------------------------------

    def ending(self, stop_reason: str = "", error: str = "") -> str:
        """How the run ended, as one of the ``ENDINGS`` keys. *stop_reason*
        is the Strands stop reason of the agent's last result; *error* the
        exception text when the run raised. A run the judge's verdict
        ended before the agent took a turn is ``judge_enough``; one the
        agent ended with enough_information is ``enough``; then the
        backstops (the maximum kept, the time budget, the turn cap); then
        ``nothing_relevant`` when every finding was judged and none kept
        — the right ending for a question the tools cannot answer — and
        ``undecided`` when the agent stopped without saying it had
        enough."""
        if error:
            return "error"
        if self.enough:
            return "judge_enough" if self.enough_by == "judge" else "enough"
        if self.at_capacity:
            return "capacity"
        if self.timed_out:
            return "timeout"
        if str(stop_reason).startswith("limit"):
            return "turn_cap"
        if self.calls and not self.kept_ids and not self.undecided:
            return "nothing_relevant"
        return "undecided"

    def describe_ending(self, ending: str, error: str = "") -> str:
        """The ``ENDINGS`` words for *ending*, filled in for this run."""
        how = ENDINGS[ending].format(
            max_kept=self.max_kept,
            max_seconds=self.max_seconds or 0,
            turn_cap=f" {self.max_turns}-turn" if self.max_turns else "",
            error=error,
        )
        if ending == "enough" and self.enough_reason:
            how += f" ({self.enough_reason})"
        return how

    def outcome(self, stop_reason: str = "", error: str = "") -> FindingsOutcome:
        """How the run ended, as a :class:`FindingsOutcome`. *stop_reason*
        is the Strands stop reason of the agent's last result; *error*
        the exception text when the run raised."""
        kept, undecided = len(self.kept_ids), len(self.undecided)
        ending = self.ending(stop_reason, error)
        how = self.describe_ending(ending, error)
        note = (
            f"Agent {how}: kept {kept} finding{'s' if kept != 1 else ''} after "
            f"{self.calls} call{'s' if self.calls != 1 else ''}, ignored "
            f"{len(self.ignored_ids)}"
            + (f", left {undecided} undecided" if undecided else "")
            + "."
        )
        self.tell(note)
        return FindingsOutcome(
            self.kept,
            note,
            list(self.trace),
            error=error,
            undecided=undecided,
            sufficient=self.sufficient,
            missing=self.missing,
            ending=ending,
        )
