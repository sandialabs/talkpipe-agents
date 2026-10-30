"""The findings loop: tools that return findings, judged as they arrive.

An agent answering a question from what its tools turn up needs more
than tool calls: something must decide, after every call, which of the
results are worth keeping, remember what is already known, tell the
agent what it still lacks and how many turns it has left, and end the
run when it has enough. This module is that loop, independent of what
the tools search or compute::

    tool → findings → judge → session → reply to the model

- :class:`Finding` — one piece of information a tool returned: the
  text the judge and the agent read, a label and a source to cite, a
  key to deduplicate by, and whatever else the tool wants to carry
  along in ``meta`` (a ``SearchResult``, say).
- :func:`finding_tool` — makes any function returning findings (or
  strings, or dicts, or search results) an agent tool. The function
  knows nothing about sessions or judges; it returns what it found.
- :class:`FindingsSession` — one question's state: the findings seen,
  kept (graded, best first) and ignored, why, the stop rule, the turn
  countdown, and the standard reply the model reads.
- :class:`FindingsJudge` — one batched model call deciding keep or
  ignore, with a relevance grade, for every finding of a call, read
  leniently (JSON, fenced, in prose, or one decision per line). Also
  the ``judgeFindings`` segment, for pipelines without an agent.
- :class:`JudgeFindings` — the Strands hook that runs the judge as a
  step of the agent loop: after a finding tool's call, before the
  model sees the result. The judging costs the agent no turn, and it
  cannot skip it. At ``max_kept`` a better-graded finding displaces
  the weakest kept one, so the kept list is the best N, not the first N.
- ``enough_information`` (the stop), ``keep_finding`` and
  ``ignore_finding`` (deciding by hand, for a session without a judge)
  — registered agent tools that act on the running session.

``llmAgent[judge=True]`` (or ``findings=True`` to decide by hand)
installs the hook and a session per item; see ``segments``.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any

from strands import Agent
from strands import tool as strands_tool
from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.types.tools import ToolContext
from talkpipe.chatterlang.registry import register_segment
from talkpipe.pipe.core import AbstractSegment
from talkpipe.search.abstract import SearchResult
from talkpipe.util.data_manipulation import assign_property, extract_property

from .models import build_agent_model
from .models import model_params as parse_model_params
from .support import strip_think
from .tools import register_agent_tool

logger = logging.getLogger(__name__)

DEFAULT_MAX_KEPT = 10
# The judge grades each finding's relevance 1..5. A finding kept without
# a grade (by hand, by a judge whose reply gave none, or as "not judged")
# ranks as the default, so an ungraded run keeps in arrival order.
MIN_SCORE, MAX_SCORE, DEFAULT_SCORE = 1, 5, 3
# How much of a finding the judge and the agent are shown. Enough for a
# search chunk at talkpipe's default indexing settings; a shorter preview
# hides the part of a finding that would have made it worth keeping.
DEFAULT_MAX_CHARS = 1000

# Where the loop keeps its per-invocation state in Strands'
# invocation_state, which the tools (through their ToolContext) and the
# hook events share: the running session, and each finding tool call's
# findings by toolUseId until the hook judges them.
SESSION_KEY = "talkpipe_agents.findings_session"
FINDINGS_KEY = "talkpipe_agents.findings"

# Document fields a search hit's text, label and source are read from,
# in order of preference.
_TEXT_FIELDS = ("content", "text", "shingle")
_LABEL_FIELDS = ("title", "label", "filename", "name")
_SOURCE_FIELDS = ("source", "path", "url")

DEFAULT_STOP_INSTRUCTION = (
    "Stop calling tools and answer the question from the kept findings."
)


def _first(doc: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = doc.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _preview(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _one_line(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -- findings ----------------------------------------------------------------


@dataclass
class Finding:
    """One piece of information a tool returned.

    ``text`` is what the judge and the agent read; ``label`` (a title,
    file name, URL) names it in the trace and the "already known" notes;
    ``source`` is where it came from, for the answer to cite; ``key``
    is its identity across calls (two findings with one key are one
    finding) and defaults to a hash of the text; ``meta`` carries
    anything else — :meth:`from_search_result` keeps the original hit
    there as ``meta["result"]``.
    """

    text: str
    label: str = ""
    source: str = ""
    key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = str(self.text)
        if not self.key:
            digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            self.key = f"text:{digest[:16]}"

    @classmethod
    def from_search_result(
        cls,
        hit: SearchResult | dict[str, Any],
        rename_fields: dict[str, str] | None = None,
    ) -> Finding:
        """A finding for a search hit: a talkpipe ``SearchResult`` or a
        dict with ``doc_id``, ``score`` and ``document``. The text is the
        document's ``content`` (or ``text``, ``shingle``), the label its
        ``title`` (or ``label``, ``filename``, ``name``), the source its
        ``source`` (or ``path``, ``url``); the key is the ``doc_id``
        (else the document). A field in *rename_fields* is renamed first,
        unless the new name is already present. The hit, normalized to a
        ``SearchResult``, is kept as ``meta["result"]``."""
        if isinstance(hit, SearchResult):
            score, doc_id = hit.score, hit.doc_id
            document = dict(hit.document or {})
        else:
            score = float(hit.get("score", 0.0))
            doc_id = str(hit.get("doc_id", ""))
            document = dict(hit.get("document") or {})
        for old, new in (rename_fields or {}).items():
            if old in document and new not in document:
                document[new] = document.pop(old)
        result = SearchResult(score=score, doc_id=doc_id, document=document)
        return cls(
            text=_first(document, _TEXT_FIELDS),
            label=_first(document, _LABEL_FIELDS),
            source=_first(document, _SOURCE_FIELDS),
            key=doc_id or json.dumps(document, sort_keys=True, default=str),
            meta={"result": result},
        )

    def to_search_result(self) -> SearchResult:
        """The finding as a talkpipe ``SearchResult``, for talkpipe's RAG
        segments (``ConstructRAGPrompt``, ``AppendRAGSources``): the
        original hit when it came from one, else one built from the text,
        label and source."""
        result = self.meta.get("result")
        if isinstance(result, SearchResult):
            return result
        document = {"content": self.text}
        if self.label:
            document["title"] = self.label
        if self.source:
            document["source"] = self.source
        return SearchResult(
            score=0.0, doc_id=self.source or self.key, document=document
        )


def as_finding(value: Any, label: str = "") -> Finding | None:
    """*value* as a :class:`Finding`, or ``None`` when it holds no text.

    A ``Finding`` passes through; a ``SearchResult``, or a dict shaped
    like one (``document`` plus ``doc_id`` or ``score``), goes through
    :meth:`Finding.from_search_result`; another dict gives its ``text``
    (or ``content``), ``label``, ``source`` and ``key``, the rest going
    to ``meta`` (a dict with no text is shown as JSON); anything else is
    its string, labelled *label* (the tool's name, usually)."""
    if value is None:
        return None
    if isinstance(value, Finding):
        return value if value.text.strip() else None
    if isinstance(value, SearchResult) or (
        isinstance(value, dict)
        and "document" in value
        and ("doc_id" in value or "score" in value)
    ):
        finding = Finding.from_search_result(value)
    elif isinstance(value, dict):
        rest = dict(value)
        text = rest.pop("text", None) or rest.pop("content", None)
        found_label = str(rest.pop("label", "") or label)
        source = str(rest.pop("source", "") or "")
        key = str(rest.pop("key", "") or "")
        if text is None:
            text = json.dumps(value, ensure_ascii=False, default=str)
        finding = Finding(str(text), found_label, source, key, rest)
    else:
        finding = Finding(str(value), label)
    return finding if finding.text.strip() else None


def as_findings(value: Any, label: str = "") -> list[Finding]:
    """What a finding tool returned, as findings: one value (a string, a
    dict, a ``SearchResult``, a ``Finding``) or an iterable of them;
    ``None`` and empty texts drop out."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict, Finding, SearchResult)):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    return [f for f in (as_finding(v, label) for v in values) if f is not None]


def render_findings(findings: list[Finding], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Findings as plain text — what the model reads from a finding
    tool used without a findings session (no ``JudgeFindings`` hook)."""
    if not findings:
        return "Nothing found."
    blocks = []
    for i, finding in enumerate(findings, 1):
        head = f"[{i}]" + (f" {finding.label}" if finding.label else "")
        blocks.append(f"{head}\n{_preview(finding.text, max_chars)}")
    return "\n\n".join(blocks)


# -- the session -------------------------------------------------------------


@dataclass
class FindingsOutcome:
    """What a findings run produced: the kept findings (best first by
    the judge's grade, ties in keep order), a one-line note saying how
    the run ended, and the trace of what the agent did. ``error`` is
    the exception text when the run stopped with one; ``undecided``
    counts findings shown but neither kept nor ignored (only possible
    without a judge)."""

    findings: list[Finding]
    note: str
    trace: list[str] = field(default_factory=list)
    error: str = ""
    undecided: int = 0


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
    then says how many turns are left. *tell* receives each trace line
    (they are also collected in ``trace``). *stop_instruction* is what
    the agent is told to do once the run is over.

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
    ) -> None:
        self.question = str(question).strip()
        self.max_kept = max(1, max_kept)
        self.max_turns = max(1, max_turns) if max_turns is not None else None
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
        # kept would answer the question well — the intended stop.
        self.enough = False
        self.enough_reason = ""
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

    # -- stopping ------------------------------------------------------------

    @property
    def at_capacity(self) -> bool:
        """True once the run has kept as much as it may."""
        return len(self.kept_ids) >= self.max_kept

    @property
    def stop_note(self) -> str:
        """Why the run is over, phrased for the model; empty while it runs."""
        if self.enough:
            return "you already said you had enough"
        if self.at_capacity:
            return f"the {self.max_kept}-finding maximum is reached"
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
        return f"{head}\n{_preview(finding.text, self.max_chars)}"

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
    ) -> str:
        """The standard reply to a judged call: the kept findings in full,
        one line per ignored one with its reason, which earlier findings
        the new ones displaced at the maximum and how many relevant ones
        it left out, what is already known, and what to do next with the
        turns left. *what* names the call (a tool name)."""
        total = len(kept) + len(ignored) + passed_over
        source = f" from {what}" if what else ""
        if not total:
            return self.with_turns(
                f"No new findings{source}. Try different terms. {self.status_line()}"
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
        lines.append(self._known_summary())
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

    def show_undecided(self, fids: list[str], *, what: str = "") -> str:
        """The reply to a call when no judge runs: every finding with its
        id, for the agent to decide on with keep_finding / ignore_finding."""
        source = f" from {what}" if what else ""
        if not fids:
            return self.with_turns(f"Nothing found{source}. Try different terms.")
        blocks = []
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

    # -- the outcome ---------------------------------------------------------

    def outcome(self, stop_reason: str = "", error: str = "") -> FindingsOutcome:
        """How the run ended, as a :class:`FindingsOutcome`. *stop_reason*
        is the Strands stop reason of the agent's last result; *error*
        the exception text when the run raised."""
        kept, undecided = len(self.kept_ids), len(self.undecided)
        if error:
            how = f"stopped with an error ({error})"
        elif self.enough:
            how = "judged it had enough to answer"
            if self.enough_reason:
                how += f" ({self.enough_reason})"
        elif self.at_capacity:
            how = f"filled its {self.max_kept}-finding maximum"
        elif str(stop_reason).startswith("limit"):
            cap = f" {self.max_turns}-turn" if self.max_turns else ""
            how = f"hit the{cap} cap"
        elif self.calls and not kept and not undecided:
            # Everything found was judged and none of it was worth
            # keeping: the right ending for a question the tools cannot
            # answer (enough_information is refused with nothing kept).
            how = "found nothing relevant to keep"
        else:
            how = "stopped without saying it had enough"
        note = (
            f"Agent {how}: kept {kept} finding{'s' if kept != 1 else ''} after "
            f"{self.calls} call{'s' if self.calls != 1 else ''}, ignored "
            f"{len(self.ignored_ids)}"
            + (f", left {undecided} undecided" if undecided else "")
            + "."
        )
        self.tell(note)
        return FindingsOutcome(
            self.kept, note, list(self.trace), error=error, undecided=undecided
        )


# -- judging -----------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You judge findings for an agent that gathers information to answer a question. You are shown the question; optionally what the tool call that produced these findings was looking for, and what is already known from findings kept earlier; then the findings, each labelled f1, f2, and so on.

Decide for every finding whether to keep it or ignore it. The default is keep: keep a finding that bears on the question, even in part, or that supplies what the call was looking for. Overlapping coverage, examples, supporting detail, and another source saying the same thing are reasons to keep, not to ignore. Ignore a finding only when it is about something else or merely shares words with the question, or when its text repeats a finding that is already known nearly word for word. Never ignore a finding because another finding in this same list covers similar ground: judge each one against the question on its own. Read the question's terms in their ordinary sense. When unsure, keep.

Also grade each finding's relevance from 1 to 5: 5 answers the question, or a large part of it, directly; 3 is useful supporting material; 1 is barely related. The agent can keep only so many findings, and the grades decide which stay when more are kept than fit, so grade the findings relative to each other.

Reply with one decision per finding, in order, giving the finding's id, keep (true or false), the grade as score, and a short reason: for a kept finding, what it contributes to the answer; for an ignored one, why it does not help. Reply with JSON only, in this shape and nothing else:
{"decisions": [{"id": "f1", "keep": true, "score": 4, "reason": "..."}, {"id": "f2", "keep": false, "score": 1, "reason": "..."}]}"""


@dataclass
class Decision:
    """One decision as the judge wrote it: the finding's label, keep or
    not, why, and the relevance grade (``None`` when it gave none)."""

    id: str
    keep: bool
    reason: str = ""
    score: int | None = None


@dataclass
class Judgement:
    """A decision applied to its finding."""

    finding: Finding
    keep: bool
    reason: str
    score: int | None = None


_KEEP_WORDS = frozenset({"keep", "true", "yes"})
_IGNORE_WORDS = frozenset({"ignore", "false", "no"})
# A decision written as a line: "f1: true - reason", "[r2] ignore: reason",
# "3 keep reason" — for models that answer in prose rather than the JSON
# the prompt asks for (Ollama's cloud models ignore its format schema).
_LINE_RE = re.compile(
    r"^\W*([fr]?\d+)\b\W*(keep|ignore|true|false|yes|no)\b\W*(.*)$", re.IGNORECASE
)


def _as_keep(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if word in _KEEP_WORDS:
        return True
    if word in _IGNORE_WORDS:
        return False
    return None


def _as_score(value: Any) -> int | None:
    """*value* as a grade in ``MIN_SCORE..MAX_SCORE``, or ``None`` when
    it is missing or unreadable (the decision then ranks as the default)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = round(float(str(value).strip()))
    except (ValueError, OverflowError):
        return None
    return max(MIN_SCORE, min(MAX_SCORE, score))


def _json_decisions(data: Any) -> list[Decision] | None:
    if isinstance(data, dict):
        data = data.get("decisions")
    if not isinstance(data, list):
        return None
    decisions = []
    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry:
            return None
        keep = _as_keep(entry.get("keep"))
        if keep is None:
            return None
        decisions.append(
            Decision(
                str(entry["id"]),
                keep,
                str(entry.get("reason") or ""),
                _as_score(entry.get("score", entry.get("grade"))),
            )
        )
    return decisions


def parse_judgements(text: str) -> list[Decision]:
    """The judge's decisions from its reply, however it wrote them.

    The prompt asks for one JSON object; that is accepted bare, fenced,
    or surrounded by prose, as is a bare JSON list of decisions. A
    decision's ``score`` (or ``grade``) is optional: a reply without
    grades reads as before, and its decisions rank as the default grade.
    Failing JSON, every line that reads as ``<id> <keep|ignore|true|false>
    <reason>`` is an ungraded decision. Raises ``ValueError`` when no
    decision can be read."""
    text = text.strip()
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start, end = text.find(start_char), text.rfind(end_char)
        if 0 <= start < end:
            try:
                decisions = _json_decisions(json.loads(text[start : end + 1]))
            except ValueError:
                decisions = None
            if decisions is not None:
                return decisions
    decisions = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        label, verdict, reason = match.groups()
        decisions.append(
            Decision(
                label,
                verdict.lower() in _KEEP_WORDS,
                # Hyphens and the en and em dashes models separate with.
                reason.strip(" \t:-–—.,"),  # noqa: RUF001 - the dashes are the point
            )
        )
    if not decisions:
        reply = " ".join(text.split())
        if not reply:
            raise ValueError("the judge's reply held no decisions: it was empty")
        if len(reply) > 120:
            reply = reply[:119] + "…"
        raise ValueError(f"the judge's reply held no decisions: {reply!r}")
    return decisions


def _label_of(decision_id: str) -> str:
    # Models write the label as "f1", "F1", "[f1]", "r1" or just "1";
    # all mean the same finding.
    label = "".join(ch for ch in decision_id.lower() if ch.isalnum())
    if label[:1] in ("f", "r") and label[1:].isdigit():
        label = label[1:]
    return f"f{label}" if label.isdigit() else label


class _JudgeModelError(ValueError):
    """The judge's model could not be built (no model configured, an
    unknown source, a missing provider extra). Unlike a failed model
    call, this is raised: every judgement would fail the same way."""


@register_segment("judgeFindings")
class FindingsJudge(AbstractSegment):
    """Have a model judge findings against a question, in one call.

    As a segment: expects dict-like items holding a list of findings
    (anything a finding tool may return: ``Finding`` objects, strings,
    dicts with ``text``/``label``/``source``, search results) and the
    question; optionally what the call that found them was looking for,
    notes on what is already known, and the keys of findings already
    decided on, which are skipped. Sets a list of judgements in order,
    ``{"finding": Finding, "keep": bool, "reason": str, "score": int |
    None}`` (the relevance grade, 1 to 5, when the model gave one), or
    with ``drop_ignored`` only the kept ``Finding`` objects.

    As a judge (what the :class:`JudgeFindings` hook calls):
    :meth:`judge`. The model is any agent model source — ``source`` and
    ``model`` resolve exactly as ``llmAgent``'s do — asked for JSON and
    read leniently (:func:`parse_judgements`); from Python, ``generate``
    (prompt text in, reply text out) replaces the model, and subclasses
    may override :meth:`decide`. When the model fails or leaves a
    finding undecided, the finding is kept with a reason starting
    ``not judged``; only a model that cannot be built at all raises
    (a ``ValueError``). Model calls are taken one at a time.
    """

    def __init__(
        self,
        findings_field: Annotated[
            str, "The field holding the findings to judge"
        ] = "findings",
        question_field: Annotated[str, "The field holding the question"] = "question",
        looking_for_field: Annotated[
            str, "The field saying what the call was looking for (optional)"
        ] = "looking_for",
        known_field: Annotated[
            str, "The field holding notes on what is already known (optional)"
        ] = "known",
        seen_field: Annotated[
            str, "The field holding keys of findings already decided on (optional)"
        ] = "seen",
        set_as: Annotated[str, "The field to set the judgements as"] = "judged",
        drop_ignored: Annotated[
            bool, "Set only the kept findings, as Finding objects"
        ] = False,
        model: Annotated[
            str | None, "The judging model (falls back to the configured default)"
        ] = None,
        source: Annotated[
            str | None,
            "The judging model's source, as for llmAgent (falls back to the configured default)",
        ] = None,
        temperature: Annotated[float | None, "The sampling temperature"] = None,
        model_params: Annotated[
            dict | str | None, "Provider request parameters, as for llmAgent"
        ] = None,
        max_chars: Annotated[
            int, "How much of each finding's text the judge is shown"
        ] = DEFAULT_MAX_CHARS,
        system_prompt: Annotated[
            str, "The judge's system prompt"
        ] = JUDGE_SYSTEM_PROMPT,
        generate: Annotated[
            Callable[[str], Any] | None,
            "From Python: a callable taking the prompt and returning the "
            "reply text, used instead of the model",
        ] = None,
    ) -> None:
        super().__init__()
        self.findings_field = findings_field
        self.question_field = question_field
        self.looking_for_field = looking_for_field
        self.known_field = known_field
        self.seen_field = seen_field
        self.set_as = set_as
        self.drop_ignored = drop_ignored
        self.model = model
        self.source = source
        self.temperature = temperature
        # parsed here so malformed JSON fails when the script compiles
        self.model_params = parse_model_params(model_params)
        self.max_chars = max(1, max_chars)
        self.system_prompt = system_prompt
        self._generate = generate
        # The model provider, built on first use like llmAgent's.
        self._model: Any = None
        # A judgement at a time: Strands runs one turn's tool calls in
        # threads, and a prompt adapter holds the message it is sending.
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        """The judge's reply text for *prompt*."""
        if self._generate is not None:
            return str(self._generate(prompt))
        if self._model is None:
            try:
                self._model = build_agent_model(
                    self.source,
                    self.model,
                    temperature=self.temperature,
                    params=self.model_params,
                )
            except Exception as exc:
                # Not a failed judgement: every one would fail the same way,
                # so keeping everything would hide a configuration error.
                raise _JudgeModelError(f"judgeFindings: {exc}") from exc
        agent = Agent(
            model=self._model,
            system_prompt=self.system_prompt,
            callback_handler=None,
        )
        return strip_think(str(agent(prompt)))

    def prompt(
        self,
        question: str,
        looking_for: str,
        known: Iterable[str],
        labelled: list[tuple[str, Finding]],
    ) -> str:
        """The prompt for one judgement: the question, what the call was
        looking for, what is already known, and the labelled findings."""
        lines = [f"Question: {question.strip()}"]
        if looking_for.strip():
            lines.append(f"This call is looking for: {looking_for.strip()}")
        notes = [str(note).strip() for note in known if str(note).strip()]
        lines.append("Already known:")
        lines.extend(f"- {note}" for note in notes)
        if not notes:
            lines.append("- nothing yet")
        lines.append("Findings:")
        for label, finding in labelled:
            head = f"[{label}]" + (f" {finding.label}" if finding.label else "")
            lines.append(f"{head}\n{_preview(finding.text, self.max_chars)}")
        return "\n".join(lines)

    def decide(
        self, prompt: str, labelled: list[tuple[str, Finding]]
    ) -> dict[str, Decision]:
        """The decisions for *labelled*, by label (``f1``, …): one model
        call on *prompt*. Raises when the reply cannot be read."""
        with self._lock:
            reply = self.generate(prompt)
        return {_label_of(d.id): d for d in parse_judgements(reply)}

    def judge(
        self,
        findings: Iterable[Any],
        question: str,
        looking_for: str = "",
        known: Iterable[str] = (),
        seen: Iterable[str] = (),
    ) -> list[Judgement]:
        """Judge *findings* for *question*, in order. Findings whose key is
        in *seen* are left out; a finding the model gave no decision for,
        or every finding when the model failed, is kept with a reason that
        starts "not judged"."""
        skip = {str(key) for key in seen}
        labelled: list[tuple[str, Finding]] = []
        for finding in as_findings(list(findings or [])):
            if finding.key in skip:
                continue
            labelled.append((f"f{len(labelled) + 1}", finding))
        if not labelled:
            return []
        prompt = self.prompt(question, looking_for, known, labelled)
        try:
            decisions = self.decide(prompt, labelled)
        except _JudgeModelError:
            raise
        except Exception as exc:
            logger.warning("Judging findings failed; keeping them", exc_info=True)
            decisions = {}
            undecided = f"not judged: {_one_line(exc)}"
        else:
            undecided = "not judged: the model gave no decision for it"
        judged = []
        for label, finding in labelled:
            decision = decisions.get(label)
            if decision is None:
                judged.append(Judgement(finding, True, undecided))
            else:
                judged.append(
                    Judgement(
                        finding, decision.keep, decision.reason.strip(), decision.score
                    )
                )
        return judged

    def transform(self, input_iter: Iterable[Any]) -> Iterator[Any]:
        for item in input_iter:

            def get(name: str, item: Any = item) -> Any:
                return extract_property(item, name, fail_on_missing=False)

            judged = self.judge(
                get(self.findings_field) or [],
                str(get(self.question_field) or ""),
                str(get(self.looking_for_field) or ""),
                get(self.known_field) or [],
                get(self.seen_field) or [],
            )
            if self.drop_ignored:
                assign_property(
                    item, self.set_as, [j.finding for j in judged if j.keep]
                )
            else:
                assign_property(
                    item,
                    self.set_as,
                    [
                        {
                            "finding": j.finding,
                            "keep": j.keep,
                            "reason": j.reason,
                            "score": j.score,
                        }
                        for j in judged
                    ],
                )
            yield item


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
        findings = stash.pop(str(event.tool_use.get("toolUseId")), None)
        result = event.result
        if findings is None or result.get("status") == "error":
            if result.get("status") != "error":
                event.result = _with_note(result, session.turns_note())
            return
        args = event.tool_use.get("input") or {}
        looking_for = (
            str(args.get("looking_for") or "") if isinstance(args, dict) else ""
        )
        # Off the event loop: the judge makes a blocking model call.
        text = await asyncio.to_thread(
            self.process, findings, looking_for, str(event.tool_use.get("name", ""))
        )
        event.result = {**result, "content": [{"text": text}]}

    def process(
        self, findings: list[Finding], looking_for: str = "", tool: str = ""
    ) -> str:
        """Run one call's *findings* through the session (and the judge,
        if any) and return the reply the model reads."""
        session = self.session
        if session is None:
            return render_findings(findings)
        with session.lock:
            session.calls += 1
            if self.judge is None:
                fids = list(dict.fromkeys(session.register(f) for f in findings))
                fresh = sum(1 for fid in fids if not session.is_decided(fid))
                session.tell(
                    f"agent ← {tool}: {len(fids)} finding(s), {fresh} undecided. "
                    + session.status_line()
                )
                return session.show_undecided(fids, what=tool)
            judgements = self.judge.judge(
                findings,
                session.question,
                looking_for,
                session.known_notes(),
                session.seen_keys(),
            )
            kept: list[str] = []
            ignored: list[str] = []
            displaced: list[str] = []
            passed_over = 0
            for judgement in judgements:
                weakest = None
                if judgement.keep and session.at_capacity:
                    weakest = session.displaceable_by(judgement.score)
                    if weakest is None:
                        # Not registered either, so it is not left undecided.
                        passed_over += 1
                        continue
                fid = session.register(judgement.finding)
                if session.is_decided(fid):
                    continue
                if weakest is not None:
                    session.apply(weakest, False, f"displaced by {fid}")
                    displaced.append(weakest)
                session.apply(fid, judgement.keep, judgement.reason, judgement.score)
                (kept if judgement.keep else ignored).append(fid)
            session.tell(
                f"agent ← {tool}: {len(kept) + len(ignored) + passed_over} "
                f"finding(s), kept {len(kept)}, ignored {len(ignored)}"
                + (f", {len(displaced)} displaced" if displaced else "")
                + (f", {passed_over} passed over" if passed_over else "")
                + f". {session.status_line()}"
            )
            return session.reply(
                kept, ignored, passed_over, what=tool, displaced=displaced
            )


def _with_note(result: Any, note: str) -> Any:
    """*result* with *note* appended as a text block, unless already there."""
    if not note:
        return result
    content = list(result.get("content") or [])
    if any(note in str(block.get("text", "")) for block in content):
        return result
    return {**result, "content": [*content, {"text": note}]}


# -- finding tools -----------------------------------------------------------


def finding_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    register: bool = True,
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
    ``register_agent_tool`` unless *register* is false. Works bare or
    with arguments::

        @finding_tool
        def search_notes(query: str, looking_for: str = "") -> list[Finding]:
            \"\"\"Search the notes ...\"\"\"
    """
    if func is None:

        def decorate(f: Callable[..., Any]) -> Any:
            return finding_tool(
                f, name=name, description=description, register=register
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
        found = as_findings(func(**kwargs), label=tool_name)
        if isinstance(state, dict) and context is not None:
            stash = state.setdefault(FINDINGS_KEY, {})
            stash[str(context.tool_use.get("toolUseId"))] = found
        return render_findings(found)

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
    if register:
        register_agent_tool(tool_obj)
    return tool_obj


# -- the session tools -------------------------------------------------------


def _no_session(name: str) -> str:
    return (
        f"{name} only works while a findings session runs "
        "(llmAgent with judge=True or findings=True)."
    )


def _session_of(tool_context: ToolContext | None) -> FindingsSession | None:
    state = getattr(tool_context, "invocation_state", None)
    session = state.get(SESSION_KEY) if isinstance(state, dict) else None
    return session if isinstance(session, FindingsSession) else None


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
    session = _session_of(tool_context)
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
    session = _session_of(tool_context)
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
    session = _session_of(tool_context)
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
