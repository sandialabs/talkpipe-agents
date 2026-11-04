"""The judge: one batched model call deciding keep or ignore, with a
relevance grade, for every finding of a call, and a verdict on whether
what is kept would answer the question.

:class:`FindingsJudge` is also the ``judgeFindings`` segment, for
pipelines without an agent. Its reply is read leniently
(:func:`parse_verdict`): JSON as asked, or fenced, or in prose, or one
decision per line, or the complete decisions of a reply a model's
output limit cut off.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any

from strands import Agent
from talkpipe.chatterlang.registry import register_segment
from talkpipe.pipe.core import AbstractSegment
from talkpipe.util.data_manipulation import assign_property, extract_property

from ..models import build_agent_model
from ..models import model_params as parse_model_params
from ..support import strip_think
from .finding import DEFAULT_MAX_CHARS, Finding, as_findings, preview
from .session import MAX_SCORE, MIN_SCORE

logger = logging.getLogger(__name__)


def _one_line(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -- judging -----------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You judge findings for an agent that gathers information to answer a question. You are shown the question; optionally what the tool call that produced these findings was looking for, and what is already known from findings kept earlier; then the findings, each labelled f1, f2, and so on.

Decide for every finding whether to keep it or ignore it. The default is keep: keep a finding that bears on the question, even in part, or that supplies what the call was looking for. Overlapping coverage, examples, supporting detail, and another source saying the same thing are reasons to keep, not to ignore. Ignore a finding only when it is about something else or merely shares words with the question, or when its text repeats a finding that is already known nearly word for word. Never ignore a finding because another finding in this same list covers similar ground: judge each one against the question on its own. Read the question's terms in their ordinary sense. When unsure, keep.

Also grade each finding's relevance from 1 to 5: 5 answers the question, or a large part of it, directly; 3 is useful supporting material; 1 is barely related. The agent can keep only so many findings, and the grades decide which stay when more are kept than fit, so grade the findings relative to each other.

Reply with one decision per finding, in order, giving the finding's id, keep (true or false), the grade as score, and a short reason: for a kept finding, what it contributes to the answer; for an ignored one, why it does not help. Then say whether what is already known together with the findings you keep would let the question be answered well: enough, true or false, and when false, missing: in one phrase, what the answer still lacks. Reply with JSON only, in this shape and nothing else:
{"decisions": [{"id": "f1", "keep": true, "score": 4, "reason": "..."}, {"id": "f2", "keep": false, "score": 1, "reason": "..."}], "enough": false, "missing": "..."}"""


@dataclass
class Decision:
    """One decision as the judge wrote it: the finding's label, keep or
    not, why, and the relevance grade (``None`` when it gave none)."""

    id: str
    keep: bool
    reason: str = ""
    score: int | None = None


@dataclass
class Verdict:
    """A judge's reply, read: its decisions, and whether what is known
    plus what it keeps would answer the question (``None`` when the
    reply did not say) with what the answer still lacks."""

    decisions: list[Decision]
    enough: bool | None = None
    missing: str = ""


@dataclass
class Judgement:
    """A decision applied to its finding."""

    finding: Finding
    keep: bool
    reason: str
    score: int | None = None


class Judged(list[Judgement]):
    """The judgements for one call, in order, with the judge's verdict
    on the whole: ``enough`` (``None`` when it gave none) and
    ``missing``. A plain list to everything that reads it as one."""

    enough: bool | None = None
    missing: str = ""


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


def _json_verdict(data: Any) -> Verdict | None:
    """The reply's JSON as a verdict: its decisions (a list, bare or under
    ``decisions``) and, from the object, ``enough`` and ``missing``."""
    enough, missing = None, ""
    if isinstance(data, dict):
        enough = _as_keep(data.get("enough", data.get("sufficient")))
        missing = str(data.get("missing") or "")
        data = data.get("decisions")
    decisions = _json_decisions(data)
    if decisions is None:
        return None
    return Verdict(decisions, enough, missing)


def _json_decisions(data: Any) -> list[Decision] | None:
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


_DECISION_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _salvaged_decisions(text: str) -> list[Decision]:
    """The complete decision objects in a JSON reply that does not parse
    as a whole — one a model's output limit cut off mid-object, as seen
    with qwen3 judging eight findings. Each ``{...}`` holding an id and
    a keep is a decision; the cut-off one is left for the judge to ask
    about again."""
    decisions = []
    for match in _DECISION_OBJECT_RE.finditer(text):
        try:
            found = _json_decisions([json.loads(match.group(0))])
        except ValueError:
            continue
        if found:
            decisions.extend(found)
    return decisions


def parse_judgements(text: str) -> list[Decision]:
    """The judge's decisions from its reply, however it wrote them: the
    decisions of :func:`parse_verdict`."""
    return parse_verdict(text).decisions


def parse_verdict(text: str) -> Verdict:
    """The judge's reply, read: its decisions and its verdict on the whole.

    The prompt asks for one JSON object; that is accepted bare, fenced,
    or surrounded by prose, as is a bare JSON list of decisions. A
    decision's ``score`` (or ``grade``) is optional: a reply without
    grades reads as before, and its decisions rank as the default grade.
    The object's ``enough`` and ``missing`` are optional too: a reply
    without them gives no verdict (``enough`` is ``None``). A reply cut
    off mid-JSON gives the decisions that are complete and no verdict.
    Failing JSON, every line that reads as ``<id> <keep|ignore|true|false>
    <reason>`` is an ungraded decision. Raises ``ValueError`` when no
    decision can be read."""
    text = text.strip()
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start, end = text.find(start_char), text.rfind(end_char)
        if 0 <= start < end:
            try:
                verdict = _json_verdict(json.loads(text[start : end + 1]))
            except ValueError:
                verdict = None
            if verdict is not None:
                return verdict
    decisions = _salvaged_decisions(text)
    if decisions:
        return Verdict(decisions)
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        label, word, reason = match.groups()
        decisions.append(
            Decision(
                label,
                word.lower() in _KEEP_WORDS,
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
    return Verdict(decisions)


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
    may override :meth:`decide`. A finding the reply left undecided
    (the model skipped it, or its output limit cut the JSON off) is
    asked about again, with the other undecided ones, up to ``retries``
    times; ``batch_size`` judges at most that many findings per model
    call. When the model fails or still leaves a finding undecided, the
    finding is kept with a reason starting ``not judged``; only a model
    that cannot be built at all raises (a ``ValueError``). Model calls
    are taken one at a time.
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
        retries: Annotated[
            int,
            "How many times to ask again about the findings a reply left "
            "undecided (skipped, or cut off by the model's output limit)",
        ] = 1,
        batch_size: Annotated[
            int | None,
            "Judge at most this many findings per model call (None = all of "
            "a call's findings at once)",
        ] = None,
    ) -> None:
        super().__init__()
        self.retries = max(0, retries)
        self.batch_size = max(1, batch_size) if batch_size is not None else None
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
            lines.append(f"{head}\n{preview(finding.text, self.max_chars)}")
        return "\n".join(lines)

    def decide(
        self, prompt: str, labelled: list[tuple[str, Finding]]
    ) -> dict[str, Decision] | Verdict:
        """The decisions for *labelled*: one model call on *prompt*, read
        as a :class:`Verdict` (decisions plus whether what is kept would
        answer the question). A subclass may return the decisions alone,
        by label (``f1``, …), and give no verdict. Raises when the reply
        cannot be read."""
        with self._lock:
            reply = self.generate(prompt)
        return parse_verdict(reply)

    def judge(
        self,
        findings: Iterable[Any],
        question: str,
        looking_for: str = "",
        known: Iterable[str] = (),
        seen: Iterable[str] = (),
    ) -> Judged:
        """Judge *findings* for *question*, in order. Findings whose key is
        in *seen* are left out; a finding the model gave no decision for,
        or every finding when the model failed, is kept with a reason that
        starts "not judged". The list's ``enough`` and ``missing`` are the
        judge's verdict on what is kept, from the last reply that gave one
        (``None`` when none did)."""
        skip = {str(key) for key in seen}
        labelled: list[tuple[str, Finding]] = []
        for finding in as_findings(list(findings or [])):
            if finding.key in skip:
                continue
            labelled.append((f"f{len(labelled) + 1}", finding))
        judged = Judged()
        if not labelled:
            return judged
        decisions: dict[str, Decision] = {}
        undecided = "not judged: the model gave no decision for it"
        size = self.batch_size or len(labelled)
        for start in range(0, len(labelled), size):
            batch = labelled[start : start + size]
            # The findings this batch has yet to decide on: all of them,
            # then whatever each reply skipped or cut off, asked about
            # again on their own with their labels unchanged.
            for _attempt in range(self.retries + 1):
                pending = [(label, f) for label, f in batch if label not in decisions]
                if not pending:
                    break
                prompt = self.prompt(question, looking_for, known, pending)
                try:
                    found = self.decide(prompt, pending)
                except _JudgeModelError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Judging findings failed; keeping them", exc_info=True
                    )
                    undecided = f"not judged: {_one_line(exc)}"
                    break
                if isinstance(found, Verdict):
                    if found.enough is not None:
                        judged.enough, judged.missing = found.enough, found.missing
                    found = {_label_of(d.id): d for d in found.decisions}
                wanted = {label for label, _ in pending}
                decisions.update(
                    (label, d) for label, d in found.items() if label in wanted
                )
                if any(label not in decisions for label, _ in pending):
                    logger.info(
                        "The judge left %d of %d finding(s) undecided",
                        sum(1 for label, _ in pending if label not in decisions),
                        len(pending),
                    )
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
