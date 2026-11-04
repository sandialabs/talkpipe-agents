"""What a finding tool returns: :class:`Finding`, and how other shapes
(strings, dicts, talkpipe search results) become one.

Also what the model reads from a finding tool when no session judges
its results (:func:`render_findings`), and the score floor a finding
tool may put under its results (:func:`below_score_floor`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from talkpipe.search.abstract import SearchResult

# How much of a finding the judge and the agent are shown. Enough for a
# search chunk at talkpipe's default indexing settings; a shorter preview
# hides the part of a finding that would have made it worth keeping.
DEFAULT_MAX_CHARS = 1000

# Document fields a search hit's text, label and source are read from,
# in order of preference.
_TEXT_FIELDS = ("content", "text", "shingle")
_LABEL_FIELDS = ("title", "label", "filename", "name")
_SOURCE_FIELDS = ("source", "path", "url")


def _first(doc: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = doc.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def preview(text: str, max_chars: int) -> str:
    """*text* on one line, clipped to *max_chars* with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars] + "…"


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

    @property
    def search_score(self) -> float | None:
        """The score the finding's tool gave it, when it gave one: the
        search hit's score (``meta["result"]``), else a numeric
        ``meta["score"]``; ``None`` for a finding with no score. Not the
        judge's relevance grade (``FindingsSession.score_of``): this is
        the retrieval's own similarity, on the tool's scale."""
        result = self.meta.get("result")
        if isinstance(result, SearchResult):
            return float(result.score)
        value = self.meta.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

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


def below_score_floor(
    findings: list[Finding], min_score: float | None
) -> tuple[list[Finding], int]:
    """*findings* without those scoring under *min_score*, and how many
    were left out. A finding with no score always passes; with
    *min_score* ``None`` nothing is left out."""
    if min_score is None:
        return findings, 0
    kept = [
        f for f in findings if f.search_score is None or f.search_score >= min_score
    ]
    return kept, len(findings) - len(kept)


def floor_note(below_floor: int) -> str:
    """What the model is told when a tool's score floor left results out."""
    return (
        f"{below_floor} result(s) scored below the tool's floor and were left out."
        if below_floor
        else ""
    )


def render_findings(
    findings: list[Finding], max_chars: int = DEFAULT_MAX_CHARS, below_floor: int = 0
) -> str:
    """Findings as plain text — what the model reads from a finding
    tool used without a findings session (no ``JudgeFindings`` hook).
    *below_floor* is how many results the tool's score floor left out,
    which the model is told so it can try other terms."""
    if not findings:
        return " ".join(filter(None, ["Nothing found.", floor_note(below_floor)]))
    blocks = []
    for i, finding in enumerate(findings, 1):
        head = f"[{i}]" + (f" {finding.label}" if finding.label else "")
        blocks.append(f"{head}\n{preview(finding.text, max_chars)}")
    if below_floor:
        blocks.append(floor_note(below_floor))
    return "\n\n".join(blocks)


@dataclass
class FoundFindings:
    """What one finding tool call stashed for the :class:`JudgeFindings`
    hook: the findings above the tool's score floor, and how many
    results the floor left out."""

    findings: list[Finding]
    below_floor: int = 0
