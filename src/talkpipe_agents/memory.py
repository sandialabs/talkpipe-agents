"""``SearchMemoryStore`` — a talkpipe search as a Strands memory store.

Strands' ``MemoryManager`` searches its stores with the user's message
before each model call and folds the hits into the model input as a
``<memory>`` block, without touching the conversation history. Backed
by a talkpipe search index, that is retrieval the agent never has to
ask for: the question's best matches are in front of the model from
the first turn, and the finding tools are left for what those miss.

``SearchMemoryStore`` wraps anything that searches — a talkpipe
``TextSearchable`` (``text_search(query, limit)``) or a callable taking
the query and a result limit — and turns each hit into a memory entry
the way the findings loop turns it into a :class:`Finding`: hits of
every shape ``as_finding`` accepts, a ``min_score`` floor, and the
label and source kept with the text so the model can cite them.
``memory_manager`` builds the manager ``llmAgent`` and ``AgentChat``
install for their ``memory=`` argument, injection on and the manager's
own ``search_memory`` tool off, so that what the agent searches for
itself still goes through its finding tools and their judge.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from strands.memory import MemoryEntry, MemoryManager, SearchOptions
from strands.memory.types import MemoryStore

from .findings import Finding, as_finding, below_score_floor

DEFAULT_MAX_SEARCH_RESULTS = 5


def _entry_text(finding: Finding) -> str:
    head = finding.label
    if finding.source and finding.source != finding.label:
        head = f"{head} ({finding.source})" if head else finding.source
    return f"{head}\n{finding.text}" if head else finding.text


class SearchMemoryStore:
    """A Strands ``MemoryStore`` over a talkpipe search.

    *search* is a ``TextSearchable`` (an object with ``text_search(query,
    limit)``, such as a ``WhooshFullTextIndex``) or any callable taking
    the query text and a result limit and returning hits in any shape a
    finding tool may return: ``SearchResult``s, dicts, ``Finding``s or
    strings. A vector index is wrapped by a callable that embeds the
    query and calls its ``vector_search``. *name* identifies the store
    in the injected ``<memory>`` block; *description* is for the
    manager's search tool, when a host turns that on;
    *max_search_results* is the hits per search; *min_score* leaves out
    hits scored below it (:attr:`Finding.search_score`). The store is
    read-only; the search is run as given, synchronously.
    """

    writable = False
    extraction = None

    def __init__(
        self,
        search: Callable[[str, int], Iterable[Any]] | Any,
        *,
        name: str = "documents",
        description: str | None = None,
        max_search_results: int | None = DEFAULT_MAX_SEARCH_RESULTS,
        min_score: float | None = None,
    ) -> None:
        if hasattr(search, "text_search"):
            self._search: Callable[[str, int], Iterable[Any]] = search.text_search
        elif callable(search):
            self._search = search
        else:
            raise TypeError(
                "search must be a TextSearchable (with text_search) or a callable "
                "taking the query and a result limit"
            )
        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.min_score = min_score

    def findings(self, query: str, limit: int | None = None) -> list[Finding]:
        """The hits for *query* as findings, the score floor applied."""
        if limit is None:
            limit = self.max_search_results or DEFAULT_MAX_SEARCH_RESULTS
        hits = self._search(query, limit)
        found = [f for f in (as_finding(h, self.name) for h in hits) if f is not None]
        return below_score_floor(found, self.min_score)[0]

    async def search(
        self, query: str, options: SearchOptions | None = None
    ) -> list[MemoryEntry]:
        limit = (options or {}).get("max_search_results")
        return [
            MemoryEntry(
                content=_entry_text(finding),
                metadata={
                    "label": finding.label,
                    "source": finding.source,
                    "key": finding.key,
                    "score": finding.search_score,
                },
            )
            for finding in self.findings(query, limit)
        ]


def memory_manager(
    memory: MemoryStore | MemoryManager | Sequence[MemoryStore] | None,
) -> MemoryManager | None:
    """The ``MemoryManager`` for an agent's ``memory=`` argument: ``None``
    for none, a manager as given, or one built over a store (or a list
    of stores) with injection on and the manager's ``search_memory``
    tool off."""
    if memory is None or isinstance(memory, MemoryManager):
        return memory
    stores = list(memory) if isinstance(memory, Sequence) else [memory]
    return MemoryManager(stores=stores, search_tool_config=False, injection=True)
