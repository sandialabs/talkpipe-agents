"""``SearchMemoryStore``: a talkpipe search as a Strands memory store,
whose hits are put before the model without a tool call. Offline: the
search is a stub, the model is scripted."""

from __future__ import annotations

import asyncio

import pytest
from strands.memory import MemoryManager
from talkpipe.search.abstract import SearchResult

from talkpipe_agents import AgentChat, LLMAgent, SearchMemoryStore, memory_manager

HITS = [
    SearchResult(
        score=0.9,
        doc_id="d1",
        document={
            "content": "Paris is the capital.",
            "title": "Geo",
            "source": "/g.txt",
        },
    ),
    SearchResult(score=0.2, doc_id="d2", document={"content": "A weak match."}),
]


class _Index:
    """The ``TextSearchable`` shape: ``text_search(query, limit)``."""

    def __init__(self):
        self.queries: list[tuple[str, int]] = []

    def text_search(self, query, limit=10):
        self.queries.append((query, limit))
        return HITS[:limit]


def test_hits_become_entries_with_label_and_source():
    index = _Index()
    store = SearchMemoryStore(index, name="notes")
    entries = asyncio.run(store.search("capital"))
    assert index.queries == [("capital", 5)]
    assert [e.content for e in entries] == [
        "Geo (/g.txt)\nParis is the capital.",
        "A weak match.",
    ]
    assert entries[0].metadata == {
        "label": "Geo",
        "source": "/g.txt",
        "key": "d1",
        "score": 0.9,
    }
    assert (store.name, store.writable, store.max_search_results) == ("notes", False, 5)


def test_the_limit_and_the_score_floor_apply():
    index = _Index()
    store = SearchMemoryStore(index, min_score=0.5, max_search_results=2)
    assert [e.content for e in asyncio.run(store.search("q"))] == [
        "Geo (/g.txt)\nParis is the capital."
    ]
    asyncio.run(store.search("q", {"max_search_results": 1}))
    assert index.queries == [("q", 2), ("q", 1)]


def test_a_callable_search_returns_hits_of_any_shape():
    store = SearchMemoryStore(lambda q, n: [f"{q} #{i}" for i in range(n)])
    assert [
        e.content for e in asyncio.run(store.search("x", {"max_search_results": 2}))
    ] == [
        "documents\nx #0",
        "documents\nx #1",
    ]
    assert [f.label for f in store.findings("x", 1)] == ["documents"]
    with pytest.raises(TypeError, match="text_search"):
        SearchMemoryStore(object())


def test_memory_manager_builds_an_injecting_manager_without_the_search_tool():
    assert memory_manager(None) is None
    manager = MemoryManager(stores=[SearchMemoryStore(_Index())])
    assert memory_manager(manager) is manager
    built = memory_manager([SearchMemoryStore(_Index(), name="a")])
    assert isinstance(built, MemoryManager)
    assert built.tools == []  # no search_memory tool: finding tools do that
    assert memory_manager(SearchMemoryStore(_Index())) is not None


def _memory_block(messages):
    return "\n".join(
        block["text"]
        for message in messages
        for block in message["content"]
        if "text" in block and "<memory>" in block["text"]
    )


def test_llm_agent_puts_the_hits_before_the_model(scripted_model):
    index = _Index()
    scripted_model.say("Paris.")
    step = LLMAgent(
        source="scripted", model="any", memory=SearchMemoryStore(index, name="notes")
    )
    assert list(step(["What is the capital of France?"])) == ["Paris."]
    assert index.queries == [("What is the capital of France?", 5)]
    block = _memory_block(scripted_model.prompts[0])
    assert '<entry source="notes">' in block
    assert "Paris is the capital." in block


def test_agent_chat_takes_memory_too(scripted_model):
    index = _Index()
    scripted_model.say("Paris.")
    chat = AgentChat(source="scripted", model="any", memory=[SearchMemoryStore(index)])
    assert chat.ask("capital?") == "Paris."
    assert index.queries == [("capital?", 5)]
    assert "Paris is the capital." in _memory_block(scripted_model.prompts[0])
    # Injection is not history: the durable messages hold the question alone.
    assert chat.messages[0]["content"] == [{"text": "capital?"}]
