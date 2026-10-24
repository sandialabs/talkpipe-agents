"""The ``ollama`` source's provider, driven through a real Strands agent
against a fake Ollama client: what it stores from a streamed reply and
what it sends back on the next request. Fully offline."""

from __future__ import annotations

from typing import ClassVar

import ollama
import pytest
from strands import Agent, tool

from talkpipe_agents import build_agent_model


@tool
def lookup(term: str) -> str:
    """Look *term* up.

    Args:
        term: What to look up.
    """
    return f"found {term}"


def _chunk(done: bool = False, **message: object) -> ollama.ChatResponse:
    message.setdefault("content", "")  # as Ollama streams it
    return ollama.ChatResponse(
        model="m",
        done=done,
        done_reason="stop" if done else None,
        message=ollama.Message(role="assistant", **message),
        prompt_eval_count=1 if done else None,
        eval_count=1 if done else None,
        total_duration=1_000_000 if done else None,
    )


def _call(name: str, **arguments: object) -> ollama.Message.ToolCall:
    return ollama.Message.ToolCall(
        function=ollama.Message.ToolCall.Function(name=name, arguments=arguments)
    )


@pytest.fixture
def fake_ollama(monkeypatch):
    """Replies streamed from a list of turns (each a list of chunks); the
    requests the provider sent are recorded in ``requests``."""

    class FakeClient:
        turns: ClassVar[list[list[ollama.ChatResponse]]] = []
        requests: ClassVar[list[dict]] = []

        def __init__(self, *args, **kwargs):
            pass

        async def chat(self, **request):
            FakeClient.requests.append(request)
            chunks = FakeClient.turns.pop(0)

            async def stream():
                for chunk in chunks:
                    yield chunk

            return stream()

    monkeypatch.setattr(ollama, "AsyncClient", FakeClient)
    return FakeClient


# A turn that says something, thinks, and calls the tool twice in parallel;
# then the answer.
THINKING_TURNS = [
    [
        _chunk(thinking="Look both up, "),
        _chunk(thinking="then keep them."),
        _chunk(content="Looking."),
        _chunk(tool_calls=[_call("lookup", term="a"), _call("lookup", term="b")]),
        _chunk(done=True),
    ],
    [_chunk(content="Both found."), _chunk(done=True)],
]


def _run(fake, turns, params=None):
    fake.turns = [list(turn) for turn in turns]
    fake.requests = []
    model = build_agent_model("ollama", "m", params=params)
    agent = Agent(model=model, tools=[lookup], callback_handler=None)
    result = agent("Find a and b.")
    return agent, str(result).strip(), fake.requests


def _expected_second_request(thinking: str | None) -> list[dict]:
    turn = {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {"function": {"name": "lookup", "arguments": {"term": "a"}}},
            {"function": {"name": "lookup", "arguments": {"term": "b"}}},
        ],
    }
    if thinking is not None:
        turn["thinking"] = thinking
    return [
        {"role": "user", "content": "Find a and b."},
        turn,
        {"role": "tool", "content": "found a", "tool_name": "lookup"},
        {"role": "tool", "content": "found b", "tool_name": "lookup"},
    ]


def test_reply_without_thinking_goes_back_as_one_turn_with_named_results(
    fake_ollama,
):
    turns = [[c.model_copy(deep=True) for c in turn] for turn in THINKING_TURNS]
    for turn in turns:
        for chunk in turn:
            chunk.message.thinking = None
    agent, answer, requests = _run(fake_ollama, turns)
    assert answer == "Both found."
    assert len(requests) == 2
    assert requests[1]["messages"] == _expected_second_request(thinking=None)
    stored = agent.messages[1]["content"]
    assert not any("reasoningContent" in block for block in stored)


def test_streamed_thinking_is_stored_and_sent_back(fake_ollama):
    agent, answer, requests = _run(fake_ollama, THINKING_TURNS)
    assert answer == "Both found."  # the reasoning is not part of the reply
    reasoning = [
        block["reasoningContent"]["reasoningText"]["text"]
        for block in agent.messages[1]["content"]
        if "reasoningContent" in block
    ]
    assert reasoning == ["Look both up, then keep them."]
    assert requests[1]["messages"] == _expected_second_request(
        thinking="Look both up, then keep them."
    )


def test_replay_thinking_false_stops_the_replay(fake_ollama):
    agent, _, requests = _run(
        fake_ollama, THINKING_TURNS, params='{"replay_thinking": false, "num_ctx": 8}'
    )
    # still stored on the message, just not sent back; not an Ollama option
    assert any("reasoningContent" in block for block in agent.messages[1]["content"])
    assert requests[1]["messages"] == _expected_second_request(thinking=None)
    assert requests[1]["options"] == {"num_ctx": 8}
