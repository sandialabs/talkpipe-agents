"""The shipped-tool examples in docs/tools-and-skills.md, run as written.

Every example under "Shipped tools" is a complete chatterlang script
that grants one shipped tool (the notepad example, all six). This
module pulls each script out of the page, registers the scripted model
under the ``"ollama"`` source the scripts name, has it make the call the
docs describe, and checks that the real tool returned the reply the
docs quote. A shipped tool without an example, an example without a
scenario here, or a reply the docs quote wrongly all fail.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import requests
from talkpipe.chatterlang import compiler

from talkpipe_agents import agent_tool_names
from talkpipe_agents.testing import scripted_source

DOC = Path(__file__).parent.parent / "docs" / "tools-and-skills.md"

SHIPPED = frozenset(
    {
        "ask_user",
        "think",
        "calculator",
        "current_time",
        "days_between",
        "regex_extract",
        "read_file",
        "list_files",
        "search_files",
        "fetch_url",
        "save_note",
        "read_notes",
        "summarize_notes",
        "update_note",
        "delete_note",
        "clear_notes",
    }
)

# The environment the docs ask the reader to provide: a notes.txt in the
# current directory for the file examples, a terminal for ask_user, and
# the network for fetch_url (faked at requests.get, as the other web
# tests do).
NOTES = "Dentist on Tuesday at 9.\nRenew the car registration before the 30th.\n"
TYPED_ANSWER = "Tuesday at 10"
EXAMPLE_PAGE = (
    b"<html><head><title>Example Domain</title></head>"
    b"<body><h1>Example Domain</h1>"
    b"<p>This domain is for use in illustrative examples in documents.</p>"
    b"</body></html>"
)


class Call(NamedTuple):
    tool: str
    args: dict[str, Any]
    reply: list[str | re.Pattern[str]]
    """What the tool's reply must contain. A string is also what the docs
    quote, so it must appear in the page too; a pattern is checked
    against the reply alone (for output that varies, like the time)."""


class Example(NamedTuple):
    calls: list[Call]
    answer: str


EXAMPLES: dict[tuple[str, ...], Example] = {
    ("ask_user",): Example(
        [Call("ask_user", {"question": "When is your next review?"}, [TYPED_ANSWER])],
        "The room is booked for Tuesday at 10.",
    ),
    ("think",): Example(
        [
            Call(
                "think",
                {"thought": "1001 = 7 * 11 * 13, so it is not prime"},
                ["(thought recorded — continue when ready)"],
            )
        ],
        "No, 1001 is 7 times 11 times 13.",
    ),
    ("calculator",): Example(
        [Call("calculator", {"expression": "17 * 23 + 5"}, ["396"])],
        "17 * 23 + 5 is 396.",
    ),
    ("current_time",): Example(
        [
            Call(
                "current_time",
                {"timezone": "Asia/Tokyo"},
                [re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+09:00 \(\w+day\)$")],
            )
        ],
        "It is Saturday in Tokyo.",
    ),
    ("days_between",): Example(
        [Call("days_between", {"start": "2026-01-31", "end": "2026-03-01"}, ["29"])],
        "There are 29 days between them.",
    ),
    ("regex_extract",): Example(
        [
            Call(
                "regex_extract",
                {
                    "pattern": r"[\w.]+@[\w.]+",
                    "text": "contact ann@example.com or bob@example.org",
                },
                ["ann@example.com", "bob@example.org"],
            )
        ],
        "ann@example.com and bob@example.org",
    ),
    ("read_file",): Example(
        [Call("read_file", {"path": "notes.txt"}, [re.compile(re.escape(NOTES))])],
        "A dentist visit and a car registration to renew.",
    ),
    ("list_files",): Example(
        [
            Call(
                "list_files",
                {"directory": "."},
                ["notes.txt", re.compile(rf"^notes\.txt\t{len(NOTES)} bytes$")],
            )
        ],
        "Just notes.txt.",
    ),
    ("search_files",): Example(
        [
            Call(
                "search_files",
                {"pattern": "car registration", "directory": "."},
                ["notes.txt:2: Renew the car registration before the 30th."],
            )
        ],
        "notes.txt mentions the car registration.",
    ),
    ("fetch_url",): Example(
        [Call("fetch_url", {"url": "https://example.com/"}, ["Example Domain"])],
        "It is the Example Domain placeholder page.",
    ),
    (
        "save_note",
        "read_notes",
        "summarize_notes",
        "update_note",
        "delete_note",
        "clear_notes",
    ): Example(
        [
            Call(
                "save_note",
                {"note": "Day 1: Alfama and the castle"},
                ["(note 1 saved)"],
            ),
            Call("save_note", {"note": "Day 2: Belém"}, ["(note 2 saved)"]),
            Call("summarize_notes", {}, ["2 notes, 40 characters in total:"]),
            Call(
                "update_note",
                {"number": 2, "note": "Day 2: Belém, then Sintra"},
                ["(note 2 updated)"],
            ),
            Call(
                "read_notes",
                {},
                ["1. Day 1: Alfama and the castle", "2. Day 2: Belém, then Sintra"],
            ),
            Call("delete_note", {"number": 1}, ["(note 1 deleted)"]),
            Call("clear_notes", {}, ["(1 notes deleted)"]),
        ],
        "Day 1: Alfama and the castle. Day 2: Belém, then Sintra.",
    ),
}


def _shipped_tools_section(text: str) -> str:
    sections = re.split(r"^(?=## )", text, flags=re.MULTILINE)
    [section] = [s for s in sections if s.startswith("## Shipped tools\n")]
    return section


SECTION = _shipped_tools_section(DOC.read_text(encoding="utf-8"))


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced code block as ``(language, body)``."""
    blocks: list[tuple[str, str]] = []
    language: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if language is None:
                language, body = line[3:].strip(), []
            else:
                blocks.append((language, "\n".join(body)))
                language = None
        elif language is not None:
            body.append(line)
    assert language is None, "unclosed fence"
    return blocks


# Bare ``` fences only: the Python snippet in the section is not a script.
SCRIPTS = [body for language, body in _fenced_blocks(SECTION) if not language]


def _granted_tools(script: str) -> tuple[str, ...]:
    assert "llmAgent[" in script, script
    match = re.search(r"tools=\[(.*?)\]", script, flags=re.DOTALL)
    assert match, script
    return tuple(re.findall(r'"(\w+)"', match.group(1)))


EXAMPLE_SCRIPTS = {_granted_tools(script): script for script in SCRIPTS}


def test_every_shipped_tool_has_an_example_and_every_example_a_scenario():
    assert set(agent_tool_names()) >= SHIPPED
    assert len(EXAMPLE_SCRIPTS) == len(SCRIPTS), "two examples grant the same tools"
    assert set(EXAMPLE_SCRIPTS) == set(EXAMPLES)
    assert set().union(*EXAMPLE_SCRIPTS) == SHIPPED


def _fake_get(url, headers=None, timeout=None, **_):
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response._content = EXAMPLE_PAGE
    return response


@pytest.mark.parametrize("tools", list(EXAMPLES), ids=lambda tools: tools[0])
def test_shipped_tool_example_runs_as_written(
    tools, scripted_model, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text(NOTES, encoding="utf-8")
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt="": TYPED_ANSWER)
    monkeypatch.setattr("talkpipe.data.html.requests.get", _fake_get)

    example = EXAMPLES[tools]
    for call in example.calls:
        scripted_model.call(call.tool, **call.args)
    scripted_model.say(example.answer)
    with scripted_source("ollama", model=scripted_model):
        assert list(compiler.compile(EXAMPLE_SCRIPTS[tools])()) == [example.answer]
    assert not scripted_model.remaining

    page = " ".join(SECTION.split())  # the docs wrap quoted replies across lines
    assert [r.name for r in scripted_model.tool_results] == [
        c.tool for c in example.calls
    ]
    for call, result in zip(example.calls, scripted_model.tool_results, strict=True):
        assert result.status == "success", result
        for expected in call.reply:
            if isinstance(expected, re.Pattern):
                assert expected.search(result.text), (call.tool, result.text)
            else:
                assert expected in result.text, (call.tool, result.text)
                assert expected in page, f"the docs no longer quote {expected!r}"
