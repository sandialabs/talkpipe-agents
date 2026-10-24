"""The agentic segment layer, fully offline.

The model is the package's own ``talkpipe_agents.testing.ScriptedModel``,
reached through the ``scripted_model`` fixture the package installs as a
pytest plugin: it replays scripted turns — text replies and tool calls —
through the real agent loop as model source "scripted". The example
agent throughout is the calculator clerk from the plan: a real ``add``
tool the scripted model calls before answering.
"""

from __future__ import annotations

from collections import deque

import pytest
from strands import tool
from strands.types.tools import ToolContext
from talkpipe.chatterlang import compiler

from talkpipe_agents import (
    AgentChat,
    LLMAgent,
    ScriptTool,
    UnknownAgentToolError,
    agent_model_sources,
    agent_tool_names,
    build_agent_model,
    make_ask_user,
    make_script_tool,
    register_agent_model,
    register_agent_tool,
    resolve_agent_tools,
    strip_tool_leak,
)
from talkpipe_agents.testing import (
    ScriptedModel,
    ScriptedToolErrorWarning,
    ToolResult,
    scripted_source,
)

added: list[tuple] = []


@register_agent_tool
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    added.append((a, b))
    return a + b


@register_agent_tool
@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@pytest.fixture(autouse=True)
def fresh_add_log():
    added.clear()
    yield
    added.clear()


def test_calculator_clerk_calls_tool_and_answers(scripted_model, capsys):
    scripted_model.script.extend(
        [("tool", "add", {"a": 2, "b": 3}), ("text", "2 + 3 = 5")]
    )
    chat = AgentChat(source="scripted", model="test", tools=["add"])
    answer = chat.ask("What is 2 plus 3?")
    assert answer == "2 + 3 = 5"
    assert added == [(2, 3)]  # the real tool function executed
    roles = [(m["role"], [k for c in m["content"] for k in c]) for m in chat.messages]
    assert ("assistant", ["toolUse"]) in roles
    assert ("user", ["toolResult"]) in roles
    assert capsys.readouterr().out == ""  # no default printing handler


def test_llm_agent_segment_from_chatterlang(scripted_model):
    scripted_model.script.extend(
        [("tool", "add", {"a": 2, "b": 3}), ("text", "2 + 3 = 5")]
    )
    script = (
        'INPUT FROM echo[data="What is 2 plus 3?"] '
        '| llmAgent[source="scripted", model="test", tools=["add"]]'
    )
    out = list(compiler.compile(script)())
    assert out == ["2 + 3 = 5"]
    assert added == [(2, 3)]


def test_segment_set_as_keeps_item(scripted_model):
    scripted_model.script.append(("text", "fine, thanks"))
    seg = LLMAgent(source="scripted", model="test", field="q", set_as="answer")
    out = list(seg.transform(iter([{"q": "how are you?"}])))
    assert out == [{"q": "how are you?", "answer": "fine, thanks"}]


def test_missing_field_is_an_error_not_a_prompt_of_none(scripted_model):
    scripted_model.script.append(("text", "an answer to nothing"))
    seg = LLMAgent(source="scripted", model="test", field="question")
    with pytest.raises(ValueError, match="field='question' not found"):
        list(seg.transform(iter([{"q": "how are you?"}])))
    assert scripted_model.calls == []  # the model was never asked


def test_tool_error_is_a_progress_line(scripted_model):
    scripted_model.script.extend(
        [("tool", "boom", {}), ("text", "the tool failed, sorry")]
    )
    told: list[str] = []
    seg = LLMAgent(
        source="scripted", model="test", tools=["boom"], progress=told.append
    )
    with pytest.warns(ScriptedToolErrorWarning):
        assert list(seg.transform(iter(["try the tool"]))) == ["the tool failed, sorry"]
    assert told == ["agent → boom({})", "agent ← boom: error: RuntimeError - kaboom"]


def test_tool_error_returns_to_model_as_observation(scripted_model):
    scripted_model.script.extend(
        [("tool", "boom", {}), ("text", "the tool failed, sorry")]
    )
    chat = AgentChat(source="scripted", model="test", tools=["boom"])
    # deliberate recovery test: the errored scripted call is also a warning
    with pytest.warns(ScriptedToolErrorWarning, match="'boom'"):
        assert chat.ask("try the tool") == "the tool failed, sorry"
    results = [
        c["toolResult"]
        for m in chat.messages
        for c in m["content"]
        if "toolResult" in c
    ]
    assert results
    assert results[0]["status"] == "error"
    assert "kaboom" in str(results[0]["content"])


def test_turn_limit_stops_loop(scripted_model):
    scripted_model.script.extend([("tool", "add", {"a": 1, "b": 1})] * 5)
    chat = AgentChat(source="scripted", model="test", tools=["add"], max_turns=2)
    chat.ask("loop forever")
    assert chat.last_result.stop_reason == "limit_turns"
    assert len(added) <= 2


def test_multi_turn_segment_keeps_one_conversation(scripted_model):
    scripted_model.script.extend([("text", "one"), ("text", "two")])
    seg = LLMAgent(source="scripted", model="test", multi_turn=True)
    assert list(seg.transform(iter(["a", "b"]))) == ["one", "two"]
    # second model call saw the first exchange in its history
    assert scripted_model.calls == [1, 3]


def test_fresh_agent_per_item_by_default(scripted_model):
    scripted_model.script.extend([("text", "one"), ("text", "two")])
    seg = LLMAgent(source="scripted", model="test")
    assert list(seg.transform(iter(["a", "b"]))) == ["one", "two"]
    assert scripted_model.calls == [1, 1]


def test_think_blocks_stripped(scripted_model):
    scripted_model.script.append(("text", "<think>secret</think>42"))
    chat = AgentChat(source="scripted", model="test")
    assert chat.ask("answer?") == "42"


def test_leaked_tool_call_stripped_and_scrubbed_from_history(scripted_model):
    """Raw tool-call syntax a server failed to parse is removed from the
    displayed answer AND from the stored conversation,
    so later turns don't see the corrupted text."""
    scripted_model.script.append(
        (
            "text",
            (
                "Closed at $271.58.⦗ to=functions.get_quotes <|constrain|>json"
                '<|message|>{"history":10,"ticker":"AMZN"}'
            ),
        )
    )
    chat = AgentChat(source="scripted", model="test")
    assert chat.ask("AMZN?") == "Closed at $271.58."
    stored = " ".join(
        c["text"] for m in chat.messages for c in m["content"] if "text" in c
    )
    assert "to=functions" not in stored
    assert "<|" not in stored


def test_json_tool_call_leak_scrubbed_and_retried(scripted_model):
    """A model that writes a tool call as prose instead of using the
    provider's tool-call channel must not have that JSON surface as the
    answer; scrubbing empties the reply, so the plain-prose retry runs."""
    scripted_model.script.extend(
        [
            ("text", '{"name": "add", "parameters": {"a": 1, "b": 2}}'),
            ("text", "The sum is 3."),
        ]
    )
    chat = AgentChat(source="scripted", model="test", tools=["add"])
    assert chat.ask("1+2?") == "The sum is 3."
    stored = " ".join(
        c["text"] for m in chat.messages for c in m["content"] if "text" in c
    )
    assert '"parameters"' not in stored


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # leaks: a name plus an argument object and nothing else
        ('{"name": "date", "parameters": {"f": "2023-04-01"}}', ""),
        ('Let me check. {"name": "add", "arguments": {"a": 1}}', "Let me check. "),
        (
            '{"type": "function", "function": {"name": "add", "arguments": "{}"}}',
            "",
        ),
        # data that merely looks JSON-ish stays exactly as it is
        ('{"name": "Ada", "age": 3}', '{"name": "Ada", "age": 3}'),
        ('{"name": "Ada"}', '{"name": "Ada"}'),
        ('{"parameters": {"a": 1}}', '{"parameters": {"a": 1}}'),
        ("The name is Ada.", "The name is Ada."),
        ("{not json at all}", "{not json at all}"),
    ],
)
def test_strip_tool_leak_on_json_shapes(text, expected):
    assert strip_tool_leak(text) == expected


def test_empty_reply_retried_with_nudge(scripted_model):
    """A reply with no user-visible text (only reasoning) gets one
    plain-prose retry through the same conversation."""
    scripted_model.script.extend(
        [("text", "<think>only thoughts</think>"), ("text", "here is the answer")]
    )
    chat = AgentChat(source="scripted", model="test")
    assert chat.ask("answer?") == "here is the answer"
    # the retry ran within the same conversation: 1 message on the
    # first call, 3 (question, empty reply, nudge) on the second
    assert scripted_model.calls == [1, 3]


def test_still_empty_reply_yields_visible_notice(scripted_model):
    scripted_model.script.extend([("text", ""), ("text", "")])
    chat = AgentChat(source="scripted", model="test")
    answer = chat.ask("answer?")
    assert answer  # an interactive chat never answers with silence
    assert "no answer" in answer


class _UnparsedToolCallOnce(ScriptedModel):
    """Fails its first model call the way Ollama rejects a truncated
    tool call, then plays its script."""

    failed = False

    async def stream(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise RuntimeError(
                "error parsing tool call: raw='{\"a\": 2', "
                "err=unexpected end of JSON input (status code: -1)"
            )
        async for event in super().stream(*args, **kwargs):
            yield event


@pytest.fixture
def no_retry_delay(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr("strands.event_loop._retry.asyncio.sleep", instant)


def test_unparsed_tool_call_is_retried(no_retry_delay):
    """A tool call the server could not parse is a sampling accident:
    the model call is retried instead of ending the run."""
    model = _UnparsedToolCallOnce().call("add", a=2, b=3).say("5")
    with scripted_source(model=model):
        seg = LLMAgent(source="scripted", model="test", tools=["add"])
        assert list(seg.transform(iter(["2 + 3?"]))) == ["5"]
        chat = AgentChat(source="scripted", model="test", tools=["add"])
        model.failed = False
        model.call("add", a=1, b=1).say("2")
        assert chat.ask("1 + 1?") == "2"
    assert added == [(2, 3), (1, 1)]


def test_other_model_errors_are_not_retried():
    class Broken(ScriptedModel):
        async def stream(self, *args, **kwargs):
            raise RuntimeError("model not found")
            yield

    with (
        scripted_source(model=Broken()),
        pytest.raises(Exception, match="model not found"),
    ):
        AgentChat(source="scripted", model="test").ask("hi")


def test_llm_agent_segment_retries_empty_reply(scripted_model):
    scripted_model.script.extend(
        [("text", "<think>hmm</think>"), ("text", "recovered")]
    )
    seg = LLMAgent(source="scripted", model="test")
    assert list(seg.transform(iter(["q"]))) == ["recovered"]


def test_unknown_tool_name_fails_at_construction():
    with pytest.raises(ValueError, match="unknown agent tool 'nope'"):
        LLMAgent(source="scripted", model="test", tools=["nope"])


def test_unknown_bare_name_hints_at_the_registering_import():
    """A bare name was meant for the registry: the hint is to import the
    module that registers it, and Strands' "no module named nope" chain
    is dropped as noise."""
    with pytest.raises(UnknownAgentToolError) as info:
        resolve_agent_tools(["nope"])
    assert "is the module that registers it imported" in str(info.value)
    assert "--load-module" in str(info.value)
    assert info.value.__cause__ is None


def test_unknown_tool_path_keeps_the_loader_error():
    """For a path-like string the loader's own error is the useful part,
    and it is in the message itself: a script runner shows one line."""
    with pytest.raises(UnknownAgentToolError) as info:
        resolve_agent_tools(["no_such_module:fn"])
    assert info.value.__cause__ is not None
    assert "is the module that registers it imported" not in str(info.value)
    with pytest.raises(UnknownAgentToolError) as info:
        resolve_agent_tools(["strands.vended_tools:no_such_tool"])
    assert "no_such_tool not found in module strands.vended_tools" in str(info.value)


def test_missing_default_source_raises_talkpipe_error(monkeypatch):
    """The fallback is talkpipe's resolve_source_and_model, so a missing
    source or model raises talkpipe's own message, as llmPrompt does —
    followed by the names of the parameters and variables that fix it."""
    from talkpipe.util import config

    monkeypatch.setattr(config, "_config", config._CaseInsensitiveDict())
    with pytest.raises(ValueError, match="Model name and source must be provided"):
        build_agent_model(None, "m")
    with pytest.raises(
        ValueError, match="Model name and source must be provided"
    ) as info:
        build_agent_model("scripted", None)
    assert "source= and model=" in str(info.value)
    assert "TALKPIPE_DEFAULT_MODEL_SOURCE / TALKPIPE_DEFAULT_MODEL_NAME" in str(
        info.value
    )
    assert isinstance(info.value.__cause__, ValueError)


def test_unknown_tool_is_a_compile_error_too():
    """Script runners format CompileError as a one-line error rather
    than a traceback; Python callers still catch a ValueError."""
    with pytest.raises(compiler.CompileError, match="'nope'") as info:
        resolve_agent_tools(["nope"])
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, UnknownAgentToolError)
    assert info.value.bad_name == "nope"


def test_single_string_is_one_tool_spec():
    # tools="add" (a common slip for tools=["add"]) must not be iterated
    # character by character into "unknown agent tool 'a'"
    assert resolve_agent_tools("add") == [add]
    assert LLMAgent(source="scripted", model="test", tools="add").tools == [add]


def test_plain_function_in_tools_is_wrapped(scripted_model):
    """A plain function handed straight to tools=[...] becomes a Strands
    tool, as it does through register_agent_tool, instead of being
    silently dropped by the agent."""

    def shout_back(text: str) -> str:
        """Return the text in upper case."""
        return text.upper()

    scripted_model.script.extend(
        [("tool", "shout_back", {"text": "hi"}), ("text", "it said HI")]
    )
    chat = AgentChat(source="scripted", model="test", tools=[shout_back])
    assert chat.tool_names == ["shout_back"]
    assert chat.ask("shout") == "it said HI"
    assert "shout_back" not in agent_tool_names()  # passing is not registering


def test_register_agent_tool_as_decorator_with_arguments():
    @register_agent_tool(name="renamed_add")
    def add_again(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    assert add_again.tool_name == "add_again"  # the Strands tool keeps its name
    assert resolve_agent_tools(["renamed_add"]) == [add_again]


def test_strands_tool_name_is_what_the_model_sees(scripted_model):
    """``register_agent_tool(name=...)`` aliases the registry only, so a
    tool the model should see under another name is renamed with
    ``@strands.tool(name=...)`` — and then one name serves both."""

    @register_agent_tool
    @tool(name="tickets")
    def lookup_ticket_impl(ticket_id: str) -> str:
        """Look up a support ticket by id."""
        return f"ticket {ticket_id}"

    assert "tickets" in agent_tool_names()
    scripted_model.call("tickets", ticket_id="T-1").say("Not urgent.")
    step = LLMAgent(source="scripted", model="test", tools=["tickets"])
    assert list(step.transform(iter(["how urgent?"]))) == ["Not urgent."]
    assert scripted_model.tool_results == [
        ToolResult("tickets", "success", "ticket T-1")
    ]


def test_plain_function_without_docstring_warns(caplog):
    def undocumented(x: int) -> int:
        return x * 2

    with caplog.at_level("WARNING", logger="talkpipe_agents.tools"):
        resolve_agent_tools([undocumented])
    assert "'undocumented' has no docstring" in caplog.text


def test_reregistering_a_name_warns_but_replaces(caplog):
    def calculator(expression: str) -> str:
        """An application's own calculator."""
        return "42"

    original = resolve_agent_tools(["calculator"])[0]
    try:
        with caplog.at_level("WARNING", logger="talkpipe_agents.tools"):
            mine = register_agent_tool(calculator)
        assert resolve_agent_tools(["calculator"]) == [mine]
        assert any("calculator" in r.message for r in caplog.records)
        # registering the very same object again is not a replacement
        caplog.clear()
        with caplog.at_level("WARNING", logger="talkpipe_agents.tools"):
            register_agent_tool(mine)
        assert not caplog.records
    finally:
        register_agent_tool(original)


def test_chat_names_the_limit_when_it_ends_the_exchange(scripted_model):
    scripted_model.script.extend([("tool", "add", {"a": 1, "b": 1})] * 3)
    chat = AgentChat(source="scripted", model="test", tools=["add"], max_turns=1)
    answer = chat.ask("loop")
    assert chat.last_result.stop_reason == "limit_turns"
    assert "turns limit (1)" in answer
    assert "no answer text" not in answer


@tool(context=True)
def pause(reason: str, tool_context: ToolContext) -> str:
    """Pause for the human (raises a Strands interrupt)."""
    return str(tool_context.interrupt("pause", reason=reason))


def test_interrupt_is_reported_not_leaked_by_the_segment(scripted_model):
    """A tool that raises a Strands interrupt ends the item: the segment
    yields the text so far (not the interrupt list that str(AgentResult)
    is), says so on the progress channel, and does not retry."""
    scripted_model.call("pause", reason="need a human")
    told: list[str] = []
    seg = LLMAgent(source="scripted", model="test", tools=[pause], progress=told.append)
    assert list(seg.transform(iter(["q"]))) == [""]
    assert seg.last_result.stop_reason == "interrupt"
    assert [i.reason for i in seg.last_result.interrupts] == ["need a human"]
    assert "agent: paused for human input by pause (need a human)" in told
    assert not any("no user-visible text" in line for line in told)
    assert scripted_model.calls == [1]


def test_chat_names_the_interrupt(scripted_model):
    scripted_model.call("pause", reason="need a human")
    chat = AgentChat(source="scripted", model="test", tools=[pause])
    answer = chat.ask("q")
    assert chat.last_result.stop_reason == "interrupt"
    assert "paused for human input by pause: need a human" in answer
    assert "strands:" not in answer
    assert "[{" not in answer


def test_chat_resumes_the_interrupt_with_the_next_message(scripted_model):
    """The message after an interrupt is the human's answer: the paused
    tool receives it and the conversation goes on, instead of Strands
    rejecting a plain string with a TypeError."""
    scripted_model.call("pause", reason="need a human").say("resumed")
    chat = AgentChat(source="scripted", model="test", tools=[pause])
    assert "paused for human input" in chat.ask("q")
    assert chat.ask("here is my answer") == "resumed"
    assert chat.last_result.stop_reason == "end_turn"
    assert scripted_model.tool_results == [("pause", "success", "here is my answer")]
    # and a later message is an ordinary turn again
    scripted_model.say("fine")
    assert chat.ask("thanks") == "fine"


def test_ask_rejects_a_non_string(scripted_model):
    chat = AgentChat(source="scripted", model="test")
    with pytest.raises(
        TypeError, match=r"ask\(\) takes the user's message as a str, got int"
    ):
        chat.ask(123)


def test_strands_tool_paths_resolve_without_registration():
    """Anything that is not a registered name goes to Strands' own
    loader: module:function for one tool, a module for all of them."""
    [sleep] = resolve_agent_tools(["strands.vended_tools:sleep"])
    assert sleep.tool_name == "sleep"
    names = {t.tool_name for t in resolve_agent_tools(["strands.vended_tools"])}
    assert {"sleep", "http_request", "shell", "file_editor"} <= names
    assert "sleep" not in agent_tool_names()


def test_token_budgets_are_strands_limits(scripted_model):
    scripted_model.script.append(("text", "ok"))
    seg = LLMAgent(
        source="scripted",
        model="test",
        max_turns=3,
        max_output_tokens=500,
        max_total_tokens=8000,
    )
    assert seg.limits == {"turns": 3, "output_tokens": 500, "total_tokens": 8000}
    assert AgentChat(source="scripted", model="test", max_turns=None).limits is None
    assert list(seg.transform(iter(["q"]))) == ["ok"]
    assert seg.last_result.stop_reason == "end_turn"


@pytest.mark.parametrize("value", ["four", 0, -1, True, 2.5])
def test_caps_are_validated_under_their_own_names(value):
    with pytest.raises(
        ValueError, match=f"max_turns must be a positive integer or None, got {value!r}"
    ):
        AgentChat(source="scripted", model="test", max_turns=value)
    with pytest.raises(ValueError, match="max_output_tokens must be"):
        LLMAgent(source="scripted", model="test", max_output_tokens=value)


def test_bad_cap_fails_when_the_script_compiles():
    script = 'INPUT FROM "hi" | llmAgent[max_turns="four"]'
    with pytest.raises(
        compiler.CompileError, match="max_turns must be a positive integer"
    ):
        compiler.compile(script)


def test_segment_keeps_last_result_with_stop_reason(scripted_model):
    scripted_model.script.extend([("tool", "add", {"a": 1, "b": 1})] * 5)
    told: list[str] = []
    seg = LLMAgent(
        source="scripted",
        model="test",
        tools=["add"],
        max_turns=2,
        progress=told.append,
    )
    list(seg.transform(iter(["loop"])))
    assert seg.last_result.stop_reason == "limit_turns"
    assert "agent: stopped at the turns limit (2)" in told


def test_unknown_source_lists_registered(scripted_model):
    with pytest.raises(ValueError, match="unknown agent model source"):
        build_agent_model("nosuch", "m")
    assert {"ollama", "scripted"} <= set(agent_model_sources())


def test_eliza_source_says_why_it_is_not_an_agent_source():
    """talkpipe's scripted responder is a common configured default, and
    it cannot call tools; the error has to say so, not just that the
    source is unknown, or the plugin looks broken."""
    with pytest.raises(ValueError) as caught:
        build_agent_model("eliza", "eliza")
    message = str(caught.value)
    assert "cannot call tools" in message
    assert "source=" in message
    assert "scripted" in message


# --- the shipped offline model: talkpipe_agents.testing -------------------


def test_scripted_source_is_scoped_to_the_block():
    """The fixture's source registration must not leak between tests (or
    into an application's real sources)."""
    assert "scripted" not in agent_model_sources()
    with scripted_source() as model:
        assert build_agent_model("scripted", "anything") is model
        assert build_agent_model("scripted", "other-name") is model  # shared
    assert "scripted" not in agent_model_sources()


def test_scripted_source_restores_a_previous_source():
    marker = ScriptedModel()
    register_agent_model("ollama_stub", lambda m, **kw: marker)
    try:
        with scripted_source("ollama_stub") as model:
            assert build_agent_model("ollama_stub", "x") is model
        assert build_agent_model("ollama_stub", "x") is marker
    finally:
        with scripted_source("ollama_stub"):
            pass  # exits restoring `marker`; then drop it entirely below
        from talkpipe_agents import models

        models._factories.pop("ollama_stub", None)


def test_fluent_script_builds_turns_and_runs_the_real_tool(scripted_model):
    """The docs/testing.md example: .call(...) then .say(...)."""
    scripted_model.call("add", a=2, b=3).say("2 + 3 = 5")
    step = LLMAgent(source="scripted", model="any", tools=["add"])
    assert list(step.transform(iter(["What is 2 plus 3?"]))) == ["2 + 3 = 5"]
    assert added == [(2, 3)]
    assert scripted_model.calls == [1, 3]
    assert scripted_model.remaining == ()  # fully consumed
    # prompts records what the agent sent on each turn
    assert scripted_model.prompts[0][0]["role"] == "user"
    assert any("toolResult" in c for c in scripted_model.prompts[1][-1]["content"])
    # tool_results records what the real tool came back with
    assert scripted_model.tool_results == [("add", "success", "5.0")]
    assert scripted_model.tool_results[0] == ToolResult("add", "success", "5.0")


def test_scripted_call_to_a_tool_the_agent_lacks_fails_the_turn(scripted_model):
    """A misspelled or ungranted tool name must not pass silently: Strands
    would feed "Unknown tool" back and the scripted answer would still run."""
    from strands.types.exceptions import EventLoopException

    # Strands lets a first-turn model error through as-is and wraps later
    # ones in EventLoopException; the message survives either way.
    scripted_model.call("ad", a=1, b=1).say("2")
    chat = AgentChat(source="scripted", model="any", tools=["add"])
    with pytest.raises(ValueError, match=r"'ad'.*has only: add") as info:
        chat.ask("1+1?")
    assert "misspelled" in str(info.value)
    assert added == []  # nothing ran

    scripted_model.reset()
    scripted_model.call("add", a=1, b=1).call("ad", a=1, b=1).say("2")
    with pytest.raises(EventLoopException, match=r"'ad'.*has only: add"):
        AgentChat(source="scripted", model="any", tools=["add"]).ask("1+1?")

    scripted_model.reset()
    scripted_model.call("add", a=1, b=1).say("2")
    with pytest.raises(ValueError, match="has no tools"):
        AgentChat(source="scripted", model="any").ask("1+1?")


def test_errored_scripted_call_is_recorded_and_warned(scripted_model):
    """Wrong argument names make the tool return an error to the model; the
    scripted answer still plays, so the mistake surfaces as a warning and
    in tool_results rather than as a green test."""
    scripted_model.call("add", x=1, y=1).say("2")
    chat = AgentChat(source="scripted", model="any", tools=["add"])
    with pytest.warns(ScriptedToolErrorWarning, match="'add' came back as an error"):
        assert chat.ask("1+1?") == "2"
    assert added == []
    (result,) = scripted_model.tool_results
    assert (result.name, result.status) == ("add", "error")
    assert "Validation failed" in result.text  # the tool never ran


def test_remaining_lists_unplayed_turns(scripted_model):
    scripted_model.say("one").say("two")
    assert AgentChat(source="scripted", model="any").ask("hi") == "one"
    assert scripted_model.remaining == (("text", "two"),)
    scripted_model.reset()
    assert scripted_model.remaining == ()
    assert scripted_model.tool_results == []


def test_scripted_model_constructor_takes_turns():
    model = ScriptedModel(("tool", "add", {"a": 1, "b": 1}), ("text", "2"))
    with scripted_source(model=model):
        chat = AgentChat(source="scripted", model="any", tools=["add"])
        assert chat.ask("1+1?") == "2"


def test_running_out_of_script_is_a_clear_error(scripted_model):
    from strands.types.exceptions import EventLoopException

    scripted_model.call("add", a=1, b=1)  # no answer scripted after the call
    chat = AgentChat(source="scripted", model="any", tools=["add"])
    # Strands wraps model errors, but the message survives — and says where
    # the agent was when the script ran out
    with pytest.raises(
        EventLoopException, match="no turn left for model call 2 after 'add' returned"
    ):
        chat.ask("1+1?")


def test_reset_clears_script_and_records(scripted_model):
    scripted_model.say("hello")
    AgentChat(source="scripted", model="any").ask("hi")
    assert scripted_model.calls == [1]
    scripted_model.say("leftover")
    scripted_model.reset()
    assert (scripted_model.script, scripted_model.calls, scripted_model.prompts) == (
        deque(),
        [],
        [],
    )


def test_missing_provider_extra_names_the_install(monkeypatch):
    from talkpipe_agents import models

    def missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(models, "import_module", missing)
    # this package's extra, not the underlying strands-agents one: installing
    # that would leave the application's own dependency metadata unaware of it
    with pytest.raises(ImportError, match=r"talkpipe-agents\[openai\]"):
        build_agent_model("openai", "gpt-4o")


def test_ollama_factory_uses_talkpipe_config(monkeypatch):
    from talkpipe.util.config import get_config

    monkeypatch.setenv("TALKPIPE_OLLAMA_SERVER_URL", "http://example:11434")
    get_config(reload=True)
    model = build_agent_model("ollama", "qwen3:8b", temperature=0.2)
    assert model.host == "http://example:11434"
    cfg = model.get_config()
    assert cfg["model_id"] == "qwen3:8b"
    assert cfg["temperature"] == 0.2
    get_config(reload=True)


def test_ollama_factory_takes_model_params_as_options():
    model = build_agent_model(
        "ollama", "qwen3:8b", options={"top_k": 5}, params={"num_ctx": 16384}
    )
    assert model.get_config()["options"] == {"top_k": 5, "num_ctx": 16384}


def test_openai_source_builds_the_responses_model():
    responses = pytest.importorskip("strands.models.openai_responses")
    model = build_agent_model(
        "openai",
        "gpt-test",
        temperature=0.3,
        params='{"reasoning": {"effort": "low"}}',
    )
    assert isinstance(model, responses.OpenAIResponsesModel)
    cfg = model.get_config()
    assert cfg["model_id"] == "gpt-test"
    assert cfg["params"] == {"reasoning": {"effort": "low"}, "temperature": 0.3}


def test_openai_is_the_openai_responses_source():
    pytest.importorskip("strands.models.openai_responses")
    pair = [build_agent_model(s, "m") for s in ("openai", "openai_responses")]
    assert type(pair[0]) is type(pair[1])


def test_openai_source_formats_a_request_without_params():
    pytest.importorskip("strands.models.openai_responses")
    model = build_agent_model("openai", "gpt-test")
    request = model._format_request([{"role": "user", "content": [{"text": "hi"}]}])
    assert request["model"] == "gpt-test"


def test_openai_chat_source_builds_the_chat_completions_model():
    chat = pytest.importorskip("strands.models.openai")
    model = build_agent_model(
        "openai_chat", "local-model", params={"reasoning_effort": "none"}
    )
    assert isinstance(model, chat.OpenAIModel)
    assert model.get_config()["params"] == {"reasoning_effort": "none"}


@pytest.fixture
def params_seen():
    """A ``capture`` source recording the kwargs its factory is given."""
    from talkpipe_agents import models

    seen: list[dict] = []
    scripted = ScriptedModel().say("ok")

    def factory(model, **kwargs):
        seen.append(kwargs)
        return scripted

    register_agent_model("capture", factory)
    try:
        yield seen
    finally:
        models._factories.pop("capture", None)


def test_llm_agent_passes_model_params_json_to_the_factory(params_seen):
    script = (
        'INPUT FROM echo[data="hi"] | llmAgent[source="capture", model="m", '
        """model_params='{"reasoning_effort": "none"}', temperature=0.5]"""
    )
    assert list(compiler.compile(script)()) == ["ok"]
    assert params_seen == [{"temperature": 0.5, "params": {"reasoning_effort": "none"}}]


def test_no_model_params_means_no_params_keyword(params_seen):
    AgentChat(source="capture", model="m", model_params={"a": 1})
    AgentChat(source="capture", model="m")
    assert params_seen == [
        {"temperature": None, "params": {"a": 1}},
        {"temperature": None},
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [("{reasoning: none}", "not valid JSON"), ('["a"]', "must be a JSON object")],
)
def test_bad_model_params_fail_when_the_script_compiles(value, message):
    script = f"INPUT FROM echo[data=\"hi\"] | llmAgent[model_params='{value}']"
    with pytest.raises(compiler.CompileError, match=message):
        compiler.compile(script)


def test_ollama_connection_error_names_the_server_and_the_setting(monkeypatch):
    import asyncio

    import httpx
    import ollama
    from talkpipe.util.config import get_config

    async def refuse(self, *args, **kwargs):
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(ollama.AsyncClient, "chat", refuse)
    monkeypatch.delenv("TALKPIPE_OLLAMA_SERVER_URL", raising=False)
    get_config(reload=True)
    try:
        model = build_agent_model("ollama", "llama3.2")
    finally:
        get_config(reload=True)

    async def drain():
        async for _ in model.stream([]):
            pass

    with pytest.raises(httpx.ConnectError) as info:
        asyncio.run(drain())
    message = str(info.value)
    assert message.startswith("All connection attempts failed")
    assert "http://localhost:11434" in message  # where the client looked
    assert "TALKPIPE_OLLAMA_SERVER_URL" in message


def test_ollama_missing_model_says_which_server_and_how_to_pull(monkeypatch):
    import asyncio

    import ollama
    from talkpipe.util.config import get_config

    async def not_found(self, *args, **kwargs):
        raise ollama.ResponseError("model 'llama3.9' not found", 404)

    monkeypatch.setattr(ollama.AsyncClient, "chat", not_found)
    monkeypatch.setenv("TALKPIPE_OLLAMA_SERVER_URL", "http://gpu-box:11434")
    get_config(reload=True)
    try:
        model = build_agent_model("ollama", "llama3.9")
    finally:
        get_config(reload=True)

    async def drain():
        async for _ in model.stream([]):
            pass

    with pytest.raises(ollama.ResponseError) as info:
        asyncio.run(drain())
    message = str(info.value)
    assert "model 'llama3.9' not found" in message
    assert "http://gpu-box:11434" in message
    assert "ollama pull llama3.9" in message
    assert info.value.status_code == 404


def test_ollama_model_without_tool_support_says_to_pick_another(monkeypatch):
    import asyncio

    import ollama
    from talkpipe.util.config import get_config

    async def no_tools(self, *args, **kwargs):
        raise ollama.ResponseError(
            "registry.ollama.ai/library/llama2:latest does not support tools", 400
        )

    monkeypatch.setattr(ollama.AsyncClient, "chat", no_tools)
    monkeypatch.setenv("TALKPIPE_OLLAMA_SERVER_URL", "http://gpu-box:11434")
    get_config(reload=True)
    try:
        model = build_agent_model("ollama", "llama2")
    finally:
        get_config(reload=True)

    async def drain():
        async for _ in model.stream([]):
            pass

    with pytest.raises(ollama.ResponseError) as info:
        asyncio.run(drain())
    message = str(info.value)
    assert "does not support tools" in message
    assert "needs a model that can call tools" in message
    assert "http://gpu-box:11434" in message
    assert "ollama pull" in message


@pytest.mark.parametrize("source", ["openai_chat", "openai", "openai_responses"])
def test_openai_connection_error_names_the_endpoint_and_the_setting(
    monkeypatch, source
):
    """The OpenAI client's ``Connection error.`` says nothing about where
    it looked; the hint names the endpoint and OPENAI_BASE_URL."""
    import asyncio

    import httpx

    openai = pytest.importorskip("openai")
    from strands.models import openai as strands_openai
    from strands.models import openai_responses as strands_responses

    async def refuse(self, *args, **kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/"))
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(strands_openai.OpenAIModel, "stream", refuse)
    monkeypatch.setattr(strands_responses.OpenAIResponsesModel, "stream", refuse)
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gpu-box:8000/v1")
    model = build_agent_model(source, "m")

    async def drain():
        async for _ in model.stream([]):
            pass

    with pytest.raises(openai.APIConnectionError) as info:
        asyncio.run(drain())
    message = str(info.value)
    assert message.startswith("Connection error.")
    assert "http://gpu-box:8000/v1" in message
    assert "OPENAI_BASE_URL" in message
    # an explicit client base_url wins over the environment
    explicit = build_agent_model(
        source, "m", client_args={"base_url": "http://box2:9/v1", "api_key": "k"}
    )

    async def drain2():
        async for _ in explicit.stream([]):
            pass

    with pytest.raises(openai.APIConnectionError) as info2:
        asyncio.run(drain2())
    assert "http://box2:9/v1" in str(info2.value)


def test_ask_user_uses_injected_callable():
    asked = []

    def ask(question):
        asked.append(question)
        return "blue"

    assert make_ask_user(ask)("Favorite color?") == "blue"
    assert asked == ["Favorite color?"]


def test_think_tool_registered_and_inert(scripted_model):
    """The generic think tool: available by name, executes as a no-op
    that hands the turn back to the model."""
    scripted_model.script.extend(
        [("tool", "think", {"thought": "step one"}), ("text", "done thinking")]
    )
    chat = AgentChat(source="scripted", model="test", tools=["think"])
    assert chat.ask("work it out") == "done thinking"
    results = [
        c["toolResult"]
        for m in chat.messages
        for c in m["content"]
        if "toolResult" in c
    ]
    assert results
    assert results[0]["status"] == "success"
    assert "continue" in str(results[0]["content"])


def test_ask_user_headless_default_degrades():
    # under pytest stdin is not a tty
    assert "not available" in make_ask_user()("Favorite color?")


def test_resolve_passes_tool_objects_through():
    assert resolve_agent_tools([add, "add"]) == [add, add]


def test_non_tool_object_is_rejected_up_front():
    """Strands drops an unrecognized tool specification at run time with a
    log line; the agent would silently have fewer tools than written."""
    with pytest.raises(TypeError, match=r"not a tool: 42 \(int\)"):
        resolve_agent_tools([42])
    with pytest.raises(TypeError, match="not a tool: None"):
        LLMAgent(source="scripted", model="test", tools=[None])


# --- chatterlang scripts as tools (script_tools) ---


def test_script_tool_runs_script_on_input():
    shout = make_script_tool(
        "shout", "Uppercase the given text.", '| lambda[expression="item.upper()"]'
    )
    assert shout.tool_name == "shout"
    assert shout.tool_spec["description"] == "Uppercase the given text."
    assert shout("hello there") == "HELLO THERE"


def test_script_tool_documents_its_argument():
    spec = make_script_tool(
        "shout2",
        "Uppercase.",
        '| lambda[expression="item.upper()"]',
        input_description="The text to shout.",
    ).tool_spec
    props = spec["inputSchema"]["json"]["properties"]
    assert props["text"]["description"] == "The text to shout."


def test_script_tool_with_own_source_ignores_input():
    letters = make_script_tool(
        "letters", "List the letters.", 'INPUT FROM echo[data="a,b,c"]'
    )
    assert letters("whatever") == "a\nb\nc"


def test_script_tool_state_is_fresh_per_call():
    # accum[reset=false] would accumulate across calls if the compiled
    # script were shared; each call compiles its own instance
    acc = make_script_tool("acc", "Accumulate.", "| accum[reset=false]")
    assert acc("a") == "a"
    assert acc("b") == "b"


def test_bad_tool_script_fails_at_definition():
    with pytest.raises(compiler.CompileError, match="'wordCount'"):
        make_script_tool("wordCount", "Count words.", "| noSuchSegment")


def test_script_tool_statement_defines_tool_for_later_llm_agent(scripted_model):
    """The whole shape in one script: a documented tooling script
    defined inline, then used by name in llmAgent."""
    scripted_model.script.extend(
        [("tool", "shout3", {"text": "hi"}), ("text", "the tool said HI")]
    )
    script = """
    scriptTool[name="shout3",
               description="Uppercase the given text.",
               script="| lambda[expression='item.upper()']"];
    INPUT FROM echo[data="please shout hi"]
    | llmAgent[source="scripted", model="test", tools=["shout3"]]
    """
    assert list(compiler.compile(script)()) == ["the tool said HI"]
    # the script's output made it back to the model as the tool result
    results = [
        c["toolResult"]
        for m in scripted_model.prompts[-1]
        for c in m["content"]
        if "toolResult" in c
    ]
    assert results
    assert "HI" in str(results[0]["content"])
    assert "shout3" in agent_tool_names()


def test_bad_script_tool_fails_when_outer_script_compiles():
    with pytest.raises(compiler.CompileError, match="'broken'"):
        compiler.compile(
            'scriptTool[name="broken", description="d", script="| noSuchSegment"]'
        )


def test_script_tool_segment_passes_items_through():
    seg = ScriptTool(
        name="pass_demo", description="d", script='| lambda[expression="item"]'
    )
    assert list(seg(iter(["a", "b"]))) == ["a", "b"]


@pytest.fixture
def skills_dir(tmp_path):
    """A skills folder holding one skill, ``greeting``."""
    skill = tmp_path / "skills" / "greeting"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: greeting\ndescription: How to greet people.\n---\n"
        'Always greet with "Ahoy".\n'
    )
    return tmp_path / "skills"


def test_skills_folder_is_listed_and_loaded_on_demand(scripted_model, skills_dir):
    scripted_model.call("skills", skill_name="greeting").say("Ahoy!")
    chat = AgentChat(source="scripted", model="test", skills=str(skills_dir))
    assert chat.ask("Say hello") == "Ahoy!"
    # the plugin's tool, and the skill's instructions came back through it
    assert chat.tool_names == ["skills"]
    assert scripted_model.tool_results[0].name == "skills"
    assert scripted_model.tool_results[0].status == "success"
    assert 'Always greet with "Ahoy".' in scripted_model.tool_results[0].text
    # and the skill was advertised in the system prompt
    assert "<name>greeting</name>" in chat.agent.system_prompt


def test_skills_from_chatterlang_across_fresh_agents(
    scripted_model, skills_dir, caplog
):
    scripted_model.call("skills", skill_name="greeting").say("Ahoy!")
    scripted_model.call("skills", skill_name="greeting").say("Ahoy again!")
    script = (
        'INPUT FROM echo[data="hi,hello"] '
        f'| llmAgent[source="scripted", model="test", skills="{skills_dir}"]'
    )
    with caplog.at_level("WARNING", logger="strands.vended_plugins.skills"):
        out = list(compiler.compile(script)())
    assert out == ["Ahoy!", "Ahoy again!"]
    assert [r.status for r in scripted_model.tool_results] == ["success"] * 2
    # the second (fresh) agent started from the plain prompt; the plugin
    # must not complain about its previous injection being missing
    assert caplog.records == []


def test_missing_skills_path_is_a_compile_error(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError, match="does not exist"):
        LLMAgent(source="scripted", model="test", skills=str(missing))
    with pytest.raises(compiler.CompileError, match="does not exist"):
        compiler.compile(
            f'INPUT FROM "x" | llmAgent[source="scripted", skills="{missing}"]'
        )
