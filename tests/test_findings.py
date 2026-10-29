"""The findings loop, fully offline.

The agent's model is the ``scripted_model`` fixture; the judge is either
``ScriptedJudge`` (decides by label) or a ``FindingsJudge`` with a fake
``generate`` (to exercise the reply parsing), or — through
``judge=True`` — the scripted model itself, answering the judge's
prompt with a scripted JSON reply between the agent's turns.
"""

from __future__ import annotations

import pytest
from strands.types.tools import ToolContext
from talkpipe.chatterlang import compiler
from talkpipe.search.abstract import SearchResult

from talkpipe_agents import (
    Finding,
    FindingsJudge,
    FindingsSession,
    JudgeFindings,
    LLMAgent,
    as_findings,
    finding_tool,
    make_finding_tool,
    parse_judgements,
    register_agent_tool,
)
from talkpipe_agents.findings import render_findings
from talkpipe_agents.testing import ScriptedJudge

QUESTION = "What is the capital of France?"

searched: list[tuple[str, str]] = []

NOTES = [
    Finding("Paris is the capital of France.", label="geography", source="geo.md"),
    Finding("Bananas are yellow.", label="fruit", source="fruit.md"),
    Finding("France's capital has been Paris since 987.", label="history"),
]


@finding_tool
def search_notes(query: str, looking_for: str = "") -> list[Finding]:
    """Search the notes for the query.

    Args:
        query: What to search for.
        looking_for: What the answer still lacks.
    """
    searched.append((query, looking_for))
    return list(NOTES)


@finding_tool(name="lookup_fact")
def a_plain_lookup(topic: str) -> str:
    """Look up one fact as a string.

    Args:
        topic: The topic.
    """
    return f"A fact about {topic}."


@register_agent_tool
def count_words(text: str) -> int:
    """Count the words in text.

    Args:
        text: The text.
    """
    return len(text.split())


@pytest.fixture(autouse=True)
def _clear():
    searched.clear()


def _step(**kwargs):
    kwargs.setdefault("judge", ScriptedJudge(keep={"geography", "history"}))
    kwargs.setdefault("tools", ["search_notes"])
    return LLMAgent(source="scripted", model="any", **kwargs)


# -- findings ----------------------------------------------------------------


def test_finding_key_defaults_to_a_hash_of_its_text():
    a, b = Finding("same text", label="a"), Finding("same text", label="b")
    assert a.key == b.key != Finding("other text").key
    assert Finding("x", key="k1").key == "k1"


def test_a_search_result_becomes_a_finding_and_back():
    hit = SearchResult(
        score=0.5,
        doc_id="d1",
        document={"content": "body", "title": "T", "source": "/a.md"},
    )
    finding = Finding.from_search_result(hit)
    assert (finding.text, finding.label, finding.source, finding.key) == (
        "body",
        "T",
        "/a.md",
        "d1",
    )
    assert finding.to_search_result() is finding.meta["result"]
    built = Finding("text", label="L", source="s").to_search_result()
    assert built.doc_id == "s"
    assert built.document == {"content": "text", "title": "L", "source": "s"}


def test_kept_findings_feed_talkpipes_rag_sources():
    """The docs/findings.md RAG example: AppendRAGSources reads ``_background``."""
    from talkpipe.pipelines.basic_rag import AppendRAGSources

    kept = [Finding("330 m", label="eiffel.md", source="eiffel.md"), Finding("1.95")]
    item = {
        "partial": "About 1.95 times.",
        "_background": [f.to_search_result() for f in kept],
    }
    [out] = AppendRAGSources(partial_answer_field="partial", set_as="answer")([item])
    assert out["answer"] == "About 1.95 times.\n\nSources:\n- eiffel.md"


def test_dict_hits_are_normalized_and_renamed():
    hit = {"doc_id": "", "score": 1, "document": {"shingle": "x", "path": "/p"}}
    finding = Finding.from_search_result(hit, rename_fields={"path": "source"})
    assert finding.text == "x"
    assert finding.source == "/p"
    assert finding.meta["result"].document == {"shingle": "x", "source": "/p"}
    assert finding.key.startswith("{")  # no doc_id: the document is the identity


def test_tool_results_of_every_shape_become_findings():
    found = as_findings(
        [
            "a string",
            {"text": "a dict", "label": "D", "source": "src", "extra": 1},
            {"doc_id": "d", "score": 0, "document": {"content": "a hit"}},
            SearchResult(score=0, doc_id="e", document={"content": "a result"}),
            {"n": 3},
            "",
            None,
        ],
        label="tool",
    )
    assert [f.text for f in found] == [
        "a string",
        "a dict",
        "a hit",
        "a result",
        '{"n": 3}',
    ]
    assert found[0].label == "tool"
    assert (found[1].label, found[1].source, found[1].meta) == (
        "D",
        "src",
        {"extra": 1},
    )
    assert as_findings("one") == [Finding("one")]
    assert as_findings(None) == []
    assert as_findings(42)[0].text == "42"


# -- the session -------------------------------------------------------------


def _session(**kwargs):
    lines: list[str] = []
    return FindingsSession(QUESTION, tell=lines.append, **kwargs), lines


def test_register_hands_out_ids_and_dedups_by_key():
    session, _ = _session()
    assert session.register(NOTES[0]) == "f1"
    assert session.register(NOTES[1]) == "f2"
    assert session.register(Finding(NOTES[0].text)) == "f1"


def test_decisions_keep_order_notes_and_trace():
    session, lines = _session(max_kept=3)
    for finding in NOTES:
        session.register(finding)
    session.apply("f3", True, "the date.")
    session.apply("f2", False, "off topic")
    session.apply("f1", True, "the answer")
    assert [f.label for f in session.kept] == ["history", "geography"]
    assert [f.label for f in session.ignored] == ["fruit"]
    assert session.notes == {"f3": "the date", "f1": "the answer"}
    assert session.known_notes() == ["history — the date", "geography — the answer"]
    assert session.seen_keys() == {f.key for f in NOTES}
    assert lines[0].startswith("agent · keep f3 (history) — the date. Kept 1")
    assert session.at_capacity is False


def test_enough_is_refused_while_nothing_is_kept():
    session, lines = _session()
    reply = session.declare_enough("done")
    assert "Nothing has been kept yet" in reply
    assert not session.enough
    assert lines == ["agent ✗ enough_information refused: nothing kept yet"]
    session.apply(session.register(NOTES[0]), True, "it")
    assert "Noted: 1 finding(s) kept" in session.declare_enough("covered")
    assert session.enough_reason == "covered"
    assert session.should_stop
    assert session.guard("search_notes").startswith(
        "The search is over — you already said you had enough."
    )


def test_capacity_is_a_stop():
    session, _ = _session(max_kept=1)
    assert session.guard("x") == ""
    session.apply(session.register(NOTES[0]), True)
    assert session.stop_note == "the 1-finding maximum is reached"


def test_the_countdown():
    session, _ = _session(max_turns=4)
    assert session.turns_note() == "4 of 4 turns left."
    session.note_turn(2)
    session.note_turn(1)  # never goes backwards
    assert session.turns_left == 2
    assert session.with_turns("Reply.") == "Reply. 2 of 4 turns left."
    assert session.with_turns("Reply. 2 of 4 turns left.").count("turns left") == 1
    session.note_turn(3)
    assert "nothing is kept yet" in session.turns_note()
    session.apply(session.register(NOTES[0]), True)
    assert "call enough_information now" in session.turns_note()
    session.declare_enough("covered")
    assert session.turns_note() == "1 of 4 turns left."  # nothing more to call
    session.note_turn(9)
    assert session.turns_note() == "No turns left."
    assert FindingsSession().turns_note() == ""


@pytest.mark.parametrize(
    ("setup", "stop_reason", "error", "expected"),
    [
        ("enough", "", "", "Agent judged it had enough to answer (have it): kept 1"),
        ("full", "", "", "Agent filled its 1-finding maximum"),
        ("none", "limit_turns", "", "Agent hit the 3-turn cap"),
        ("ignored", "end_turn", "", "Agent found nothing relevant to keep"),
        ("none", "end_turn", "", "Agent stopped without saying it had enough"),
        ("none", "", "boom", "Agent stopped with an error (boom)"),
    ],
)
def test_the_outcome_says_how_the_run_ended(setup, stop_reason, error, expected):
    session, _ = _session(max_kept=1, max_turns=3)
    fid = session.register(NOTES[0])
    if setup in ("enough", "full"):
        session.apply(fid, True)
    if setup == "enough":
        session.max_kept = 5
        session.declare_enough("have it")
    if setup == "ignored":
        session.calls = 1
        session.apply(fid, False)
    outcome = session.outcome(stop_reason, error=error)
    assert outcome.note.startswith(expected)
    assert outcome.error == error
    assert outcome.trace[-1] == outcome.note


# -- the judge ---------------------------------------------------------------

JSON_REPLY = (
    '{"decisions": [{"id": "f1", "keep": true, "reason": "yes"}, '
    '{"id": "f2", "keep": false, "reason": "no"}]}'
)


@pytest.mark.parametrize(
    "reply",
    [
        JSON_REPLY,
        f"Here you go:\n```json\n{JSON_REPLY}\n```",
        JSON_REPLY[len('{"decisions": ') : -1],  # the bare list
        "f1: true – yes\nf2: false – no",  # noqa: RUF001 - en dashes, as gpt-oss writes
        "[F1] KEEP — yes.\n[r2] ignore: no",
        "1 keep yes\n2 ignore no\nThat is all.",
    ],
)
def test_replies_in_json_or_prose_are_both_read(reply):
    judge = FindingsJudge(generate=lambda _prompt: reply)
    judged = judge.judge(NOTES[:2], QUESTION)
    assert [(j.finding.label, j.keep, j.reason) for j in judged] == [
        ("geography", True, "yes"),
        ("fruit", False, "no"),
    ]


def test_an_unreadable_reply_raises_in_the_parser():
    with pytest.raises(ValueError, match=r"held no decisions: 'I cannot decide\.'"):
        parse_judgements("I cannot decide.")
    with pytest.raises(ValueError, match="held no decisions: it was empty"):
        parse_judgements("  \n")


def test_a_judge_whose_model_cannot_be_built_raises():
    # A configuration error is not a failed judgement: keeping every
    # finding would hide it.
    judge = FindingsJudge(source="no_such_source", model="any")
    with pytest.raises(ValueError, match="unknown agent model source"):
        judge.judge(NOTES[:2], QUESTION)
    with pytest.raises(ValueError, match="judgeFindings: unknown agent model"):
        list(judge([{"question": QUESTION, "findings": ["Paris."]}]))


@pytest.mark.parametrize(
    ("generate", "reason"),
    [
        (lambda _p: "no decisions here", "not judged: the judge's reply held no"),
        (lambda _p: (_ for _ in ()).throw(RuntimeError("down")), "not judged: down"),
        (
            lambda _p: '{"decisions": [{"id": "f9", "keep": false}]}',
            "not judged: the model gave no",
        ),
    ],
)
def test_a_failed_or_partial_judgement_keeps_everything(generate, reason):
    judged = FindingsJudge(generate=generate).judge(NOTES[:2], QUESTION)
    assert [j.keep for j in judged] == [True, True]
    assert all(j.reason.startswith(reason) for j in judged)


def test_seen_findings_are_skipped_and_nothing_left_makes_no_call():
    prompts: list[str] = []

    def generate(prompt):
        prompts.append(prompt)
        return "f1 keep new"

    judge = FindingsJudge(generate=generate)
    judged = judge.judge(NOTES[:2], QUESTION, seen=[NOTES[0].key])
    assert [j.finding.label for j in judged] == ["fruit"]
    assert judge.judge(NOTES[:1], QUESTION, seen=[NOTES[0].key]) == []
    assert len(prompts) == 1


def test_the_prompt_carries_question_looking_for_known_and_capped_text():
    judge = FindingsJudge(max_chars=10)
    labelled = [("f1", Finding("x" * 50, label="long"))]
    prompt = judge.prompt(QUESTION, "the date", ["geography — the answer"], labelled)
    assert f"Question: {QUESTION}" in prompt
    assert "This call is looking for: the date" in prompt
    assert "- geography — the answer" in prompt
    assert "[f1] long\n" + "x" * 10 + "…" in prompt
    bare = judge.prompt(QUESTION, "", [], labelled)
    assert "looking for" not in bare
    assert "- nothing yet" in bare


def test_judge_findings_segment_sets_judgements():
    judge = FindingsJudge(generate=lambda _p: JSON_REPLY)
    item = {"question": QUESTION, "findings": ["Paris.", "Bananas."]}
    [out] = judge([item])
    assert [(j["finding"].text, j["keep"]) for j in out["judged"]] == [
        ("Paris.", True),
        ("Bananas.", False),
    ]
    dropper = FindingsJudge(
        generate=lambda _p: JSON_REPLY, drop_ignored=True, set_as="kept"
    )
    [out] = dropper([{"question": QUESTION, "findings": ["Paris.", "Bananas."]}])
    assert [f.text for f in out["kept"]] == ["Paris."]


def test_judge_findings_segment_from_chatterlang(scripted_model):
    scripted_model.say(JSON_REPLY)
    script = (
        "| judgeFindings[source='scripted', model='any', findings_field='hits', "
        "drop_ignored=True]"
    )
    pipeline = compiler.compile(script).as_function(single_in=True, single_out=True)
    out = pipeline({"question": QUESTION, "hits": ["Paris.", "Bananas."]})
    assert [f.text for f in out["judged"]] == ["Paris."]


# -- the loop inside llmAgent ------------------------------------------------


def test_judged_loop_keeps_relevant_findings_and_answers(scripted_model):
    scripted_model.call("search_notes", query="capital", looking_for="")
    scripted_model.call("enough_information", reason="the capital is known")
    scripted_model.say("Paris.")
    lines: list[str] = []
    step = _step(max_turns=5, progress=lines.append)
    assert list(step([QUESTION])) == ["Paris."]
    search, enough = scripted_model.tool_results
    assert search.status == "success"
    assert "3 new finding(s) from search_notes: kept 2, ignored 1." in search.text
    # the source is shown, so the answer can cite it rather than the id
    assert (
        "[f1] geography (source: geo.md)\nParis is the capital of France."
        in search.text
    )
    assert "[f2] fruit — not relevant" in search.text
    assert "4 of 5 turns left." in search.text
    assert "Noted: 2 finding(s) kept." in enough.text
    outcome = step.last_outcome
    assert [f.label for f in outcome.findings] == ["geography", "history"]
    assert outcome.note.startswith("Agent judged it had enough to answer")
    assert "agent · ignore f2 (fruit) — not relevant." in "\n".join(outcome.trace)
    assert outcome.trace == lines  # progress lines are the trace


def test_the_protocol_and_session_tools_are_added(scripted_model):
    step = _step(system_prompt="You answer geography questions.")
    assert step.system_prompt.startswith("You answer geography questions.\n\n")
    assert "enough_information" in step.system_prompt
    names = [t.tool_name for t in step.tools]
    assert names == ["search_notes", "enough_information"]
    unjudged = _step(
        judge=False, findings=True, tools=["search_notes", "enough_information"]
    )
    assert [t.tool_name for t in unjudged.tools] == [
        "search_notes",
        "enough_information",
        "keep_finding",
        "ignore_finding",
    ]
    assert "keep_finding" in unjudged.system_prompt
    assert LLMAgent(source="scripted", model="any").system_prompt == (
        "You are a helpful assistant."
    )


def test_looking_for_and_known_reach_the_judge(scripted_model):
    judge = ScriptedJudge(keep={"geography"})
    scripted_model.call("search_notes", query="capital")
    scripted_model.call("search_notes", query="since when", looking_for="the date")
    scripted_model.say("Paris.")
    list(_step(judge=judge)([QUESTION]))
    # The second call skips what the first decided on.
    assert judge.batches == [["geography", "fruit", "history"]]
    assert scripted_model.tool_results[1].text.startswith(
        "No new findings from search_notes."
    )
    assert searched == [("capital", ""), ("since when", "the date")]


def test_second_call_is_judged_against_what_is_known(scripted_model):
    first, second = NOTES[:2], [NOTES[2]]
    results = iter([first, second])

    @finding_tool(register=False)
    def paged(page: int) -> list[Finding]:
        """Return a page of notes.

        Args:
            page: The page.
        """
        return next(results)

    judge = ScriptedJudge(keep={"geography", "history"})
    scripted_model.call("paged", page=1).call("paged", page=2).say("Paris.")
    list(_step(tools=[paged], judge=judge)([QUESTION]))
    assert judge.batches == [["geography", "fruit"], ["history"]]
    assert "Already known:\n- geography — relevant" in judge.prompts[1]
    assert "looking for" not in judge.prompts[1]


def test_capacity_passes_over_relevant_findings_and_refuses_more(scripted_model):
    scripted_model.call("search_notes", query="capital")
    scripted_model.call("search_notes", query="again")
    scripted_model.say("Paris.")
    step = _step(max_kept=1)
    list(step([QUESTION]))
    first, second = scripted_model.tool_results
    assert (
        "1 more relevant finding(s) were not kept: the 1-finding maximum" in first.text
    )
    assert "You have kept the maximum of 1 findings." in first.text
    assert second.text.startswith(
        "The search is over — the 1-finding maximum is reached."
    )
    assert searched == [("capital", "")]  # the refused call never ran
    assert step.last_outcome.note.startswith("Agent filled its 1-finding maximum")
    assert step.last_outcome.undecided == 0  # passed over is not undecided


def test_plain_tools_get_the_countdown_too(scripted_model):
    scripted_model.call("count_words", text="a b c").say("3")
    list(_step(tools=["count_words"], max_turns=4)(["How many words?"]))
    assert scripted_model.tool_results[0].text == "3\n3 of 4 turns left."


def test_the_last_turn_is_announced(scripted_model):
    scripted_model.call("search_notes", query="capital").say("Paris.")
    list(_step(max_turns=2)([QUESTION]))
    assert (
        "1 turn left: the next one is your last, so call enough_information now"
        in scripted_model.tool_results[0].text
    )


def test_a_failing_judge_keeps_everything_and_the_run_goes_on(scripted_model):
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(judge=ScriptedJudge(error=RuntimeError("judge down")))
    assert list(step([QUESTION])) == ["Paris."]
    assert "kept 3, ignored 0" in scripted_model.tool_results[0].text
    assert "not judged: judge down" in "\n".join(step.last_outcome.trace)


def test_judge_true_judges_with_the_agents_own_model(scripted_model):
    scripted_model.call("search_notes", query="capital")
    # The judge's turn, between the tool call and the agent's next turn.
    scripted_model.say(
        '{"decisions": [{"id": "f1", "keep": true, "reason": "the answer"},'
        ' {"id": "f2", "keep": false, "reason": "fruit"},'
        ' {"id": "f3", "keep": false, "reason": "repeats f1"}]}'
    )
    scripted_model.say("Paris.")
    step = _step(judge=True)
    assert list(step([QUESTION])) == ["Paris."]
    assert [f.label for f in step.last_outcome.findings] == ["geography"]
    judge_prompt = scripted_model.prompts[1][0]["content"][0]["text"]
    assert judge_prompt.startswith(f"Question: {QUESTION}")
    assert "[f3] history" in judge_prompt


def test_findings_without_a_judge_are_decided_by_hand(scripted_model):
    scripted_model.call("search_notes", query="capital")
    scripted_model.call("keep_finding", finding_id="f1", reason="the answer")
    scripted_model.call("ignore_finding", finding_id="f2", reason="fruit")
    scripted_model.call("keep_finding", finding_id="f9")
    scripted_model.call("enough_information")
    scripted_model.say("Paris.")
    step = _step(judge=False, findings=True)
    list(step([QUESTION]))
    shown, kept, ignored, unknown, enough = scripted_model.tool_results
    assert "3 finding(s) from search_notes. Decide on each undecided one" in shown.text
    assert "[f2] fruit (source: fruit.md)\nBananas are yellow." in shown.text
    assert kept.text.startswith("Keep f1.")
    assert ignored.text.startswith("Ignore f2.")
    assert unknown.text.startswith("Unknown finding id 'f9'")
    assert "Noted: 1 finding(s) kept." in enough.text
    outcome = step.last_outcome
    assert [f.label for f in outcome.findings] == ["geography"]
    assert outcome.undecided == 1
    assert outcome.note.endswith("ignored 1, left 1 undecided.")


def test_findings_as_and_question_field(scripted_model):
    judge = ScriptedJudge(keep={"geography"})
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(
        judge=judge,
        field="prompt",
        question_field="question",
        set_as="answer",
        findings_as="found",
    )
    item = {"prompt": "Answer briefly: " + QUESTION, "question": QUESTION}
    [out] = step([item])
    assert out["answer"] == "Paris."
    assert [f.label for f in out["found"].findings] == ["geography"]
    assert judge.prompts[0].startswith(f"Question: {QUESTION}\n")


@pytest.mark.parametrize("kwargs", [{"set_as": "answer"}, {"findings_as": "found"}])
def test_a_plain_item_is_refused_before_the_model_runs(scripted_model, kwargs):
    # nothing scripted: a model call would fail with "no turn left"
    with pytest.raises(ValueError, match=f"{next(iter(kwargs))}=.*item is a str"):
        list(_step(**kwargs)([QUESTION]))
    assert scripted_model.calls == []


def test_findings_as_without_the_loop_still_accepts_plain_items(scripted_model):
    scripted_model.say("Paris.")
    step = LLMAgent(source="scripted", model="any", findings_as="found")
    assert list(step([QUESTION])) == ["Paris."]


def test_each_item_gets_a_fresh_session_even_in_one_conversation(scripted_model):
    for answer in ("Paris.", "Still Paris."):
        scripted_model.call("search_notes", query="capital").say(answer)
    step = _step(multi_turn=True)
    assert list(step([QUESTION, QUESTION])) == ["Paris.", "Still Paris."]
    first, second = scripted_model.tool_results
    assert "kept 2, ignored 1" in first.text
    assert "kept 2, ignored 1" in second.text  # not "no new findings"
    assert step.last_outcome.note.startswith("Agent stopped without saying")


def test_an_error_is_recorded_in_the_outcome_and_raised(scripted_model):
    scripted_model.call("search_notes", query="capital")  # then the script runs out
    step = _step()
    with pytest.raises(Exception, match="no turn left"):
        list(step([QUESTION]))
    assert step.last_outcome.error
    assert [f.label for f in step.last_outcome.findings] == ["geography", "history"]


def test_session_tools_without_a_session_say_so(scripted_model):
    scripted_model.call("enough_information", reason="x").say("ok")
    list(LLMAgent(source="scripted", model="any", tools=["enough_information"])(["q"]))
    assert (
        "only works while a findings session runs"
        in scripted_model.tool_results[0].text
    )


# -- finding tools -----------------------------------------------------------


def test_a_finding_tool_without_a_session_returns_text(scripted_model):
    scripted_model.call("lookup_fact", topic="Paris").say("ok")
    list(LLMAgent(source="scripted", model="any", tools=["lookup_fact"])(["q"]))
    assert scripted_model.tool_results[0].text == "[1] lookup_fact\nA fact about Paris."
    assert search_notes.tool_spec["inputSchema"]["json"]["required"] == ["query"]
    assert "tool_context" not in str(search_notes.tool_spec)
    assert render_findings([]) == "Nothing found."


def test_a_finding_tool_may_take_the_tool_context(scripted_model):
    seen = []

    @finding_tool(register=False)
    def with_context(q: str, tool_context: ToolContext) -> str:
        """Echo.

        Args:
            q: Anything.
        """
        seen.append(tool_context.tool_use["name"])
        return q

    scripted_model.call("with_context", q="hello").say("ok")
    judge = ScriptedJudge(keep={"with_context"})
    list(_step(tools=[with_context], judge=judge)([QUESTION]))
    assert seen == ["with_context"]
    assert "[f1] with_context\nhello" in scripted_model.tool_results[0].text


def test_the_hook_alone_serves_a_plain_strands_agent(scripted_model):
    from strands import Agent

    session = FindingsSession(QUESTION, tell=lambda _m: None)
    hook = JudgeFindings(session, ScriptedJudge(keep={"history"}))
    scripted_model.call("search_notes", query="capital").say("Paris.")
    agent = Agent(
        model=scripted_model,
        tools=[search_notes],
        hooks=[hook],
        callback_handler=None,
    )
    agent(QUESTION)
    assert [f.label for f in session.kept] == ["history"]


# -- chatterlang -------------------------------------------------------------


def test_a_finding_tool_script_and_a_judged_agent_need_no_python(scripted_model):
    script = """
    findingTool[name="search_facts",
                description="Search the facts for the given text.",
                script='| lambda[expression="item.splitlines()"] | flatten'];
    INPUT FROM echo[data="What is the capital of France?"]
    | llmAgent[source="scripted", model="any", tools=["search_facts", "enough_information"], judge=True]
    """
    scripted_model.call(
        "search_facts",
        text="Paris is the capital of France.\nBananas are yellow.",
        looking_for="",
    )
    scripted_model.say("f1 keep the answer\nf2 ignore fruit")
    scripted_model.call("enough_information", reason="have it")
    scripted_model.say("Paris.")
    assert list(compiler.compile(script)()) == ["Paris."]
    search = scripted_model.tool_results[0]
    assert "kept 1, ignored 1" in search.text
    assert "[f1] search_facts\nParis is the capital of France." in search.text


def test_make_finding_tool_documents_looking_for():
    tool = make_finding_tool("t", "Find things.", "| lambda[expression='item']")
    properties = tool.tool_spec["inputSchema"]["json"]["properties"]
    assert set(properties) == {"text", "looking_for"}
    with pytest.raises(Exception, match="does not compile"):
        make_finding_tool("bad", "x", "| noSuchSegment")
