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
    Judgement,
    LLMAgent,
    Need,
    as_findings,
    finding_tool,
    make_finding_tool,
    parse_judgements,
    parse_verdict,
    register_agent_tool,
)
from talkpipe_agents.findings import render_findings
from talkpipe_agents.testing import ScriptedJudge, ScriptedToolErrorWarning

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


def test_kept_findings_rank_best_first_and_a_better_one_displaces_the_weakest():
    session, lines = _session(max_kept=2)
    f1, f2, f3 = (session.register(f) for f in NOTES)
    session.apply(f1, True, "the answer", score=3)
    session.apply(f2, True, "a fruit", score=1)
    assert [f.label for f in session.kept] == ["geography", "fruit"]
    assert session.score_of(f3) == 3  # ungraded: the default
    assert session.at_capacity
    # a new finding graded no better than the weakest does not displace it
    assert session.displaceable_by(1) is None
    assert session.displaceable_by(2) == f2
    assert session.displaceable_by(None) == f2  # the default grade is 3
    session.apply(f2, False, f"displaced by {f3}")
    session.apply(f3, True, "the date", score=5)
    assert session.kept_ids == [f1, f3]
    assert [f.label for f in session.kept] == ["history", "geography"]  # best first
    assert [f.label for f in session.ignored] == ["fruit"]
    assert session.reasons[f2] == f"displaced by {f3}"
    assert session.known_notes() == ["geography — the answer", "history — the date"]
    assert lines[0].startswith("agent · keep f1 (geography) [3/5] — the answer.")
    assert lines[2].startswith("agent · ignore f2 (fruit) — displaced by f3.")
    assert "No longer kept, outranked at the 2-finding maximum:\n\n[f2] fruit" in (
        session.reply([f3], [], displaced=[f2])
    )
    assert session.displaceable_by(3) is None  # the weakest kept is graded 3
    assert session.displaceable_by(5) == f1
    session, _ = _session(max_kept=2)
    session.apply(session.register(NOTES[0]), True)
    assert session.displaceable_by(5) is None  # not at capacity


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


def test_the_grade_is_optional_and_clamped():
    from talkpipe_agents.findings import JUDGE_SYSTEM_PROMPT

    graded = parse_judgements(
        '{"decisions": [{"id": "f1", "keep": true, "score": 4, "reason": "a"}, '
        '{"id": "f2", "keep": false, "grade": "9", "reason": "b"}, '
        '{"id": "f3", "keep": true, "score": "high", "reason": "c"}, '
        '{"id": "f4", "keep": true, "reason": "d"}]}'
    )
    assert [d.score for d in graded] == [4, 5, None, None]
    assert [d.score for d in parse_judgements(JSON_REPLY)] == [None, None]
    assert [d.score for d in parse_judgements("f1 keep yes\nf2 ignore no")] == [
        None,
        None,
    ]
    judged = FindingsJudge(generate=lambda _p: JSON_REPLY).judge(NOTES[:2], QUESTION)
    assert [j.score for j in judged] == [None, None]
    # the prompt asks for the grade, and keeps by default: no novelty gate
    assert '"score": 4' in JUDGE_SYSTEM_PROMPT
    assert "The default is keep" in JUDGE_SYSTEM_PROMPT
    assert "not already known" not in JUDGE_SYSTEM_PROMPT
    assert "must add" not in JUDGE_SYSTEM_PROMPT
    assert "Never ignore a finding because another finding" in JUDGE_SYSTEM_PROMPT


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
    assert [(j["finding"].text, j["keep"], j["score"]) for j in out["judged"]] == [
        ("Paris.", True, None),
        ("Bananas.", False, None),
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


def test_a_better_graded_finding_displaces_the_weakest_at_capacity(scripted_model):
    # NOTES arrive as geography (3), fruit (1), history (5): the first two
    # fill max_kept=2, then history outranks fruit and takes its place.
    grades = {"geography": 3, "fruit": 1, "history": 5}
    judge = ScriptedJudge(keep=lambda f: (True, f"graded {f.label}", grades[f.label]))
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(max_kept=2, judge=judge)
    list(step([QUESTION]))
    reply = scripted_model.tool_results[0].text
    assert "3 new finding(s) from search_notes: kept 3, ignored 0." in reply
    assert "No longer kept, outranked at the 2-finding maximum:" in reply
    assert "[f2] fruit — displaced by f3" in reply
    assert "Already known:\n- geography — graded geography\n- history" in reply
    assert "You have kept the maximum of 2 findings." in reply
    outcome = step.last_outcome
    assert [f.label for f in outcome.findings] == ["history", "geography"]
    trace = "\n".join(outcome.trace)
    assert "agent · keep f3 (history) [5/5] — graded history." in trace
    assert "agent · ignore f2 (fruit) — displaced by f3." in trace
    assert "3 finding(s), kept 3, ignored 0, 1 displaced." in trace
    assert outcome.note.startswith(
        "Agent filled its 2-finding maximum: kept 2 findings after 1 call, ignored 1."
    )


def test_an_equal_grade_is_passed_over_not_displaced(scripted_model):
    # Ungraded decisions rank as the default, so a run without grades
    # keeps in arrival order and passes the rest over, as before.
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(max_kept=2, judge=ScriptedJudge(keep=lambda _f: True))
    list(step([QUESTION]))
    reply = scripted_model.tool_results[0].text
    assert "1 more relevant finding(s) were not kept" in reply
    assert "No longer kept" not in reply
    assert [f.label for f in step.last_outcome.findings] == ["geography", "fruit"]


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


# -- judge robustness: cut-off replies, re-asking, batches --------------------

TRUNCATED_REPLY = (
    '{"decisions": [{"id": "f1", "keep": true, "reason": "the answer"}, '
    '{"id": "f2", "keep": false, "reason": "fruit"}, {"id": "f3", "keep": tr'
)


def test_a_reply_cut_off_mid_json_gives_the_complete_decisions():
    assert [(d.id, d.keep, d.reason) for d in parse_judgements(TRUNCATED_REPLY)] == [
        ("f1", True, "the answer"),
        ("f2", False, "fruit"),
    ]


def test_undecided_findings_are_asked_about_again_on_their_own():
    prompts: list[str] = []

    def generate(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return TRUNCATED_REPLY  # the model's output limit cut f3 off
        return '{"decisions": [{"id": "f3", "keep": false, "reason": "repeats f1"}]}'

    judged = FindingsJudge(generate=generate).judge(NOTES, QUESTION)
    assert [(j.finding.label, j.keep, j.reason) for j in judged] == [
        ("geography", True, "the answer"),
        ("fruit", False, "fruit"),
        ("history", False, "repeats f1"),
    ]
    assert len(prompts) == 2
    # The second prompt holds only the undecided finding, under its own label.
    assert "[f3] history" in prompts[1]
    assert "[f1]" not in prompts[1]
    assert "[f2]" not in prompts[1]


def test_retries_zero_keeps_an_undecided_finding_after_one_call():
    prompts: list[str] = []

    def generate(prompt):
        prompts.append(prompt)
        return TRUNCATED_REPLY

    judged = FindingsJudge(generate=generate, retries=0).judge(NOTES, QUESTION)
    assert [j.keep for j in judged] == [True, False, True]
    assert judged[2].reason == "not judged: the model gave no decision for it"
    assert len(prompts) == 1
    # With the default one retry, a model that keeps skipping it is asked twice.
    prompts.clear()
    FindingsJudge(generate=generate).judge(NOTES, QUESTION)
    assert len(prompts) == 2


def test_batch_size_judges_a_few_findings_per_call():
    prompts: list[str] = []

    def generate(prompt):
        prompts.append(prompt)
        labels = [line[1:3] for line in prompt.splitlines() if line.startswith("[f")]
        return "\n".join(f"{label} keep ok" for label in labels)

    judged = FindingsJudge(generate=generate, batch_size=2).judge(NOTES, QUESTION)
    assert [(j.finding.label, j.keep) for j in judged] == [
        ("geography", True),
        ("fruit", True),
        ("history", True),
    ]
    assert len(prompts) == 2
    assert "[f1] geography" in prompts[0]
    assert "[f2] fruit" in prompts[0]
    assert "[f3] history" in prompts[1]
    assert "[f1]" not in prompts[1]


# -- the score floor ---------------------------------------------------------


def _hit(score: float, text: str) -> SearchResult:
    return SearchResult(score=score, doc_id=text, document={"content": text})


def test_a_finding_knows_its_search_score():
    assert Finding.from_search_result(_hit(0.42, "x")).search_score == 0.42
    assert Finding("x", meta={"score": 3}).search_score == 3.0
    assert Finding("x").search_score is None
    assert Finding("x", meta={"score": True}).search_score is None


@finding_tool(register=False, min_score=0.5)
def search_scored(query: str, looking_for: str = "") -> list[SearchResult]:
    """Search with scores.

    Args:
        query: What to search for.
        looking_for: What the answer still lacks.
    """
    if query == "nothing good":
        return [_hit(0.1, "weak one"), _hit(0.2, "weak two")]
    return [
        _hit(0.9, "Paris is the capital of France."),
        _hit(0.3, "Bananas are yellow."),
        _hit(0.5, "France's capital has been Paris since 987."),
    ]


def test_hits_under_the_floor_are_left_out_before_the_judge(scripted_model):
    scripted_model.call("search_scored", query="capital")
    scripted_model.call("enough_information", reason="known")
    scripted_model.say("Paris.")
    judge = ScriptedJudge(keep=lambda f: "Paris" in f.text)
    step = _step(judge=judge, tools=[search_scored])
    assert list(step([QUESTION])) == ["Paris."]
    reply = scripted_model.tool_results[0].text
    assert "2 new finding(s) from search_scored: kept 2, ignored 0." in reply
    assert "1 result(s) scored below the tool's floor and were left out." in reply
    assert "Bananas" not in reply
    # The judge was shown only the findings above the floor.
    assert len(judge.batches) == 1
    assert len(judge.batches[0]) == 2
    assert "Bananas" not in judge.prompts[0]
    assert [f.search_score for f in step.last_outcome.findings] == [0.9, 0.5]
    assert "2 finding(s), kept 2, ignored 0, 1 below the score floor" in "\n".join(
        step.last_outcome.trace
    )


def test_nothing_above_the_floor_costs_no_judge_call(scripted_model):
    scripted_model.call("search_scored", query="nothing good")
    scripted_model.say("Nothing relevant was found.")
    judge = ScriptedJudge(keep=lambda f: True)
    step = _step(judge=judge, tools=[search_scored], max_turns=3)
    list(step([QUESTION]))
    reply = scripted_model.tool_results[0].text
    assert reply.startswith(
        "No new findings from search_scored. 2 result(s) scored below the "
        "tool's floor and were left out. Try different terms."
    )
    assert "2 of 3 turns left." in reply
    assert judge.batches == []
    assert step.last_outcome.note.startswith("Agent found nothing relevant to keep")


def test_the_floor_applies_without_a_session_and_by_hand(scripted_model):
    scripted_model.call("search_scored", query="nothing good").say("ok")
    list(LLMAgent(source="scripted", model="any", tools=[search_scored])(["q"]))
    assert scripted_model.tool_results[0].text == (
        "Nothing found. 2 result(s) scored below the tool's floor and were left out."
    )
    scripted_model.call("search_scored", query="capital").say("ok")
    step = _step(judge=False, findings=True, tools=[search_scored])
    list(step([QUESTION]))
    reply = scripted_model.tool_results[1].text
    assert "2 finding(s) from search_scored." in reply
    assert "1 result(s) scored below the tool's floor" in reply
    assert len(step.last_outcome.trace)
    assert step.last_outcome.undecided == 2


def test_a_finding_tool_script_takes_a_score_floor(scripted_model):
    from talkpipe.chatterlang import register_segment
    from talkpipe.pipe import core

    @register_segment("scoredHits")
    @core.segment()
    def scored_hits(items):
        for _ in items:
            yield _hit(0.9, "Paris is the capital of France.")
            yield _hit(0.1, "Bananas are yellow.")

    script = """
    findingTool[name="search_hits", description="Search.", min_score=0.5,
                script="| scoredHits"];
    INPUT FROM echo[data="What is the capital of France?"]
    | llmAgent[source="scripted", model="any", tools=["search_hits"], judge=True]
    """
    scripted_model.call("search_hits", text="capital", looking_for="")
    scripted_model.say("f1 keep the answer")
    scripted_model.say("Paris.")
    assert list(compiler.compile(script)()) == ["Paris."]
    reply = scripted_model.tool_results[0].text
    assert "1 new finding(s) from search_hits: kept 1, ignored 0." in reply
    assert "1 result(s) scored below the tool's floor" in reply


# -- the time budget ---------------------------------------------------------


class _Clock:
    def __init__(self, step: float = 0.0) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_the_session_knows_when_its_time_budget_is_spent():
    clock = _Clock()
    session = FindingsSession(QUESTION, max_seconds=10, clock=clock)
    fid = session.register(NOTES[0])
    session.apply(fid, True, "the answer")
    assert not session.timed_out
    assert session.guard("search_notes") == ""
    clock.now = 10
    assert session.timed_out
    assert session.stop_note == "the 10-second time budget is spent"
    assert session.guard("search_notes").startswith(
        "The search is over — the 10-second time budget is spent."
    )
    assert session.outcome("end_turn").note.startswith(
        "Agent hit its 10-second time budget: kept 1 finding"
    )
    # Saying it had enough still counts as the intended ending.
    session.enough = True
    assert session.outcome("end_turn").note.startswith("Agent judged it had enough")


def test_max_seconds_refuses_tool_calls_once_spent(scripted_model, monkeypatch):
    import time

    monkeypatch.setattr(time, "monotonic", _Clock(step=20.0))
    scripted_model.call("search_notes", query="capital")
    scripted_model.say("I ran out of time before finding anything.")
    lines: list[str] = []
    step = _step(max_seconds=30, max_turns=4, progress=lines.append)
    with pytest.warns(ScriptedToolErrorWarning, match="time budget is spent"):
        answer = list(step([QUESTION]))
    assert answer == ["I ran out of time before finding anything."]
    result = scripted_model.tool_results[0]
    assert result.status == "error"
    assert "The 30-second time budget is spent" in result.text
    assert "stop calling tools and answer now from what you have" in result.text
    assert searched == []  # the tool itself never ran
    assert "agent: stopped at the time budget (30 s)" in lines
    assert step.last_outcome.note.startswith("Agent hit its 30-second time budget")


def test_max_seconds_must_be_positive():
    with pytest.raises(ValueError, match="max_seconds must be positive"):
        _step(max_seconds=0)
    with pytest.raises(ValueError, match="max_seconds must be a number"):
        _step(max_seconds="soon")


# -- the judge's verdict -----------------------------------------------------
#
# Besides deciding on each finding, the judge says whether what is kept
# would answer the question and what it still lacks; the session passes
# that on to the model and the outcome.


def test_parse_verdict_reads_enough_and_missing():
    from talkpipe_agents import Verdict, parse_judgements, parse_verdict

    reply = (
        '{"decisions": [{"id": "f1", "keep": true, "score": 5, "reason": "r"}], '
        '"enough": false, "missing": "the tower\'s height"}'
    )
    verdict = parse_verdict(reply)
    assert isinstance(verdict, Verdict)
    assert (verdict.enough, verdict.missing) == (False, "the tower's height")
    assert [d.id for d in verdict.decisions] == ["f1"]
    assert [d.id for d in parse_judgements(reply)] == ["f1"]
    # Without the fields, or as a bare list, there is no verdict.
    assert parse_verdict('{"decisions": [{"id": "f1", "keep": true}]}').enough is None
    assert parse_verdict('[{"id": "f1", "keep": "yes"}]').enough is None
    assert parse_verdict("f1 keep fine").enough is None
    # A reply cut off mid-JSON gives its complete decisions and no verdict.
    cut = '{"decisions": [{"id": "f1", "keep": true}, {"id": "f2", "ke'
    assert (
        [d.id for d in parse_verdict(cut).decisions],
        parse_verdict(cut).enough,
    ) == (
        ["f1"],
        None,
    )


def test_the_judge_returns_the_verdict_with_its_judgements():
    from talkpipe_agents import Judged

    replies = iter(
        [
            '{"decisions": [{"id": "f1", "keep": true}], "enough": true}',
            '{"decisions": [{"id": "f1", "keep": true}]}',
        ]
    )
    judge = FindingsJudge(generate=lambda _p: next(replies))
    judged = judge.judge(NOTES[:1], QUESTION)
    assert isinstance(judged, Judged)
    assert list(judged) == judged
    assert (judged.enough, judged.missing) == (True, "")
    assert judge.judge(NOTES[:1], QUESTION).enough is None


def test_the_session_tells_the_model_what_the_judge_finds_missing(scripted_model):
    judge = ScriptedJudge(keep={"geography"}, enough=False, missing="the population")
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(judge=judge, field="q", set_as="answer", findings_as="outcome")
    (item,) = step([{"q": QUESTION}])
    reply = scripted_model.tool_results[0].text
    assert "The judge finds the answer still lacks: the population." in reply
    outcome = item["outcome"]
    assert (outcome.sufficient, outcome.missing) == (False, "the population")
    assert any("Judge: still missing the population." in line for line in outcome.trace)


def test_the_session_tells_the_model_when_the_judge_says_enough(scripted_model):
    judge = ScriptedJudge(keep={"geography"}, enough=True)
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(judge=judge, field="q", set_as="answer", findings_as="outcome")
    (item,) = step([{"q": QUESTION}])
    reply = scripted_model.tool_results[0].text
    assert "call enough_information now" in reply
    assert item["outcome"].sufficient is True
    assert item["outcome"].missing == ""
    assert any(
        line.endswith("Judge: enough to answer.") for line in item["outcome"].trace
    )


def test_a_verdict_with_nothing_kept_is_not_passed_on(scripted_model):
    judge = ScriptedJudge(keep=(), enough=False, missing="everything")
    scripted_model.call("search_notes", query="capital").say("No idea.")
    step = _step(judge=judge, field="q", set_as="answer", findings_as="outcome")
    (item,) = step([{"q": QUESTION}])
    assert "The judge finds" not in scripted_model.tool_results[0].text
    assert item["outcome"].sufficient is False


def test_a_judge_returning_a_plain_list_gives_no_verdict(scripted_model):
    class ListJudge(FindingsJudge):
        def judge(self, findings, question, looking_for="", known=(), seen=(), **kw):
            return [Judgement(f, True, "fine") for f in as_findings(list(findings))]

    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(judge=ListJudge(), field="q", set_as="answer", findings_as="outcome")
    (item,) = step([{"q": QUESTION}])
    assert "The judge finds" not in scripted_model.tool_results[0].text
    assert item["outcome"].sufficient is None
    assert len(item["outcome"].findings) == len(NOTES)


# -- a call before the agent ---------------------------------------------------


def _opening(scripted_model) -> str:
    return scripted_model.prompts[0][0]["content"][0]["text"]


def test_first_tool_judged_enough_means_the_agent_only_answers(scripted_model):
    judge = ScriptedJudge(keep={"geography", "history"}, enough=True)
    scripted_model.say("Paris.")
    step = _step(
        judge=judge,
        first_tool="search_notes",
        findings_as="outcome",
        field="q",
        set_as="a",
    )
    (item,) = step([{"q": QUESTION}])
    # The tool ran with the question, before any model turn, and was judged.
    assert searched == [(QUESTION, "")]
    outcome = item["outcome"]
    assert outcome.ending == "judge_enough"
    assert outcome.note.startswith(
        "Agent found the first call enough: answered without an agent turn: "
        "kept 2 findings after 1 call, ignored 1."
    )
    assert [f.label for f in outcome.findings] == ["geography", "history"]
    trace = " | ".join(outcome.trace)
    assert "agent → search_notes({'query': " in trace
    assert "before the agent" in trace
    assert "enough information, by the judge's verdict" in trace
    text = _opening(scripted_model)
    assert text.startswith(f"Question: {QUESTION}\n")
    assert "the judge finds that what it kept answers the question" in text
    assert "Kept:" in text
    assert "[f1] geography" in text
    # The model took one turn: the answer, with no tool call.
    assert len(scripted_model.prompts) == 1


def test_first_tool_judged_wanting_hands_over_to_the_agent(scripted_model):
    judge = ScriptedJudge(keep={"geography"}, enough=False, missing="the date")
    scripted_model.call("search_notes", query="since when", looking_for="the date")
    scripted_model.call("enough_information", reason="date found").say("987.")
    step = _step(
        judge=judge,
        first_tool="search_notes",
        findings_as="outcome",
        field="q",
        set_as="a",
    )
    (item,) = step([{"q": QUESTION}])
    text = _opening(scripted_model)
    assert "has already been made and judged; its reply is below. Do not repeat" in text
    assert "The judge says it still lacks: the date." in text
    assert "At most 10 findings are kept, and you have 8 turns" in text
    outcome = item["outcome"]
    assert outcome.ending == "enough"
    assert outcome.note.startswith("Agent judged it had enough to answer (date found)")
    assert "after 2 calls" in outcome.note
    # Every note the agent's own search found had been decided by the first
    # call, so the judge was not asked again and the reply says so.
    assert len(judge.prompts) == 1
    assert scripted_model.tool_results[0].text.startswith(
        "No new findings from search_notes."
    )


def test_a_judge_without_a_verdict_leaves_the_first_call_to_the_agent(scripted_model):
    scripted_model.call("enough_information", reason="ok").say("Paris.")
    step = _step(
        first_tool="search_notes", findings_as="outcome", field="q", set_as="a"
    )
    (item,) = step([{"q": QUESTION}])
    assert "Do not repeat it." in _opening(scripted_model)
    assert "still lacks:" not in _opening(scripted_model)
    assert item["outcome"].ending == "enough"


def test_a_failing_first_tool_starts_the_agent_as_usual(scripted_model):
    @finding_tool(register=False)
    def broken(query: str, looking_for: str = "") -> list[Finding]:
        """Always fails.

        Args:
            query: Ignored.
            looking_for: Ignored.
        """
        raise RuntimeError("index offline")

    scripted_model.call("search_notes", query="capital")
    scripted_model.call("enough_information", reason="ok").say("Paris.")
    step = _step(tools=["search_notes", broken], first_tool="broken")
    [answer] = step([QUESTION])
    assert answer == "Paris."
    assert _opening(scripted_model) == QUESTION
    assert any(
        "agent ← broken: error: index offline" in line
        for line in step.last_outcome.trace
    )


def test_first_tool_is_added_to_the_tools_when_missing(scripted_model):
    judge = ScriptedJudge(keep={"geography"}, enough=True)
    scripted_model.say("Paris.")
    step = _step(judge=judge, tools=[], first_tool="search_notes")
    assert "search_notes" in {t.tool_name for t in step.tools}
    assert list(step([QUESTION])) == ["Paris."]


def test_first_tool_without_the_loop_or_not_a_finding_tool_is_refused():
    with pytest.raises(ValueError, match="needs the findings loop"):
        LLMAgent(source="scripted", model="any", first_tool="search_notes")
    with pytest.raises(ValueError, match="not a finding tool"):
        _step(tools=["search_notes", "count_words"], first_tool="count_words")


def test_the_session_names_its_ending_and_the_hook_seeds_without_an_agent():
    session = FindingsSession(QUESTION, max_kept=5, max_turns=3)
    loop = JudgeFindings(session, ScriptedJudge(keep={"geography"}, enough=True))
    reply = loop.seed("search_notes", NOTES, args={"query": QUESTION})
    assert reply.startswith("3 new finding(s) from search_notes: kept 1, ignored 2.")
    assert session.enough
    assert session.enough_by == "judge"
    assert session.ending() == "judge_enough"
    assert "Answer the question now from the kept findings" in session.opening_prompt(
        reply, tool="search_notes"
    )
    # Without a session the hook only renders what was found.
    assert JudgeFindings().seed("t", ["one"]) == "[1] t\none"
    # The other endings, by their conditions.
    fresh = FindingsSession(QUESTION, max_kept=1, max_turns=3)
    assert fresh.ending() == "undecided"
    assert fresh.ending("limit_turns") == "turn_cap"
    assert fresh.ending(error="boom") == "error"
    assert fresh.opening_prompt().startswith(
        f"Question: {QUESTION}\nNothing has been called yet."
    )
    fresh.calls = 1
    assert fresh.ending() == "nothing_relevant"
    fresh.apply(fresh.register(NOTES[0]), True, "fine")
    assert fresh.ending() == "capacity"
    fresh.declare_enough("done")
    assert fresh.ending() == "enough"
    assert fresh.outcome().ending == "enough"


# -- the facts the answer needs ---------------------------------------------


def test_parse_verdict_reads_the_needs_and_derives_the_verdict():
    reply = """{"needed": [{"fact": "the capital", "found": "f1"},
                           {"fact": "since when", "found": null}],
                "decisions": [{"id": "f1", "keep": true, "score": 5, "reason": "r"}],
                "enough": true}"""
    verdict = parse_verdict(reply)
    assert [(n.fact, n.found, n.id) for n in verdict.needs] == [
        ("the capital", "f1", ""),
        ("since when", None, ""),
    ]
    # Derived from the list, whatever the model said about enough.
    assert verdict.enough is False
    assert verdict.missing == "since when"
    all_found = parse_verdict(
        '{"needed": [{"id": "N1", "fact": "the capital", "found": "f1"}],'
        ' "decisions": [{"id": "f1", "keep": true}]}'
    )
    assert (all_found.enough, all_found.missing) == (True, "")
    # A reply without a list reads as before.
    assert parse_verdict('[{"id": "f1", "keep": true}]').needs is None


def test_the_judge_merges_the_needs_of_its_replies():
    replies = iter(
        [
            (
                '{"needed": [{"fact": "a", "found": "f1"}, {"fact": "b", "found": null}],'
                ' "decisions": [{"id": "f1", "keep": true}]}'
            ),
            (
                '{"needed": [{"fact": "b", "found": "f2"}],'
                ' "decisions": [{"id": "f2", "keep": true}]}'
            ),
        ]
    )
    judge = FindingsJudge(generate=lambda prompt: next(replies))
    judged = judge.judge(NOTES[:2], QUESTION)
    assert [(n.fact, n.found) for n in judged.needs] == [("a", "f1"), ("b", "f2")]
    assert judged.enough is True
    assert [j.label for j in judged] == ["f1", "f2"]


def _needs(*facts_found):
    """A ScriptedJudge ``needs`` rule returning the same list every call."""
    return lambda labelled: [Need(fact, found) for fact, found in facts_found]


def test_the_session_derives_the_verdict_from_the_needs(scripted_model):
    judge = ScriptedJudge(
        keep={"geography", "history"},
        needs=_needs(("the capital", "f1"), ("since when", None)),
    )
    scripted_model.call("search_notes", query="capital").say("Paris.")
    step = _step(judge=judge, field="q", set_as="a", findings_as="outcome")
    (item,) = step([{"q": QUESTION}])
    reply = scripted_model.tool_results[0].text
    assert (
        "The answer needs:\n- the capital: stated by f1\n- since when: not yet found"
        in reply
    )
    assert "The judge finds the answer still lacks: since when." in reply
    outcome = item["outcome"]
    assert (outcome.sufficient, outcome.missing) == (False, "since when")
    assert any(
        "Judge: needs 2, 1 found. Judge: still missing since when." in line
        for line in outcome.trace
    )


def test_needs_carry_across_calls_and_the_judge_is_shown_them(scripted_model):
    calls = []

    def needs(labelled):
        calls.append([label for label, _ in labelled])
        if len(calls) == 1:
            return [Need("the capital", "f1"), Need("since when", None)]
        # The second call: the judge keeps the ids and marks the gap found.
        return [Need("the capital", "known", "N1"), Need("since when", "f1", "N2")]

    judge = ScriptedJudge(keep={"geography", "history", "lookup_fact"}, needs=needs)
    scripted_model.call("search_notes", query="capital")
    scripted_model.call("lookup_fact", topic="the date")
    scripted_model.say("Paris, since 987.")
    step = _step(
        judge=judge,
        tools=["search_notes", "lookup_fact"],
        field="q",
        set_as="a",
        findings_as="outcome",
    )
    (item,) = step([{"q": QUESTION}])
    # The second judgement saw the list worked out on the first.
    assert "The answer needs (from earlier calls):" in judge.prompts[1]
    assert "- N1: the capital (covered)" in judge.prompts[1]
    assert "- N2: since when (not yet found)" in judge.prompts[1]
    outcome = item["outcome"]
    assert outcome.sufficient is True
    assert outcome.missing == ""
    assert "call enough_information now" in scripted_model.tool_results[1].text


def test_a_need_is_only_covered_by_a_kept_finding():
    session = FindingsSession(QUESTION, max_kept=1)
    judge = ScriptedJudge(
        keep={"fruit"}, needs=_needs(("the capital", "f1"), ("bananas", "f2"))
    )
    loop = JudgeFindings(session, judge)
    loop.process(NOTES[:2], "", "search_notes")
    # f1 (geography) was ignored, so citing it covers nothing; f2 is kept.
    assert [(n.fact, n.found) for n in session.needs] == [
        ("the capital", None),
        ("bananas", "f2"),
    ]
    assert session.sufficient is False
    assert session.missing == "the capital"


def test_coverage_is_lost_when_the_finding_is_displaced():
    session = FindingsSession(QUESTION, max_kept=1)
    grades = {"geography": 2, "history": 5}
    judge = ScriptedJudge(
        keep=lambda f: (True, "", grades[f.label]),
        needs=lambda labelled: [Need("the capital", labelled[0][0])],
    )
    loop = JudgeFindings(session, judge)
    loop.process(NOTES[:1], "", "search_notes")
    assert session.sufficient is True
    # The better-graded history note displaces geography; the need it
    # covered is open again, and the judge's new citation covers it.
    loop.process(NOTES[2:], "", "search_notes")
    assert session.kept_ids == ["f2"]
    assert [(n.fact, n.found) for n in session.needs] == [("the capital", "f2")]


def test_first_tool_ends_the_run_only_when_every_need_is_found(scripted_model):
    judge = ScriptedJudge(
        keep={"geography", "history"},
        needs=_needs(("the capital", "f1"), ("since when", None)),
    )
    scripted_model.call("enough_information", reason="fine").say("Paris.")
    step = _step(
        judge=judge, first_tool="search_notes", findings_as="o", field="q", set_as="a"
    )
    (item,) = step([{"q": QUESTION}])
    # Not short-circuited: one fact was not found, so the agent ran.
    assert item["o"].ending == "enough"
    assert "The judge says it still lacks: since when." in _opening(scripted_model)


def test_a_judge_without_a_needs_list_still_gives_a_prose_verdict(scripted_model):
    judge = ScriptedJudge(keep={"geography"}, enough=True)
    scripted_model.say("Paris.")
    step = _step(
        judge=judge, first_tool="search_notes", findings_as="o", field="q", set_as="a"
    )
    (item,) = step([{"q": QUESTION}])
    assert item["o"].ending == "judge_enough"
    assert step._loop is not None
