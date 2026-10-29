# The findings loop

*Part of the [talkpipe-agents](../README.md) documentation. Read
[Tools and skills](tools-and-skills.md) first; this page builds on the
tool registry and `scriptTool`.*

Many agents gather before they answer: they call a search, a lookup, a
calculation, a fetch, one result at a time, until they know enough. Any
one call may or may not help. If the model has to sort the useful
results from the rest on its own turns, it spends a turn on every
decision and can skip deciding altogether. The findings loop moves that
work into the agent loop:

```
tool → findings → judge → session → reply to the model
```

- A **finding tool** returns *findings* (pieces of information) rather
  than prose. It knows nothing about the loop.
- After each call, and before the model sees the result, a **judge**
  (one batched model call) decides keep or ignore for every new
  finding. It judges against the question, against what this call was
  `looking_for`, and against what is already known.
- A per-question **session** accumulates the kept findings, skips any
  it has already decided on, and enforces the stop rule.
- The model then reads a standard reply: the kept findings in full,
  one line per ignored finding with the reason, what is already known,
  and how many turns it has left. It chooses its next call from that,
  and ends the run by calling `enough_information`.

Turning it on is one parameter:

```
INPUT FROM "What is the capital of France, and since when?"
| llmAgent[source="ollama", model="qwen3", tools=["search_notes"], judge=True]
| print
```

## A complete example in Python

This example has three finding tools of different kinds: a search over
notes, a calculation, and a URL fetch. One agent with `judge=True`
answers from what they turn up:

```python
# findings_example.py
import urllib.request
from html.parser import HTMLParser

from talkpipe_agents import Finding, LLMAgent, finding_tool

NOTES = {
    "eiffel.md": "The Eiffel Tower in Paris is 330 metres tall including antennas.",
    "washington.md": "The Washington Monument in Washington, D.C. is 169 metres tall.",
    "bananas.md": "Bananas are botanically berries and grow in clusters called hands.",
}


@finding_tool
def search_notes(query: str, looking_for: str = "") -> list[Finding]:
    """Search my notes by keyword; returns the notes sharing a word with the query.

    Args:
        query: Keywords to search for.
        looking_for: What this search should find that the answer still lacks.
    """
    words = {w.lower().strip("?.,") for w in query.split() if len(w) > 3}
    return [
        Finding(text, label=name, source=name)
        for name, text in NOTES.items()
        if words & {w.lower().strip("?.,") for w in text.split()}
    ]


@finding_tool
def calculate(expression: str, looking_for: str = "") -> str:
    """Evaluate an arithmetic expression such as 330 / 169.

    Args:
        expression: The arithmetic expression.
        looking_for: What this calculation should establish.
    """
    if not set(expression) <= set("0123456789.+-*/() "):
        return f"cannot evaluate {expression!r}"
    return f"{expression} = {eval(expression):.4g}"  # digits and operators only


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


@finding_tool
def fetch_page(url: str, looking_for: str = "") -> Finding:
    """Fetch a web page and return its text.

    Args:
        url: The https:// URL to fetch.
        looking_for: What this page should supply.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "findings-example"})
    with urllib.request.urlopen(request, timeout=20) as response:
        parser = _Text()
        parser.feed(response.read(200_000).decode("utf-8", "replace"))
    return Finding(" ".join(parser.parts)[:3000], label=url, source=url)


agent = LLMAgent(
    source="ollama",
    model="qwen3",
    tools=["search_notes", "calculate", "fetch_page"],
    judge=True,
    progress=print,
)
[answer] = agent(["How many times taller is the Eiffel Tower than the Washington Monument?"])
print(answer)
print(agent.last_outcome.note)
```

`progress=print` shows the trace as the agent works. A run looks like
this (abridged):

```
agent → search_notes({'query': 'Eiffel Tower height'})
agent → search_notes({'query': 'Washington Monument height'})
agent · keep f1 (washington.md) — Provides the Washington Monument's height (169 metres) … Kept 1 finding(s), at most 10. 7 of 8 turns left.
agent ← search_notes: 1 finding(s), kept 1, ignored 0. Kept 1 finding(s), at most 10. 7 of 8 turns left.
agent · keep f2 (eiffel.md) — Provides the Eiffel Tower's height (330 metres) … Kept 2 finding(s), at most 10. 7 of 8 turns left.
agent → calculate({'expression': '330 / 169', 'looking_for': "The ratio of the two heights"})
agent · keep f3 (calculate) — Calculates the height ratio as 330 ÷ 169 ≈ 1.953 … 6 of 8 turns left.
agent → enough_information({'reason': 'Both heights and the ratio are established'})
Agent judged it had enough to answer (Both heights and the ratio are established): kept 3 findings after 3 calls, ignored 0.
```

Each kept line is the judge's decision, made inside the loop. The
countdown runs down with every turn the agent takes, and the last line
is the run's outcome.

## Finding tools

`@finding_tool` turns an ordinary function into an agent tool and
registers it by name, the way `register_agent_tool` does. As with any
tool, the docstring describes the tool to the model and the type hints
give its schema. The function can return:

- `Finding` objects;
- strings (each one a finding labelled with the tool's name);
- dicts with `text` (or `content`), plus optional `label`, `source` and
  `key` (other keys go to `meta`);
- talkpipe `SearchResult`s, or dicts shaped like one (`doc_id`,
  `score`, `document`), through `Finding.from_search_result`.

It can return one of these or a list of them. A `Finding` has:

| Field | Meaning |
|---|---|
| `text` | what the judge and the agent read |
| `label` | a title, file name or URL; names the finding in the trace and the "already known" notes |
| `source` | where it came from; the answer cites it |
| `key` | its identity across calls, so the same finding returned twice is decided only once; defaults to a hash of `text` (a search hit's `doc_id`) |
| `meta` | anything else; a search hit's original `SearchResult` is kept as `meta["result"]` |

A finding tool may take a `looking_for` argument. This is a
convention, not a requirement: the model says in one phrase what the
call should find that the answer still lacks, and the judge keeps a
finding that supplies exactly that, even if it is not the whole
answer. Once the run is over (the agent has said it has enough, or the
maximum is kept), the tool refuses further calls without running the
function.

Outside the loop, a finding tool is still an ordinary tool. Given to an
agent without `judge=True`, its findings come back as numbered text.

## Finding tools in chatterlang

`findingTool` is `scriptTool` for findings. Each item the script emits
becomes a finding. With it, adding a tool is a script edit, with no
Python:

```
findingTool[name="read_page",
            description="Fetch a web page by its https:// URL and return its text, in passages.",
            input_description="The https:// URL of the page to read.",
            script="| downloadURL | htmlToText | splitText[criteria=1500]"];
INPUT FROM "According to https://www.iana.org/help/example-domains what are example.com and example.net for?"
| llmAgent[source="ollama", model="qwen3",
           tools=["read_page", "enough_information"], judge=True]
| print
```

Each 1500-character passage of the page is judged separately, so only
the passages that help reach the answer. The tool takes `text`, which is
fed to the script as its input stream, and `looking_for`, which goes to
the judge. Like `scriptTool`, the tool is registered when the script
compiles, and each call runs a fresh instance of the script.
`make_finding_tool(name, description, script)` builds the same tool
from Python.

## `llmAgent` parameters

| Parameter | Default | Meaning |
|---|---|---|
| `judge` | `False` | `True` runs the loop and judges with the agent's own model and source; from Python, a `FindingsJudge` (or `ScriptedJudge`) to judge with instead |
| `findings` | `False` | runs the loop without a judge: the agent sees every finding with an id and decides itself with `keep_finding` / `ignore_finding` |
| `max_kept` | `10` | the most findings one item may keep. This is a backstop, not a goal: past it, relevant findings are passed over and the run is over |
| `question_field` | — | the property holding the question findings are judged against (default: the prompt) |
| `findings_as` | — | sets the run's `FindingsOutcome` on the item |
| `hooks` | — | from Python: more Strands hook providers for every agent |

With the loop on, `llmAgent` does four things:

- It appends the protocol (how to use `looking_for` and when to call
  `enough_information`) to the system prompt.
- It adds `enough_information` to the tools if it is not already there
  (plus `keep_finding` and `ignore_finding` with `findings=True`).
- It starts a fresh session for every item, even with
  `multi_turn=True`.
- It appends the turn countdown to every tool reply, including replies
  from tools that are not finding tools. The countdown is counted
  against `max_turns`.

The agent's final reply is its answer, written from the kept findings,
as the protocol asks.

`enough_information` is refused while nothing is kept: there would be
nothing to answer from. When a question cannot be answered from what
the tools find, the agent says so instead.

### The outcome

After each item, `last_outcome` holds a `FindingsOutcome`:

- `findings`: the kept `Finding`s, in keep order;
- `note`: one line saying how the run ended: it judged it had enough,
  filled the maximum, hit the turn cap, found nothing relevant to
  keep, stopped without saying it had enough, or stopped with an
  error;
- `trace`: every progress line;
- `error` and `undecided`.

### Answering with talkpipe's RAG segments

Instead of the agent's own reply, you can answer from the kept findings
with talkpipe's RAG stages, like any retrieval. `Finding.to_search_result()`
gives back the original `SearchResult`, or builds one from the
finding's text, label and source:

```python
from talkpipe.llm.chat import LLMPrompt
from talkpipe.pipelines.basic_rag import AppendRAGSources, ConstructRAGPrompt

answer = (
    ConstructRAGPrompt(prompt_directive="Answer the question.",
                       background_field="background", content_field="question",
                       set_as="prompt")
    | LLMPrompt(source="ollama", model="qwen3", field="prompt", set_as="partial")
    | AppendRAGSources(partial_answer_field="partial", set_as="answer")
).as_function(single_in=True, single_out=True)

outcome = agent.last_outcome
result = answer({"question": question,
                 "background": [f.to_search_result() for f in outcome.findings]})
```

## The pieces, for use elsewhere

Each part also works on its own, so the loop can be carried into
another system with different tools:

- **`FindingsSession(question, max_kept=…, max_turns=…, tell=…)`**
  holds one question's state. It provides `register` (short ids `f1`,
  `f2`, …, deduplicated by key), `apply`, `kept`, `ignored`, `notes`,
  `known_notes()`, `seen_keys()`, `declare_enough`, the `guard` that
  finding tools enforce, `note_turn` / `turns_left` / `turns_note()`,
  `reply(...)` (the standard tool reply) and `outcome(...)`. The
  `stop_instruction` argument sets what the agent is told once the run
  is over. The default tells it to answer from the kept findings; an
  application that answers in a later step can tell it to reply `done`.
- **`FindingsJudge`** is the judge. Its `judge(findings, question,
  looking_for, known, seen)` returns `Judgement`s in order. The model
  is any agent model source; `generate=` (prompt in, reply text out)
  replaces it, and subclasses can override `decide`. Replies are read
  leniently by `parse_judgements`: JSON bare, fenced or in prose, a
  bare list, or one `<id> <keep|ignore> <reason>` line per finding.
  Some models ignore a requested output format, so schema-guided
  generation alone is not enough. When the model fails, every finding
  is kept with a `not judged: …` reason; the judge never raises.
- **`judgeFindings`** is the same judge as a segment, for pipelines
  without an agent:

  ```
  | judgeFindings[findings_field="hits", question_field="question", drop_ignored=True]
  ```

  It sets `judged` to `{"finding", "keep", "reason"}` dicts, or, with
  `drop_ignored`, to the kept findings only.
- **`JudgeFindings(session, judge)`** is the Strands hook provider.
  Install it on any Strands `Agent`, together with your finding tools
  and `enough_information`, and the loop runs there. Without a judge,
  findings are registered and shown undecided.

## Testing it offline

`talkpipe_agents.testing.ScriptedJudge` decides by label instead of
asking a model. Pair it with the [scripted model](testing.md) and the
whole loop runs offline:

```python
from talkpipe_agents import LLMAgent
from talkpipe_agents.testing import ScriptedJudge


def test_answers_from_the_kept_note(scripted_model):
    scripted_model.call("search_notes", query="Eiffel Tower height")
    scripted_model.call("enough_information", reason="have the height")
    scripted_model.say("330 metres.")
    judge = ScriptedJudge(keep={"eiffel.md"})
    step = LLMAgent(source="scripted", model="any", tools=["search_notes"], judge=judge)

    assert list(step(["How tall is the Eiffel Tower?"])) == ["330 metres."]
    assert [f.label for f in step.last_outcome.findings] == ["eiffel.md"]
    assert judge.batches == [["eiffel.md"]]
```

`ScriptedJudge(keep=...)` takes either the labels to keep, or a
callable that is given each `Finding` and returns `True`/`False` or
`(keep, reason)`. `error=` makes every judgement fail, for testing the
keep-everything fallback. `batches` and `prompts` record what each
judgement was shown. To test the JSON or prose parsing as well, use
`judge=True`: the judge then asks the scripted model, so script its
reply as a `.say(...)` right after the tool call it judges.
