# Testing without a model server

*Part of the [talkpipe-agents](../README.md) documentation. The tool
under test here, `lookup_ticket`, is the one registered in
[Getting started](getting-started.md).*

The package ships the fake model its own tests use. In an environment
with pytest (`pip install pytest` — it is not a runtime dependency),
installing `talkpipe-agents` adds a pytest fixture, `scripted_model`,
that registers a `ScriptedModel` as the agent model source `"scripted"`
for one test (the `model=` string is accepted and ignored): you script
the model's turns — tool calls and answers — and the real agent loop
runs your real tool against them, offline:

```python
# test_my_tools.py — next to my_tools.py, which it imports
from talkpipe_agents import LLMAgent

import my_tools  # noqa: F401  (registers lookup_ticket)


def test_agent_looks_the_ticket_up(scripted_model):
    scripted_model.call("lookup_ticket", ticket_id="T-1").say("Not urgent.")
    step = LLMAgent(source="scripted", model="any", tools=["lookup_ticket"])
    assert list(step(["How urgent is T-1?"])) == ["Not urgent."]
    assert scripted_model.tool_results == [
        ("lookup_ticket", "success", "Printer on fire, third floor")
    ]
```

(The test imports `my_tools`, so run it where that module is importable:
`python -m pytest` in the directory that holds both, or a `tests/`
folder with a `conftest.py`.)
`tool_results` records what each scripted call came back with — tool
name, `"success"` or `"error"`, result text — so the test proves the
real tool ran, not just that the scripted answer was repeated. Two
mistakes are caught rather than masked by that answer: a `.call()` to a
tool the agent was not given (misspelled, or missing from `tools=`)
fails the test outright, and a call the agent reports as an error
(wrong argument names, a tool that raised) is also emitted as a
`ScriptedToolErrorWarning`, so pytest shows it even when the test
passes (a deliberate recovery test wraps it in `pytest.warns`). Running
out of scripted turns fails the test with a message saying where the
agent was (the last line of the traceback, under Strands' event-loop
frames); `remaining` is the turns not yet played (`assert not
scripted_model.remaining` checks the whole script was used); `calls`
holds the message count of each model turn (`[1, 3]` above: the
question; then question, tool use, tool result) and `prompts` the
messages themselves.

## Testing a script as written

The same `"scripted"` source works from a chatterlang script under
test. To test a script as written — one that says `source="ollama"`,
or your application's own `register_agent_model` source — register the
same model under that name for the block:

```python
from talkpipe.chatterlang import compiler

from talkpipe_agents.testing import scripted_source

import my_tools  # noqa: F401  (registers lookup_ticket)

SCRIPT = """
  INPUT FROM "How urgent is ticket T-1?"
  | llmAgent[source="ollama", model="llama3.2", tools=["lookup_ticket"]]
"""  # or read it from the file your application ships


def test_production_script(scripted_model):
    with scripted_source("ollama", model=scripted_model):
        scripted_model.call("lookup_ticket", ticket_id="T-1").say("Not urgent.")
        assert list(compiler.compile(SCRIPT)()) == ["Not urgent."]
```

## Limits and configuration under test

The scripted model reports no token usage, so `max_turns` is enforced
in an offline test (`stop_reason` becomes `"limit_turns"`) while the
token budgets — which depend on the provider's usage numbers — are not;
test those against a real provider.

Outside pytest, `scripted_source()` alone is the context manager behind
the fixture: build the model, script it, and register it for a block —
the `"scripted"` source exists only inside that block (outside it,
`source="scripted"` is an unknown source):

```python
from talkpipe_agents.testing import ScriptedModel, scripted_source

model = ScriptedModel()
model.call("lookup_ticket", ticket_id="T-1").say("Not urgent.")
with scripted_source("scripted", model=model):
    ...  # LLMAgent(source="scripted", ...) / AgentChat(source="scripted", ...)
```

A script that omits `source=` falls back to
`TALKPIPE_DEFAULT_MODEL_SOURCE`, but talkpipe reads its configuration
once per process, so set that variable before starting pytest rather
than with `monkeypatch.setenv`. The fixture is an ordinary pytest
plugin; `pytest -p no:talkpipe_agents` turns it off.
