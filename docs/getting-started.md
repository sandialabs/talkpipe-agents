# Getting started: your tools, in your pipeline

*Part of the [talkpipe-agents](../README.md) documentation. Start with
the README's "Hello, world" if you have not run an agent yet.*

You are writing an application on talkpipe and want an agent step that
can call *your* functions. That takes three things: a tool, a model,
and a script that names both.

## 1. A tool

A tool is a plain Python function with a docstring (the model's
description of it) and type hints (its argument schema). Registering
it gives it a name scripts can use:

```python
# my_tools.py
from talkpipe_agents import register_agent_tool

TICKETS = {"T-1": "Printer on fire, third floor", "T-2": "Password reset"}


@register_agent_tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket by its id (like "T-1") and return its text."""
    return TICKETS.get(ticket_id, f"no ticket {ticket_id}")
```

`register_agent_tool` accepts a plain function or a ready-made
`@strands.tool`; `@register_agent_tool(name="tickets")` registers under
another name — the name a *script* uses, while the model (and the
progress lines, and `scripted_model.call(...)` in a test) still sees the
function's own name. Use `@strands.tool(name="tickets")` under the
registration to rename it for everyone. A tool that needs runtime state (a database handle,
settings) is made by a factory that closes over that state, and the
application registers the result at startup. The shipped `ask_user` is
built this way: `make_ask_user(ask)` returns the tool bound to *ask*, the
callable that puts a question to the human, and
`register_agent_tool(make_ask_user(my_prompt))` makes every script's
`ask_user` use yours (see [Tools and skills](tools-and-skills.md)). Your own version of the
pattern:

```python
def make_ticket_tool(db):
    def lookup_ticket(ticket_id: str) -> str:
        """Look up a support ticket by id and return its text."""
        return db.get(ticket_id)

    return register_agent_tool(lookup_ticket)
```

The package also ships a suite of generic tools (calculator, files, web,
notepad), and a chatterlang script can itself be a tool; both are
described in [Tools and skills](tools-and-skills.md).

## 2. A model

A model that supports tool calling, from any of the providers
under [Model providers](model-providers.md): Ollama, OpenAI, Anthropic,
or anything LiteLLM reaches. `source=` names the provider and `model=`
the model on it. The examples below use a local Ollama
(`source="ollama", model="llama3.2"`) because it needs no key; to use
another provider, install its extra, set its key, and change those two
strings — `source="openai", model="gpt-4o"` or
`source="anthropic", model="claude-haiku-4-5"` — and nothing else
changes. Setting `TALKPIPE_DEFAULT_MODEL_SOURCE` /
`TALKPIPE_DEFAULT_MODEL_NAME` (or the same keys in `~/.talkpipe.toml`)
lets scripts omit `source`/`model`, exactly as for talkpipe's
`llmPrompt`.

## 3. A script

From the shell, talkpipe's `chatterlang_script` runs a
script; `--load-module` imports your module first so its registrations
are in place when the script compiles:

```bash
chatterlang_script --load-module my_tools.py --script '
  INPUT FROM "How urgent is ticket T-1?"
  | llmAgent[source="ollama", model="llama3.2",
             tools=["lookup_ticket"], max_turns=4]
  | print'
```

Tool names in a script are quoted strings (`tools=["lookup_ticket"]`);
an unquoted `tools=[lookup_ticket]` is read as a variable reference and
fails with a bare `'lookup_ticket'` error. `INPUT FROM "…"` sends the
text as one prompt; the longer `INPUT FROM echo[data="…"]` splits its
data on commas unless `delimiter=` says otherwise, so a question with a
comma in it would become several.

From Python, import your module (so its tools are registered) and
compose the segment directly next to any other talkpipe segment:

```python
import my_tools  # noqa: F401  (registers lookup_ticket)
from talkpipe.pipe import io

from talkpipe_agents import LLMAgent

pipeline = io.echo(data="How urgent is ticket T-1?") | LLMAgent(
    source="ollama", model="llama3.2", tools=["lookup_ticket"]
)
print(list(pipeline()))
```

or compile the same chatterlang script instead of writing the pipeline out:

```python
import my_tools  # noqa: F401  (registers lookup_ticket)
from talkpipe.chatterlang import compiler

script = """
  INPUT FROM "How urgent is ticket T-1?"
  | llmAgent[source="ollama", model="llama3.2", tools=["lookup_ticket"]]
"""
print(list(compiler.compile(script)()))
```

In Python `tools=` also takes the functions themselves (`tools=[lookup_ticket]`),
so registration is only needed when a *script* has to name the tool.
The same three ingredients drive the interactive `AgentChat`: one
instance is one conversation, `ask()` sends a user message and returns
the reply, and a chat UI is a loop around it:

```python
import my_tools  # noqa: F401  (registers lookup_ticket)

from talkpipe_agents import AgentChat

chat = AgentChat(source="ollama", model="llama3.2", tools=["lookup_ticket"])
while (line := input("> ").strip()) not in ("", "quit"):
    print(chat.ask(line))
```

`chat.messages` is the conversation so far (Strands messages),
`chat.tool_names` the tools it was given, and starting over is a new
instance; `help(AgentChat)` lists every keyword argument.

## Seeing what the agent does

Every tool call is reported as one
line, `agent → lookup_ticket({'ticket_id': 'T-1'})`, and a tool call that
came back as an error as a second, `agent ← lookup_ticket: error: …`, at
INFO on the `talkpipe_agents` loggers — silent by default. Show them with
`chatterlang_script --logger_levels talkpipe_agents:INFO`,
`logging.basicConfig(level=logging.INFO)`, or by passing your own
`progress=` callable to `llmAgent`/`AgentChat` (a UI's status line).
After each item, `LLMAgent.last_result` / `AgentChat.last_result` is
the Strands `AgentResult`: `stop_reason` is `"end_turn"` when the agent
finished or `"limit_turns"` (etc.) when a budget cut it off. An answer
that *explains* how to do the task — often as a code snippet — with no
`agent → …` line means the model described the call instead of making
it; small models do that now and then. Ask again, or use a stronger
tool-calling model.

## Testing it

The package ships the fake model its own tests use, so the agent step
you just wrote can be tested offline — the real agent loop runs your
real tool against scripted model turns. See
[Testing without a model server](testing.md).
