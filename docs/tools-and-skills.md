# Tools and skills

*Part of the [talkpipe-agents](../README.md) documentation.
[Getting started](getting-started.md) walks through registering your
first tool.*

An agent gets two things from you beyond the model itself: **tools**,
functions it may call, and **skills**, written instructions it may
follow. Both are granted in the script, by name, and nothing is
available to the agent unless the script names it.

```
INPUT FROM "How many days until 2027-01-01?"
| llmAgent[source="ollama", model="gpt-oss",
           tools=["current_time", "days_between"], skills="./skills"]
| print
```

This page covers, in order: a [complete example](#a-complete-example)
you can run as-is, what a tool is, how `tools=[...]` finds one, the
[shipped tools](#shipped-tools) with a working example of each, what
happens when a tool pauses for a human, chatterlang
[scripts as tools](#scripts-as-tools), and [skills](#skills) — with a
complete skill to save and run.

The examples run against a local Ollama with `gpt-oss` pulled, as the
README's do; any other provider works by changing `source=` and `model=`
(see [Model providers](model-providers.md)).

## A complete example

Everything the example needs is in one file. The "database" is a
dictionary, so there is nothing to set up before running it. Save this
as `support_tools.py`:

```python
# support_tools.py
from talkpipe_agents import register_agent_tool

TICKETS = {
    "T-1": {"text": "Printer on fire, third floor", "status": "open"},
    "T-2": {"text": "Password reset for jdoe", "status": "closed"},
    "T-3": {"text": "Monitor flickers when the heater is on", "status": "open"},
}


@register_agent_tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket by its id (like "T-1") and return its text and status."""
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        return f"no ticket {ticket_id}"
    return f"{ticket_id} [{ticket['status']}]: {ticket['text']}"


@register_agent_tool
def list_open_tickets() -> str:
    """List every open support ticket, one per line as "id: text"."""
    return "\n".join(
        f"{tid}: {t['text']}" for tid, t in TICKETS.items() if t["status"] == "open"
    )
```

Run it from the directory holding that file. `--load-module` imports
the file first, so its two registrations are in place when the script
compiles, and `--logger_levels` shows each tool call as it happens:

```bash
chatterlang_script --logger_levels talkpipe_agents:INFO \
  --load-module support_tools.py --script '
  INPUT FROM "What is ticket T-1 about, and is it still open?"
  | llmAgent[source="ollama", model="gpt-oss",
             tools=["lookup_ticket", "list_open_tickets"]]
  | print'
```

```
2026-10-29 12:43:34,620 - INFO:talkpipe_agents.segments:agent → lookup_ticket({'ticket_id': 'T-1'})
Ticket **T‑1** is about a **printer on fire on the third floor**.
It is still marked as **open**.
```

The agent was given both tools and chose the one the question needed.
The rest of this page explains each part of that example; the same
module and script work from Python too, as shown in [Getting
started](getting-started.md#3-a-script).

## What a tool is

A tool is an ordinary Python function. Its docstring is the model's
description of it (what it does and when to call it) and its type hints
are its argument schema. Strands' `@strands.tool` decorator turns the
function into a tool object; `register_agent_tool` and `tools=[...]`
apply that decorator for you when handed a plain function, so you
rarely write it yourself.

Registering a tool, as `support_tools.py` does with `@register_agent_tool`,
gives it a name that scripts can put in `tools=[...]`.
`agent_tool_names()` lists what is registered at any point. A function
without a docstring is accepted with a warning: the model is told
nothing about it.

The example keeps its tickets in a module-level dictionary. A tool that
instead needs runtime state, such as a database handle or settings, is
made by a factory that binds that state, and the application registers
the result at startup:

```python
def make_ticket_tool(db):
    def lookup_ticket(ticket_id: str) -> str:
        """Look up a support ticket by id and return its text."""
        return db.get(ticket_id)

    return register_agent_tool(lookup_ticket)
```

## How a script names a tool

Each string in `tools=[...]` is resolved in two steps:

1. **A registered name.** Whatever `register_agent_tool` was given,
   including the shipped common tools.
2. **A Strands tool path**, with no registration at all. Strands' own
   loader accepts a module path to one tool
   (`"strands.vended_tools:http_request"`), a whole module
   (`"strands.vended_tools"`), or a file path to a Python file.

A string that is neither fails when the script is compiled, not in the
middle of a run.

Two cautions:

- **A whole module grants *every* tool in it.** `strands.vended_tools`
  alone brings `shell`, `python_repl`, `file_editor` and
  `handoff_to_user`. Name tools one by one unless that is what you mean.
- **Registration is last-wins.** An application may deliberately replace
  a shipped tool by registering another under the same name; every
  replacement is logged as a warning.

From Python, `tools=` also takes tool objects and plain functions
directly, mixed with names as you like.

## Shipped tools

Sixteen generic tools come with the layer, registered under their own
names, so a script grants any of them with `tools=[...]` and nothing
else. Each keeps deliberate bounds:

- file access is read-only;
- file and URL contents are clipped to a stated size, with an explicit
  truncation notice;
- `fetch_url` speaks only http(s);
- the calculator evaluates arithmetic through an AST walker rather than
  `eval`.

The file and web tools are talkpipe's own segments (`readFile`,
`listFiles`, `downloadURL | htmlToText`) compiled into callables with
`as_function`. They read what talkpipe reads and fetch as talkpipe
fetches (robots.txt, browser-like headers, retries); the tools add only
the agent-facing bounds above.

| Tool | What it does |
|------|--------------|
| `ask_user` | put one question to the human and return the answer (stdin by default; a UI binds its own prompt) |
| `think` | a no-op scratchpad: an explicit place to reason between tool calls |
| `calculator` | exact arithmetic (`+ - * / // % **`, math functions and constants) via a safe AST evaluator |
| `current_time` | the current date and time, optionally in a named IANA timezone |
| `days_between` | exact day count between two ISO dates |
| `regex_extract` | every match of a regular expression in a text, one per line |
| `read_file` | read a file as text — text, markdown, HTML, Word, PDF (with talkpipe's `pypdf` extra), CSV, JSON, JSONL, anything else as plain text (read-only, truncated with notice) |
| `list_files` | list a directory's entries, with sizes, filtered by glob |
| `search_files` | recursive grep — regex over files read as `read_file` reads them, `path:line: text` output |
| `fetch_url` | fetch an http(s) resource; HTML is reduced to its readable text |
| `save_note` / `read_notes` | a scratch pad kept in the agent's Strands `AgentState` — one pad per conversation in `AgentChat`, one per segment in `llmAgent` (carried across the items of a stream) |
| `summarize_notes` | a compact overview of the pad — counts, sizes, clipped previews — for surveying memory without reading it all |
| `update_note` / `delete_note` / `clear_notes` | revise or expunge memory: replace a note in place, delete one (numbers stay stable), or wipe the pad |

Three more names are registered alongside them, so they appear in
`agent_tool_names()` and in an unknown-tool error's list:
`enough_information`, `keep_finding` and `ignore_finding`. They belong
to [the findings loop](findings.md), which adds them itself.

Each tool below comes with a complete script that runs as shown with
the README's model (`ollama pull gpt-oss`): nothing to install, write
or register first. Save a script to a file and run it with
`chatterlang_script --script FILE` (the quotes inside rule out passing
it inline in a shell). The call and reply given with each example are
what the tool returns to the model; how the model words its final
answer is its own affair. The test suite runs every one of these
scripts as written, with a scripted model making exactly the call
shown, so the replies quoted here are the tools' actual output.

### Asking and thinking

**`ask_user`** puts one question to the human and returns the answer.

```
INPUT FROM "Book the meeting room for my next review."
| llmAgent[source="ollama", model="gpt-oss", tools=["ask_user"]]
| print
```

Run it in a terminal: the shipped tool prints the question and reads
one line back, so the model calls `ask_user(question="When is your next
review?")` and is told whatever you type, say `Tuesday at 10`. Without
a terminal (piped input, a service) it is told that the user is not
available and continues with what it has. A UI binds its own prompt
instead:

```python
from talkpipe_agents import make_ask_user, register_agent_tool

ask_user = make_ask_user(my_prompt)  # my_prompt(question: str) -> str
```

Pass the result in `tools=[...]`, or register it with
`register_agent_tool(ask_user)` so that every script naming
`"ask_user"` gets yours.

**`think`** is a no-op scratchpad. Calling it does nothing except give
the agent an explicit place to reason between tool calls, which helps a
model work through a problem step by step rather than answering at
once.

```
INPUT FROM "Is 1001 prime? Think it through before you answer."
| llmAgent[source="ollama", model="gpt-oss", tools=["think"]]
| print
```

The model calls `think(thought="1001 = 7 * 11 * 13, so it is not
prime")` and is told `(thought recorded — continue when ready)`.

### Arithmetic

**`calculator`** evaluates one arithmetic expression exactly, so the
model does not have to do sums itself.

```
INPUT FROM "What is 17 * 23 + 5?"
| llmAgent[source="ollama", model="gpt-oss", tools=["calculator"]]
| print
```

The model calls `calculator(expression="17 * 23 + 5")` and is told
`396`.

### Dates and times

**`current_time`** returns the current date and time, in the local
timezone or a named IANA one.

```
INPUT FROM "What day of the week is it in Tokyo right now?"
| llmAgent[source="ollama", model="gpt-oss", tools=["current_time"]]
| print
```

The model calls `current_time(timezone="Asia/Tokyo")` and is told an
ISO timestamp with the weekday, like
`2026-10-18T22:15:00+09:00 (Saturday)`.

**`days_between`** counts the days from one ISO date to another,
negative when the end comes first.

```
INPUT FROM "How many days are there from 2026-01-31 to 2026-03-01?"
| llmAgent[source="ollama", model="gpt-oss", tools=["days_between"]]
| print
```

The model calls `days_between(start="2026-01-31", end="2026-03-01")`
and is told `29`.

### Text

**`regex_extract`** returns every match of a regular expression in a
text, one per line (capture groups tab-separated).

```
INPUT FROM "Pull the email addresses out of: contact ann@example.com or bob@example.org"
| llmAgent[source="ollama", model="gpt-oss", tools=["regex_extract"]]
| print
```

The model calls `regex_extract(pattern="[\w.]+@[\w.]+", text="contact
ann@example.com or bob@example.org")` and is told
`ann@example.com` and `bob@example.org`, one per line.

### Files (read-only)

The three examples here read a `notes.txt` in the current directory;
make one with a couple of lines (a to-do list will do) before running
them.

**`read_file`** returns a file's text — Word, PDF, HTML and CSV by
their extracted text, anything else as plain text — truncated with a
notice when long.

```
INPUT FROM "Summarize notes.txt in one line."
| llmAgent[source="ollama", model="gpt-oss", tools=["read_file"]]
| print
```

The model calls `read_file(path="notes.txt")` and is told the file's
text.

**`list_files`** lists a directory, one entry per line, with each
file's size and a trailing `/` on subdirectories.

```
INPUT FROM "What files are in the current directory?"
| llmAgent[source="ollama", model="gpt-oss", tools=["list_files"]]
| print
```

The model calls `list_files(directory=".")` and is told one line per
entry, `notes.txt` with its size in bytes among them.

**`search_files`** greps under a directory, recursively, for a regular
expression and returns the matching lines as `path:line: text`.

```
INPUT FROM "Which files here mention the car registration?"
| llmAgent[source="ollama", model="gpt-oss", tools=["search_files"]]
| print
```

The model calls `search_files(pattern="car registration",
directory=".")` and is told the matching lines, such as
`notes.txt:2: Renew the car registration before the 30th.`

### Web

**`fetch_url`** fetches an http(s) resource and returns its text, with
HTML reduced to its readable content. It needs the network, nothing
else.

```
INPUT FROM "What does https://example.com/ say?"
| llmAgent[source="ollama", model="gpt-oss", tools=["fetch_url"]]
| print
```

The model calls `fetch_url(url="https://example.com/")` and is told the
page's readable text, which begins `Example Domain`.

### Notepad

The six notepad tools share one scratch pad, kept in the agent's
Strands state, so one script shows all of them. Notes are numbered
and the numbers stay stable: deleting a note never renumbers the rest.

```
INPUT FROM "Plan a two-day visit to Lisbon. Keep notes as you go, tidy them up, and finish with the plan."
| llmAgent[source="ollama", model="gpt-oss",
           tools=["save_note", "read_notes", "summarize_notes",
                  "update_note", "delete_note", "clear_notes"]]
| print
```

One run of that script, call by call:

- **`save_note`** stores a note and reports its number:
  `save_note(note="Day 1: Alfama and the castle")` is told
  `(note 1 saved)`, and a second note is `(note 2 saved)`.
- **`summarize_notes`** gives a compact overview, each note clipped to
  a preview: `summarize_notes()` is told `2 notes, 40 characters in
  total:` followed by one line per note.
- **`update_note`** replaces a note in place:
  `update_note(number=2, note="Day 2: Belém, then Sintra")` is told
  `(note 2 updated)`.
- **`read_notes`** returns every note in full, numbered:
  `read_notes()` is told `1. Day 1: Alfama and the castle` and
  `2. Day 2: Belém, then Sintra`, one per line.
- **`delete_note`** removes one note for good: `delete_note(number=1)`
  is told `(note 1 deleted)`, and note 2 keeps its number.
- **`clear_notes`** empties the pad: `clear_notes()` is told
  `(1 notes deleted)`.

## Tools that pause for a human

Some tools stop the agent and wait for a person, as Strands'
`handoff_to_user` does. This is a Strands *interrupt*, and it ends the
current item or exchange:

- **`llmAgent`** reports the pause on the progress channel and yields
  whatever text the agent had produced so far.
- **`AgentChat`** returns a notice naming the tool. The next `ask()`
  resumes the agent with that message as the human's answer. A host
  that wants to resume some other way finds the request in
  `last_result.interrupts`.

## Scripts as tools

Tools do not have to be Python. A chatterlang script, with a
description for the model, can itself be a tool. A single script can
therefore carry its own short tooling scripts and hand them to the
agent:

```
scriptTool[
    name="word_count",
    description="Count the words in the given text and return the number.",
    script="| lambda[expression='len(item.split())']"];

INPUT FROM prompt
| llmAgent[source="ollama", model="gpt-oss", tools=["word_count", "think"]]
| print
```

Save that as `word_count.tp` and run it with
`chatterlang_script --script word_count.tp`. `--script` takes a file path
as well as inline script text, and a script like this one cannot be
passed inline in a shell because of the quotes inside it. `INPUT FROM
prompt` reads questions one line at a time, from an interactive terminal
or from piped input (`echo "How many words are in: the quick brown fox" |
chatterlang_script --script word_count.tp`, which warns that its input is
not a terminal); swap in `INPUT FROM "…"` for a fixed one. Any
tool-calling model works here; the README's `gpt-oss` reasons before
answering, which suits a `think` tool.

How a script tool behaves:

- **It is registered at compile time.** Segments are constructed in
  statement order, so a `scriptTool` statement makes its name available
  to every later `llmAgent` in the same script. A broken tool script
  fails at compile time, before anything runs, just like an unknown tool
  name.
- **The model sees `description`** (and the optional
  `input_description`) as the tool's documentation.
- **It takes one string argument**, fed to the script as its input
  stream. A script that opens with its own `INPUT FROM` ignores it.
- **It returns the script's output items**, joined with newlines.
- **Each call runs a fresh instance** of the script, so stateful
  segments and `@variables` start clean and concurrent calls share
  nothing.

From Python, the same thing is one call,
`register_agent_tool(make_script_tool(name, description, script))`, both
imported from `talkpipe_agents`.

`findingTool` takes the same parameters but treats each item the script
emits as a *finding*. With `llmAgent[judge=True]`, the agent judges
those findings and accumulates them; see
[The findings loop](findings.md).

The placement story from the README carries over: the tool's body is
deterministic, reviewable dataflow sitting next to the pipeline that
grants it, and the agent's license is exactly the set of scripts you
wrote down.

## Skills

Instructions do not have to be Python either. An [agent
skill](https://agentskills.io) is a directory holding a `SKILL.md`:
YAML frontmatter with the skill's `name` and `description`, then the
instructions as markdown. The agent sees only the name and description
until it decides the skill applies, so the description should say
*when* to use it as well as what it covers.

### A complete skill

This skill teaches the agent one way to triage a support ticket. Save
it as `skills/ticket-triage/SKILL.md`, in the directory you will run the
script from. The skill directory must be named after the skill — the
`name` in the frontmatter is `ticket-triage`, so the directory is
`ticket-triage` — and `skills/` is the folder that `skills="./skills"`
will name:

```markdown
---
name: ticket-triage
description: How to rate the urgency of a support ticket. Use whenever asked how urgent, serious, or important a ticket is.
---

# Ticket triage

Rate a ticket at exactly one of these levels, then give one sentence of reasoning.

- **P1** — a danger to people or property (fire, smoke, flooding, a sparking outlet), or the whole office cannot work.
- **P2** — one team or one shared device is blocked, with no workaround.
- **P3** — one person is inconvenienced but can still work.
- **P4** — a question, a request, or anything already resolved.

Answer in the form `P<n>: <reason>`, nothing else.
```

The layout is then:

```
skills/
└── ticket-triage/
    └── SKILL.md
```

Nothing else is needed. Run it, from the directory containing `skills/`:

```bash
chatterlang_script --logger_levels talkpipe_agents:INFO --script '
  INPUT FROM "How urgent is this ticket: Printer on fire, third floor"
  | llmAgent[source="ollama", model="gpt-oss", skills="./skills"]
  | print'
```

```
2026-10-29 12:43:36,024 - INFO:talkpipe_agents.segments:agent → skills({'skill_name': 'ticket-triage'})
P1: The printer is on fire on the third floor, posing a direct danger to
people and property.
```

The `agent → skills(...)` line is the agent loading the skill; without
the skill the model would answer in its own words, with no `P1`. A
folder may hold many skills, each in its own subdirectory, and the
agent loads only the ones it decides apply:

```
skills/
├── ticket-triage/
│   └── SKILL.md
└── release-notes/
    ├── SKILL.md
    └── references/style-guide.md
```

Tools and skills combine in the obvious way — `tools=["lookup_ticket"],
skills="./skills"` with the module from the [complete
example](#a-complete-example) lets the agent look a ticket up and then
triage it — but a small model asked to do both may do only one (look up
the ticket and skip the skill, or load the skill and guess at the
ticket). Prefer a stronger model for a task that needs both.

### How skills work

`skills=` on `llmAgent` and `AgentChat` hands skills to Strands'
`AgentSkills` plugin. It accepts any of these, singly or as a list:

- a folder whose subdirectories are skills (`"./skills"` above);
- a single skill directory (`"./skills/ticket-triage"`);
- a `SKILL.md` path;
- an `https://` URL to raw `SKILL.md` content.

Paths are resolved relative to the directory the command runs from, as
file paths in scripts usually are.

Strands does the rest. Every skill's name and description go into the
system prompt as an `<available_skills>` block, and the agent gets one
extra tool, `skills`, that loads a skill's full instructions when the
agent decides it applies. A folder of twenty skills therefore costs a
line each until one is used.

Things to know:

- **Resource files are only named.** Loading a skill lists its
  `scripts/`, `references/` and `assets/` files but does not read them.
  Give the agent `read_file` (or another file tool) if it is meant to
  read them.
- **A missing path fails at compile time**, like an unknown tool name.
- **A malformed `SKILL.md`** inside a folder is logged by Strands and
  skipped: the frontmatter must open and close with `---` lines and
  carry a non-empty `name` and `description`. A `name` that is not
  lowercase letters, digits and single hyphens, or that differs from
  its directory's name, is only warned about; the skill still loads.
- **Activated skills are recorded** in the agent's state under
  `agent_skills`, the same state the notepad lives in.
