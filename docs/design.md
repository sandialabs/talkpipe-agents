# Design

*Part of the [talkpipe-agents](../README.md) documentation. The README's
"Why" and "Agents as a segment" sections are the starting point; this
page is how the package carries them out.*

## Structure

The package is small and each module has one role:

| Module        | Role |
|---------------|------|
| `segments.py` | `llmAgent` — the talkpipe segment that runs a tool-using agent as one pipeline step |
| `chat.py`     | `AgentChat` — the same agent as a multi-turn conversation, for interactive UIs |
| `models.py`   | source/model string resolution → Strands `Model` objects, via a pluggable `register_agent_model` registry |
| `ollama_model.py` | `ThinkingOllamaModel` — Strands' `OllamaModel` for the `ollama` source, keeping a thinking model's reasoning between turns and sending each turn back as one message with named tool results |
| `tools.py`    | a name → tool registry (`register_agent_tool`) so scripts can reference tools by name, plus the generic `ask_user` and `think` tools |
| `script_tools.py` | chatterlang scripts as tools — `make_script_tool` and the `scriptTool` segment, so a script can define documented tooling scripts inline; `make_finding_tool` and `findingTool` for finding tools |
| `common_tools.py` | a suite of generally useful, bounded tools registered by name — see [Tools and skills](tools-and-skills.md) |
| `findings.py` | the findings loop — `Finding`, `finding_tool`, `FindingsSession`, the `FindingsJudge` (also the `judgeFindings` segment), the `JudgeFindings` hook that judges a finding tool's results before the model sees them, and the `enough_information` / `keep_finding` / `ignore_finding` session tools — see [The findings loop](findings.md) |
| `hooks.py`    | `AgentHooks` — the Strands hook provider both entry points install: per-tool-call progress lines, and scrubbing of leaked tool-call syntax as Strands stores each message |
| `replies.py`  | `ask_and_clean` — one prompt under a Strands `limits` budget, with the empty-reply retry — and `agent_limits` |
| `support.py`  | the shared one-line-progress convention (`teller`) and reply cleanup primitives (`strip_think`, `strip_tool_leak`) |
| `testing.py`  | `ScriptedModel` and `scripted_source` — a Strands `Model` that replays scripted tool calls and answers and records what each tool call returned (`tool_results`), so applications test their agent wiring offline; the `scripted_model` pytest fixture is installed as a plugin; `ScriptedJudge`, a findings judge that decides by label |

Two registries make the layer generic rather than bound to any one
provider or application: the model registry, described under
[Model providers](model-providers.md), and the tool registry, described
under [Tools and skills](tools-and-skills.md).

## Agents as a conversation

For interactive use, `AgentChat` is the thin Python-side counterpart
to the `llmAgent` segment: one instance per conversation, `ask()` per
user message, the same model resolution, tool registry, limits, and
reply hygiene. Long conversations can opt into LLM-summarized
compaction (`summarizing=True`). A chat UI is just a loop around it.

## Reply hygiene

Both entry points share `hooks.py` and `replies.py`, which exist
because real models misbehave in recurring ways: leaked raw tool-call
syntax is scrubbed from each assistant message as Strands stores it
(so it neither reaches the reply nor derails later turns) — both the
harmony format of the gpt-oss family and a bare
`{"name": …, "parameters": {…}}` object written as prose, as the llama
family does when it means to call a tool but bypasses the provider's
tool-call channel; reasoning-model `<think>` blocks are removed; and a
reply with no user-visible text at all — including one that was nothing
but a leak — is retried once with a nudge to answer in plain prose
rather than surfacing as silence.

## Where Strands fits

[Strands Agents](https://strandsagents.com) supplies the agent loop
itself — model streaming, tool schema generation and dispatch,
conversation state, hooks. This library does not reimplement any of
that; it *places* it:

- A `strands.Agent` lives inside the segment (or `AgentChat`), one per
  item or one per conversation, never shared implicitly.
- The model registry exists because a `strands.Agent` needs a Strands
  `Model` object: talkpipe's prompt adapters are text-in/text-out and
  cannot stream tool calls, so the registry parallels talkpipe's
  adapter registry while yielding what Strands needs — keeping
  talkpipe's configuration semantics without forking its plumbing.
- Tools are plain Strands tools, so anything written for Strands drops
  in unchanged, and tools written here work in any Strands agent.
- Strands hooks (`AgentHooks`) drive per-tool-call progress lines
  (`agent → toolName(args…)`) through the same one-line-message
  convention the surrounding pipelines use, and scrub leaked tool-call
  syntax on `MessageAddedEvent`, so the stored conversation is clean
  the moment Strands writes it rather than repaired afterwards.
- Effort caps are Strands' own `limits` (`turns`, `output_tokens`,
  `total_tokens`), surfaced as segment parameters; the `AgentResult`
  of the last prompt is kept as `last_result`, so a host reads
  `stop_reason` and metrics straight from Strands.
- Tools that keep state (the notepad) keep it in the agent's Strands
  `AgentState` through `ToolContext` — per agent, JSON-serializable,
  persisted along with the agent by any Strands session manager — so
  there is no pad factory and no process-global state.
- Tool strings that are not registered names go to Strands' own
  loader, so any Strands tool is reachable from a script by path.

In short: Strands is the engine, talkpipe is the map, and this library
is the mounting bracket between them.

## Results

The layer is in production use in the host application it was
extracted from:

- Chatterlang scripts run `llmAgent` steps with application tools —
  semantic news search, stored quotes, learned principles, web
  research — registered by name from a tool-pack factory.
- The application's interactive chat is built on `AgentChat` handed
  that same tool pack wholesale: new tools become available to every
  consumer automatically, and the agent decides per question which
  (if any) to call, replacing a hand-routed retrieval step.
- The reply-hygiene layer came from failure modes observed with real
  local models (mangled tool-call JSON leaking into answers, empty
  replies consisting only of reasoning) and turned them from
  user-visible glitches into logged, recovered events.
- The whole layer is tested fully offline: a scripted Strands `Model`
  registered as just another model source replays planned tool calls
  and answers, exercising the segment, the chat, turn caps, tool-error
  recovery, and hygiene without a live model server. That model ships
  in the package (`talkpipe_agents.testing`) so applications can test
  their own agent steps the same way.

## Provenance

This library was extracted from the staging area of the host
application it was proven in (nothing in the layer ever imported from
the host). The host currently still carries its own bundled copy —
**do not install this plugin into an environment that also registers
that copy**, or the `llmAgent` registration will collide; the host's
migration to depend on this package is a separate, future change.
