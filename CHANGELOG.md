# Changelog

All notable changes to talkpipe-agents are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- The findings loop, for agents that gather information before
  answering. A *finding tool* (`@finding_tool` in Python, or the
  `findingTool` chatterlang segment, which turns a script into one)
  returns findings: `Finding` objects, strings, dicts or search
  results. With `llmAgent[judge=True]`, each item gets a
  `FindingsSession` for its question. The agent's own model judges
  every finding tool call's results in one batched call before the
  agent sees them, against the question, the call's `looking_for`
  and what is already known. Relevant findings are kept and the rest
  ignored. The agent then reads the kept findings in full, the
  ignored ones with reasons, and a turn countdown. It stops by
  calling `enough_information`, which is refused while nothing is
  kept, and answers from what it kept. The run's `FindingsOutcome`
  (kept findings, how the run ended, the trace) is `last_outcome`,
  or set on the item with `findings_as`. `findings=True` runs the
  session without a judge, and the agent decides with `keep_finding`
  and `ignore_finding`. `max_kept` caps how many findings a run
  keeps, and `question_field` sets the question findings are judged
  against. The parts also work on their own: `FindingsSession`, the
  `FindingsJudge` (also the `judgeFindings` segment), the
  `JudgeFindings` Strands hook, and a lenient `parse_judgements` that
  reads JSON, fenced JSON or prose decisions.
  `talkpipe_agents.testing.ScriptedJudge` decides by label, so the
  whole loop can be tested offline. A judge keeps every finding when
  its model call fails or its reply holds no decisions (the error
  quotes the start of that reply), but raises when its model cannot be
  built at all (no model configured, an unknown source, a missing
  provider extra), since every judgement would fail the same way. The
  agent is shown each finding's source next to its label when the two
  differ, so it has the source to cite.
- `hooks` on `llmAgent`: more Strands hook providers to install on
  every agent it builds.

### Fixed

- `llmAgent` with `set_as` (or, with the findings loop on,
  `findings_as`) given a plain string item now stops before the model
  runs, with an error saying the items must be dicts and how to make
  them in chatterlang. It used to run the agent and then fail with
  `'str' object has no attribute ...`.
- Ollama models that reason between tool calls no longer forget their
  plan every turn. Strands' Ollama provider discarded the model's
  `thinking`, so a model such as minimax-m3.1-flash that decides in its
  reasoning which results to keep searched again on its next turn,
  often until it hit the turn limit. The `ollama` source now stores
  the reasoning on the assistant message and sends it back as that
  message's `thinking`. It also sends a turn's text and all its tool
  calls as one assistant message instead of one message per call, and
  names each tool result with `tool_name`, as Ollama's chat API expects.
  Models that do not think are unaffected by the replay, but the
  message threading changes for every model. `model_params='{"replay_thinking": false}'`
  turns the replay off for a model that reacts badly to it.
- `list_files` and `search_files` now tell the model that their
  `directory` defaults to `"."`, the current directory. Their
  descriptions mentioned only that `"~"` is expanded, and gpt-oss
  took that as a hint and listed or searched the home directory when
  asked about the current one.
- An `llmAgent` item that its turn or token limit cuts off before any
  answer no longer passes unnoticed. The segment still yields the
  empty text, but now also logs a warning naming the limit, unless a
  `progress=` callable was given (the host already shows the limit
  line then). Without logging configured, a capped item used to show
  up only as a blank line.

### Documentation

- The examples use `gpt-oss` on Ollama instead of `llama3.2` and
  `qwen3`. The findings loop example asks about "my notes" so the
  agent searches them first, and the `search_files` example asks
  about the car registration, a phrase the file contains.
- The findings loop page's examples run as written: the complete
  example keeps its agent run under a `__main__` guard, so the opening
  chatterlang script (via `--load-module`) and the offline test (which
  now imports it) can use its `search_notes`; the `judgeFindings`
  snippet names its model. The page also shows a dict item with
  `findings_as`, a bare Strands agent with `JudgeFindings` and the
  `FINDINGS_PROTOCOL` system prompt it needs, which of `judge` and
  `findings` wins when both are set, that `findings=True` spends a
  turn per decision, and that the loop and `hooks` are `llmAgent`-only.
- Smaller corrections: the unquoted-tool-name error quoted in Getting
  started is the current one; the tools page shows the progress lines
  with their timestamp, says `INPUT FROM prompt` reads piped input
  too, and names the findings loop's three registered session tools;
  the providers page explains the `reasoningContent` warning the
  OpenAI and LiteLLM sources log for reasoning models.
- The findings loop page's RAG example now puts the kept findings in
  the `_background` field, the one `AppendRAGSources` reads, so the
  answer gets its `Sources:` list; it used a field `AppendRAGSources`
  never looks at, and the sources were silently left off. The example
  also says it continues the complete example's `__main__` block. The
  page explains the empty source the complete example prints for its
  calculation (a plain-string finding has none), says that
  `findingTool` items and `judgeFindings`' findings field take the
  same shapes a finding tool may return, so a script can label and
  source its findings, and that the loop has nothing to judge when no
  finding tool is among the tools.
- The custom provider example on the providers page passes the
  script's `model_params` (`params=`) on to its model. It used to
  drop them silently.

## 0.1.1 (2026-10-23)

### Fixed

- A tool call the model server could not parse no longer ends the agent
  run. Ollama rejects a turn whose tool-call JSON is malformed (gpt-oss
  sometimes stops mid-object) with `error parsing tool call`; `llmAgent`
  and `AgentChat` now retry that model call with Strands' throttling
  backoff, as they already did for throttling. Other model errors still
  raise at once.

## 0.1.0 (2026-10-21)

### Added

- `skills` on `llmAgent` and `AgentChat`: an agent skills folder (or a
  single skill directory, a `SKILL.md` path, or an `https://` URL, singly
  or as a list), handed to Strands' `AgentSkills` plugin. The skills are
  listed in the system prompt and the agent gets a `skills` tool that
  loads one's full instructions on demand. A path that does not exist
  fails at compile time, like an unknown tool name.
- OpenAI agent model sources named for the API they call:
  `openai_responses` (OpenAI's Responses API, `/v1/responses`, through
  Strands' `OpenAIResponsesModel`; needs `openai>=2.0`) and `openai_chat`
  (Chat Completions, `/v1/chat/completions`, through Strands'
  `OpenAIModel`, for servers that imitate OpenAI's API: vLLM, LM Studio,
  gateways, …). Both come with the `openai` extra.
- `model_params` on `llmAgent` and `AgentChat` (`params=` on
  `build_agent_model`): provider request parameters passed through to the
  model, such as `{"reasoning_effort": "none"}` or `{"reasoning":
  {"effort": "low"}}`. chatterlang has no dict literal, so scripts write
  it as JSON text; malformed JSON fails when the script compiles. The
  Ollama source takes them as its `options`.
- `talkpipe_agents.testing`: the scripted Strands `Model` the package's
  own tests use is now public, so applications can test their agent
  steps offline without copying it. `ScriptedModel` replays turns built
  with `.call(tool, **args)` and `.say(text)` through the real agent
  loop and records `calls`/`prompts`; `scripted_source()` registers one
  as agent model source `"scripted"` for a block (or under any other
  source name, so a script that says `source="ollama"` can be tested as
  written). A `scripted_model` pytest fixture doing the same per test is
  installed as a pytest plugin (`pytest11` entry point) with the package;
  `pytest -p no:talkpipe_agents` turns it off.
- `ScriptedModel.tool_results` records what each scripted tool call came
  back with (`ToolResult(name, status, text)`), so a test can assert the
  real tool ran; `remaining` is the script not yet played. Two mistakes
  that the following scripted answer would otherwise mask are surfaced: a
  scripted call to a tool the agent does not have fails the turn with a
  message naming the tools it has, and a call the agent reports as an
  error (wrong argument names, a tool that raised) is emitted as a
  `ScriptedToolErrorWarning`. Running out of script now says which tool
  had just returned.
- A "Getting started" walkthrough in the README for application
  developers: registering your own tool, choosing a model (including
  `TALKPIPE_OLLAMA_SERVER_URL` for a non-local Ollama), running a script
  with `chatterlang_script --load-module`, composing `LLMAgent` in a
  Python pipeline, seeing progress lines, and testing offline with a
  scripted model.
- A "Model providers" section in the README: the built-in agent model
  sources (`ollama`, `openai`, `anthropic`, `litellm`) with the pip
  extra, credentials, and model naming each needs, how the configured
  defaults select one, why talkpipe's `eliza` is not an agent source, and
  how to register any other Strands provider. The Getting started
  walkthrough and the `llmAgent`/`AgentChat` parameter documentation now
  list every source instead of naming only Ollama.
- A "First run" section at the top of the README: one command that runs a
  tool-using agent with a shipped tool, no registration and no API key,
  plus what to do when nothing is listening on the Ollama endpoint.
- `register_agent_tool` works as a decorator with arguments
  (`@register_agent_tool(name="...")`), not only bare.
- `UnknownAgentToolError`, raised for a tool string that neither the
  registry nor Strands' loader can resolve. It is still a `ValueError`
  for Python callers and now also a chatterlang `CompileError`, so
  `chatterlang_script` reports a misspelled tool name as a one-line
  compile error instead of a traceback.
- The notepad tools (`save_note`, `read_notes`, `summarize_notes`,
  `update_note`, `delete_note`, `clear_notes`) are exported from the
  package top level alongside the other common tools.
- Parameter documentation on `AgentChat`, and docstrings for
  `agent_tool_names` and `agent_model_sources`.
- CI publishes releases. A pushed `vX.Y.Z` tag uploads the sdist and
  wheel to the hosting server's PyPI package registry (`publish-registry`;
  switch it off with the `PUBLISH_REGISTRY=false` Actions variable), and a
  published GitHub release uploads them to PyPI (`publish-package`, with
  the `PYPI_API_TOKEN` secret; version tags only). A `package` job builds
  the distributions on every run, `twine check`s them, installs the wheel
  into a clean environment, and checks that importing talkpipe registers
  `llmAgent` and `scriptTool` and that every entry point resolves, so a
  broken wheel fails the build before release time. RELEASING.md
  describes both publish paths, says which refs to push to GitHub at
  release time (nothing mirrors between the hosts), and has a one-time
  GitHub setup section: an empty repository, `stable` as its default
  branch, the `PYPI_API_TOKEN` secret (account-wide for the first upload,
  since PyPI cannot scope a token to a project that does not exist yet) and
  the optional `SAFETY_API_KEY`.
- Release metadata, matching talkpipe's: the package is licensed under the
  Apache License 2.0 (LICENSE and COPYRIGHT files, shipped in the
  distributions; the earlier `pyproject.toml` said MIT with no license
  file), and `pyproject.toml` now carries authors, keywords, classifiers,
  and the project URLs (homepage, repository, documentation, changelog, bug
  tracker).
- The GitHub-side files the other suite repositories carry: a Dependabot
  configuration (pip and GitHub Actions, weekly, against `main`), a
  security policy, and a release-checklist issue template.

### Fixed

- A tool that raises a Strands interrupt (human input required, as
  `handoff_to_user` does) no longer leaks the raw interrupt list
  (`[{'id': …, 'name': 'strands:handoff-to-user', …}]`) downstream as the
  answer. `llmAgent` yields whatever answer text there was, reports
  `agent: paused for human input by <tool> (<reason>)` on the progress
  channel, and skips the empty-reply retry; `AgentChat.ask` returns a
  visible notice naming the tool and reason. `last_result.stop_reason` is
  `"interrupt"` and `last_result.interrupts` holds the request for a host
  that resumes it.
- The error for a Strands tool path that does not resolve
  (`tools=["strands.vended_tools:no_such"]`) now includes the loader's own
  reason (`Tool no_such not found in module strands.vended_tools`) instead
  of only listing the registered names and suggesting a path.
- The sdist no longer ships the README's 6.5 MB illustration or `uv.lock`
  (it was 7 MB for a 40 KB package).
- The common tools accept `null` for their optional arguments. Models
  (llama3.2 among them) send `null` for optional parameters they leave
  unset, and Strands validates arguments against the type hints, so
  `fetch_url`, `read_file`, `list_files`, `current_time`, `regex_extract`
  and `search_files` rejected such calls before running — the agent's
  first web or file task failed for a schema reason and the model talked
  around it. `None` now means the parameter's default.
- `llmAgent` with a `field=` that the item does not have raises
  `field='…' not found on the item` instead of prompting the model with
  the text `None` and letting a made-up answer flow downstream.
- `AgentChat` is usable again after a tool's interrupt. The next `ask()`
  resumes the paused agent with that message as the human's answer
  (which is what `handoff_to_user` and the like are waiting for) instead
  of raising Strands' `must resume from interrupt with list of
  interruptResponse's` on every later message; the notice returned for
  the interrupt says so.
- A refused connection from the `openai`, `openai_responses` and
  `openai_chat` sources names the endpoint the client tried
  (`client_args`' `base_url`, else `OPENAI_BASE_URL`, else OpenAI) and
  the variable that moves it, instead of the bare `Connection error.`.
- An Ollama model that cannot call tools (`… does not support tools
  (status code: 400)`) is told apart from other errors: the message now
  says the agent needs a tool-calling model and how to pull one.
- `max_turns`, `max_output_tokens` and `max_total_tokens` are validated
  where they are read, so a value that is not a positive integer (a
  string from a script, `0`, a negative) fails when the script compiles
  with the parameter's own name — `max_turns must be a positive integer
  or None, got 'four'` — rather than at run time as Strands'
  `limits['turns'] must be a positive int`. `0` used to mean no cap
  silently; `None` is the documented way to say that.
- A `tools=` entry that is neither a string, a callable, a tool object,
  a module nor a Strands tool dict (a stray number, `None`) raises
  `TypeError` at construction instead of being dropped by Strands at run
  time with a log line, which left the agent with fewer tools than
  written.
- `AgentChat.ask()` given something other than a string raises
  `TypeError` naming the argument, not `'int' object has no attribute
  'strip'`.
- `PackageNotFoundError` and `version` from `importlib.metadata` no
  longer leak into the `talkpipe_agents` namespace.

### Changed

- The tools documentation is now `docs/tools-and-skills.md`, "Tools and
  skills". Its opening paragraph, which had grown to cover the tool
  model, name resolution, module-wide grants, interrupts, factories and
  the two shipped tools in one block, is split into short sections with
  one topic each, and the common-tools bounds, script-tool behaviour and
  skills caveats are lists rather than prose. The page now opens with a
  complete, runnable example — a tool module whose "database" is a
  dictionary, the command that runs it, and the output — and the skills
  section gives a complete `SKILL.md`, says where to save it, and shows
  the run; neither needs anything set up beyond a model.
- The tools documentation's `ask_user`/`think` section and its
  common-tools section, which listed the rest of the shipped tools
  further down the page, are merged into one "Shipped tools" section:
  the overview table now names all sixteen tools, and each tool comes
  with a complete script that runs as shown with the README's model,
  nothing extra to install or register. The test suite runs every one
  of those scripts as written against the scripted model, making the
  call the docs describe, and checks the reply the docs quote.
- **Breaking:** agent model source `openai` is now an alias of
  `openai_responses` instead of Chat Completions. OpenAI's reasoning
  models reject function tools on Chat Completions (`Function tools with
  reasoning_effort are not supported … in /v1/chat/completions`), so the
  old `openai` source could not run them as agents. Scripts that point
  `openai` at another server through `OPENAI_BASE_URL` should use
  `source="openai_chat"`; few such servers implement the Responses API.
- Tool calls that come back as errors are reported on the progress channel
  (`agent ← fetch_url: error: …`, next to the existing `agent → …` line),
  so a bad answer caused by a failed tool is visible in the log or UI
  rather than only in the model's prose.
- Ollama errors say where the client looked. A refused connection reads
  `All connection attempts failed (nothing is listening at
  http://localhost:11434: start Ollama there, or set
  TALKPIPE_OLLAMA_SERVER_URL to the server to use)`, and a 404 for a model
  names the server that lacks it and the `ollama pull` command; the
  exception types are unchanged.
- Registering (or passing in `tools=[...]`) a plain function with no
  docstring logs a warning: the docstring is the model's only description
  of the tool.
- `AgentChat.ask` has a docstring, and `llmAgent`'s `progress` parameter is
  described in the chatterlang component reference like the others.
- Documentation: the examples write a single prompt as `INPUT FROM "…"`
  rather than `INPUT FROM echo[data="…"]`, and say why: `echo` splits
  its data on commas unless `delimiter=` says otherwise, so a question
  with a comma in it became several prompts (the README's claim that the
  two forms are equivalent was wrong in exactly that case). The README
  shows `echo[data="first|second", delimiter="|"]` as the way to feed
  several items, which `multi_turn` and the notepad need. testing.md
  shows `ScriptedModel` + `scripted_source` outside pytest and says
  where the example test file lives; tools.md and model-providers.md
  name `agent_tool_names()` / `agent_model_sources()`; the providers
  page notes that current Ollama serves both OpenAI APIs on its `/v1`.
  `help(LLMAgent)` now describes `last_result`, and `AgentChat.messages`
  / `tool_names` have docstrings.
- Documentation: naming a whole module in `tools=` grants every tool in it
  (`strands.vended_tools` includes `shell` and `python_repl`), and what an
  interrupting tool does to an item; LiteLLM reads its own provider
  variables (`OLLAMA_API_BASE` for `ollama_chat/…` — `TALKPIPE_OLLAMA_SERVER_URL`
  is for the `ollama` source only); an answer that explains the task as
  code with no `agent → …` line means the model described the tool call
  instead of making it; and why the scripts-as-tools example uses `qwen3`.
- The README notes under the pip install where the from-source install is
  described, explains that `INPUT FROM "…"` is shorthand for `echo` (the
  prose described the hello world as using `echo`, which it does not
  show), and names the hinted Ollama errors. Getting started says tool
  names in a script are quoted strings; the "test a script as written"
  example in the testing page is complete and runnable rather than ending
  in `...`.
- The README is reorganized for a first-time reader. It opens with the
  "Why" — the agent as agent-oriented engineering has long defined it
  (an object with an attitude: a component that is handed a goal and
  chooses its own actions, so it acts probabilistically), agents as an
  engineering design pattern that can be the foundation of an
  application or one bounded step in a larger workflow, and why this
  package is built for the second placement —
  then a `pip install talkpipe-agents` hello world, then "Agents as a
  segment". The technical material moved to pages under `docs/`:
  `getting-started.md` (your tools, in your pipeline), `testing.md`
  (the scripted model), `model-providers.md`, `tools.md` (the registry,
  the common tools, scripts as tools), and `design.md` (structure,
  reply hygiene, where Strands fits, results, provenance). Installing
  from a clone and the provider extras from a checkout are described
  under "Development".
- The file and web tools in the common suite are talkpipe segments
  compiled into callables (`as_function`) instead of code of their own.
  `read_file` is `readFile`: it now understands every format talkpipe has
  an extractor for (text, markdown, HTML reduced to readable text, Word
  `.docx`, PDF with the `pypdf` extra, CSV, JSON and JSONL) and reads any
  other extension as plain text; a file that cannot be read raises
  rather than returning garbled text. `list_files` is `listFiles`.
  `search_files` lists with `listFiles` and reads each file as
  `read_file` does, so Word, PDF and CSV documents are searched by their
  text and its line numbers match `read_file`'s output; files that
  cannot be read are skipped as before. `fetch_url` is `downloadURL |
  htmlToText`: it respects robots.txt, sends browser-like headers (the
  User-Agent follows talkpipe's `user_agent` config key), retries
  transient failures, and reduces HTML to its readable main content
  while leaving plain text, JSON and markdown untouched; the private
  HTML text extractor is gone.
- The `dev` extra moves ruff to the 0.17 series (`>=0.17.0,<0.18`), with
  the ruff-pre-commit hook on v0.17.0 to match. The code needed no changes
  for the new release's rules or formatting.
- Requires talkpipe 1.1.0 or later. `build_agent_model` (and so
  `llmAgent` / `AgentChat`) fills in a missing `source` / `model` with
  talkpipe's own `resolve_source_and_model` instead of a copy of that
  logic, so the fallback cannot drift from `llmPrompt`'s. When neither
  the arguments nor the configuration supply a value, the error is now
  talkpipe's "Model name and source must be provided..." message rather
  than this package's own wording.
- `tools=[...]` on `llmAgent` and `AgentChat` wraps a plain function as
  a Strands tool, the same as `register_agent_tool` does. Previously a
  bare function passed directly was dropped by the agent with only a
  log line, leaving it tool-less.
- `tools="name"` (a single string) is taken as one tool, instead of
  being iterated into `unknown agent tool 'n'`.
- Re-registering a tool name logs a warning that the previous tool was
  replaced (registration remains last-wins so an application can
  deliberately override a shipped tool).
- An unknown bare tool name now asks whether the module that registers
  it was imported (`chatterlang_script --load-module`), and no longer
  chains Strands' attempt to import a module of that name; path-like
  tool strings keep the loader's error as the cause.
- The missing source/model error (talkpipe's "Model name and source must
  be provided...") goes on to name the fix: pass `source=` and `model=` to
  the agent, or set `TALKPIPE_DEFAULT_MODEL_SOURCE` /
  `TALKPIPE_DEFAULT_MODEL_NAME` (or the `~/.talkpipe.toml` keys) before
  the process starts — talkpipe reads its configuration once per process,
  a trap for tests that patch the environment.
- Documentation: Tools and Getting started say how to bind `ask_user` to
  your own prompt (`make_ask_user(my_prompt)`, passed in `tools=[...]` or
  registered so scripts get it) — Getting started had presented
  `make_ask_user` as a factory that registers, which it does not — and
  Getting started shows an `AgentChat` loop with the attributes a chat UI
  uses (`messages`, `tool_names`, a new instance to start over). Tools
  gives the import for `make_script_tool` and notes that the
  `INPUT FROM prompt` example needs an interactive terminal.
- When a turn or token limit ends an `AgentChat.ask()` before any
  answer text, the reply names the exhausted limit and its value rather
  than reporting that the model produced no answer.
- Raw tool-call syntax scrubbing now also recognizes a bare JSON
  tool-call object written as prose (`{"name": ..., "parameters":
  {...}}`, and the OpenAI-style `{"type": "function", ...}` envelope),
  as the llama family emits when it means to call a tool but bypasses
  the provider's tool-call channel. Such a reply is scrubbed from the
  answer and the stored conversation, and the existing plain-prose retry
  produces a real answer instead of the JSON flowing down the pipeline.
  Data that merely looks JSON-ish (`{"name": "Ada", "age": 3}`) is left
  alone.
- The missing-provider-extra error names this package's extra
  (`pip install 'talkpipe-agents[openai]'`, and the checkout form) rather
  than the underlying `strands-agents` one, which installs the provider
  without recording it in the application's own dependencies.
- `unknown agent model source 'eliza'` now says why: talkpipe's scripted
  responder cannot call tools, so it is a `llmPrompt` source only — pass
  `source=`, or use the `scripted` source in tests. A configured
  `default_model_source = "eliza"` is a common way to reach this error.
- The README's install block gives the PyPI install and the checkout
  install, states the Python floor, and says that the source install needs
  a git clone: the version comes from git tags, so building a directory
  without `.git` fails.
- The README's Python quickstart is two runnable snippets — compose
  `LLMAgent` into a pipeline, or compile the equivalent chatterlang
  script — replacing one that referenced an undefined `script` variable.
- The README documents that `register_agent_tool(name=...)` aliases the
  registry only (the model, progress lines and scripted tests see the
  tool's own name, which `@strands.tool(name=...)` changes), gives the
  checkout form of the provider extras, notes that the
  `register_agent_model` example needs the provider package of the
  Strands model class it imports, says a multi-statement script (the
  `scriptTool` example) is run from a file with `--script path`, notes
  that the offline fixture needs pytest and that the scripted model
  reports no token usage, and points at talkpipe's component reference
  for every `llmAgent` parameter. `AgentChat` and reply hygiene have
  their own headings instead of trailing the scripts-as-tools section.
- Public docstrings no longer cite issue numbers from the host
  application the layer was extracted from.
