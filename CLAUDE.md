# talkpipe-agents — project notes for Claude

Agentic segments for talkpipe built on Strands Agents, packaged as a
**talkpipe plugin** (`talkpipe.plugins` entry point → importing
talkpipe registers the `llmAgent` segment). Extracted from the staging
area of the host application it was proven in; see README.md for the
design ("Putting Agents in their Place").

## Layout

- `src/talkpipe_agents/` — the package: `segments.py` (`llmAgent`),
  `chat.py` (`AgentChat`), `models.py` (source→Strands-Model registry),
  `ollama_model.py` (`ThinkingOllamaModel`: the `ollama` source's
  provider, replaying `thinking` and threading turns),
  `tools.py` (name→tool registry + `ask_user`/`think`),
  `script_tools.py` (chatterlang scripts as tools: `make_script_tool`
  + the `scriptTool` segment), `common_tools.py` (the generic tool
  suite: calculator, time/date, regex, read-only files, `fetch_url`,
  notepad, kept in the agent's Strands state), `hooks.py`
  (`AgentHooks`: progress lines + leak scrubbing as Strands stores
  messages), `replies.py` (`ask_and_clean` under Strands `limits`, with
  the empty-reply retry), `support.py` (`teller`, `strip_think`,
  `strip_tool_leak`), `testing.py` (`ScriptedModel` + `scripted_source`,
  the public offline model; `_pytest_plugin.py` exposes it as the
  `scripted_model` fixture through the `pytest11` entry point).
- `tests/` — pytest, fully offline (the package's own `scripted_model`
  fixture; no live model server, no network). Keep it that way.

## Environment

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Code quality — CI fails on any of these, so run them before pushing:

```bash
.venv/bin/ruff check .           # fix with: ruff check --fix .
.venv/bin/ruff format --check .  # fix with: ruff format .
.venv/bin/mypy                   # src/, config in pyproject.toml
```

`pre-commit install` (opt-in, per clone) runs the same checks on commit;
`pre-commit run --all-files` reproduces the CI gate locally.

## Constraints

- **Nothing here may import from any host application** — this
  package must stay generic. Application-specific tools belong in the
  applications, registered via `register_agent_tool`.
- The host application this layer was extracted from still bundles its
  own copy and does **not** depend on this repo; never install this
  plugin into an environment that also registers that bundled copy
  (the `llmAgent` registration would collide), and do not modify other
  projects from here. The host's migration onto this package is a
  separate, deliberate change.
- The model provider is pluggable — `ollama`, `openai`, `anthropic`,
  `litellm`, or any `register_agent_model` source
  (`docs/model-providers.md`); keep docs and code provider-neutral. This workspace has
  no OpenAI/Anthropic keys, so live experiments use Ollama via
  `TALKPIPE_OLLAMA_SERVER_URL` (`http://deeplearn:11434`); tests must
  never require a live model of any provider.

## Workflow

This repo is hosted on the deeplearn.local Gitea. The branch → commit →
AGit merge request workflow, no-attribution rule, API-token review
loop, and issue handling are defined globally in `~/.claude/CLAUDE.md`
and apply here. MRs target `main`. Leave the checkout clean and on
`main` when done — the issue agent refuses dirty trees.
