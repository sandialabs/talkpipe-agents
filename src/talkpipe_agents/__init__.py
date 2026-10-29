"""talkpipe-agents — agentic segments for talkpipe, built on Strands.

A talkpipe segment on the outside, a ``strands.Agent`` on the inside:
pipelines stay deterministic while individual steps may reason and call
tools. Installed as a talkpipe plugin, importing talkpipe registers the
``llmAgent`` segment automatically; the package is also importable
directly for Python composition.

- ``models`` — source/model strings resolved with ``llmPrompt``'s
  semantics (same config fallbacks), decoded to Strands model providers
  through a ``register_agent_model`` registry. ``ollama`` works with the
  base install; ``openai`` (= ``openai_responses``), ``openai_chat``,
  ``anthropic`` and ``litellm`` are registered too and import lazily
  behind their pip extras.
- ``tools`` — a name→tool registry so chatterlang scripts can reference
  tools by name (any other string goes to Strands' own tool loader),
  plus the generic ``ask_user`` and ``think`` tools.
- ``script_tools`` — chatterlang scripts as tools: ``make_script_tool``
  and the ``scriptTool`` segment, which lets one script define
  documented tooling scripts inline and hand them to ``llmAgent``.
- ``common_tools`` — a suite of generally useful, bounded tools
  (calculator, time and date, text extraction, read-only file access,
  URL fetching, a notepad kept in the agent's Strands state), all
  registered by name.
- ``findings`` — the findings loop: ``Finding``, ``finding_tool``,
  ``FindingsSession`` (one question's kept/ignored findings, stop rule,
  turn countdown), the ``FindingsJudge`` (also the ``judgeFindings``
  segment) and the ``JudgeFindings`` hook that judges every finding
  tool's results inside the agent loop, plus the ``enough_information``
  / ``keep_finding`` / ``ignore_finding`` session tools; turned on with
  ``llmAgent[judge=True]``. ``findingTool`` (in ``script_tools``) makes
  a chatterlang script a finding tool.
- ``segments`` — the ``llmAgent`` chatterlang segment.
- ``chat`` — ``AgentChat``, a thin multi-turn wrapper for interactive
  sessions (what a chat UI drives).
- ``hooks`` — ``AgentHooks`` and ``AgentRetryStrategy``, the Strands
  hook providers both entry points install: per-tool-call progress
  lines, scrubbing of leaked tool-call syntax as messages are stored,
  and retrying a model call whose tool call the server could not parse.
- ``replies`` — ``ask_and_clean`` (one prompt under Strands ``limits``,
  with the empty-reply retry) and ``agent_limits``.
- ``support`` — the shared progress convention (``teller``) and reply
  cleanup (``strip_think``, ``strip_tool_leak``).
- ``testing`` — ``ScriptedModel`` and ``scripted_source``: a Strands
  ``Model`` that replays scripted tool calls and answers, so an
  application tests its agent steps offline; the ``scripted_model``
  pytest fixture is installed with the package as a pytest plugin.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .chat import AgentChat
from .common_tools import (
    calculator,
    clear_notes,
    current_time,
    days_between,
    delete_note,
    fetch_url,
    list_files,
    read_file,
    read_notes,
    regex_extract,
    save_note,
    search_files,
    summarize_notes,
    update_note,
)
from .findings import (
    Finding,
    FindingsJudge,
    FindingsOutcome,
    FindingsSession,
    JudgeFindings,
    Judgement,
    as_findings,
    enough_information,
    finding_tool,
    ignore_finding,
    keep_finding,
    parse_judgements,
)
from .hooks import AgentHooks
from .models import agent_model_sources, build_agent_model, register_agent_model
from .replies import agent_limits, ask_and_clean

try:
    __version__ = _version("talkpipe-agents")
except _PackageNotFoundError:  # pragma: no cover - only when run from an unbuilt tree
    __version__ = "0.0.0+unknown"
from .script_tools import FindingTool, ScriptTool, make_finding_tool, make_script_tool
from .segments import LLMAgent
from .support import strip_think, strip_tool_leak, teller
from .tools import (
    UnknownAgentToolError,
    agent_tool_names,
    make_ask_user,
    register_agent_tool,
    resolve_agent_tools,
    think,
)

__all__ = [
    "AgentChat",
    "AgentHooks",
    "Finding",
    "FindingTool",
    "FindingsJudge",
    "FindingsOutcome",
    "FindingsSession",
    "JudgeFindings",
    "Judgement",
    "LLMAgent",
    "ScriptTool",
    "UnknownAgentToolError",
    "__version__",
    "agent_limits",
    "agent_model_sources",
    "agent_tool_names",
    "as_findings",
    "ask_and_clean",
    "build_agent_model",
    "calculator",
    "clear_notes",
    "current_time",
    "days_between",
    "delete_note",
    "enough_information",
    "fetch_url",
    "finding_tool",
    "ignore_finding",
    "keep_finding",
    "list_files",
    "make_ask_user",
    "make_finding_tool",
    "make_script_tool",
    "parse_judgements",
    "read_file",
    "read_notes",
    "regex_extract",
    "register_agent_model",
    "register_agent_tool",
    "resolve_agent_tools",
    "save_note",
    "search_files",
    "strip_think",
    "strip_tool_leak",
    "summarize_notes",
    "teller",
    "think",
    "update_note",
]


def initialize_plugin() -> None:
    """Talkpipe plugin hook. Importing this module has already
    registered the ``llmAgent`` segment with talkpipe's registry (the
    decorators run at import), so there is nothing further to do —
    the hook exists to make the plugin contract explicit."""
