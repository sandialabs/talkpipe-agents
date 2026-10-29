"""A suite of generally useful agent tools, registered by name.

Everything here is generic — no host-application knowledge — and
follows the conventions of ``tools.py``: plain ``@strands.tool``
functions whose docstrings become the description and whose type hints
become the schema, registered with ``register_agent_tool`` so
chatterlang scripts can opt into them by name. Nothing is available to
an agent unless the script names it.

The file and web tools are talkpipe segments compiled into callables
(``as_function``) rather than code of their own: ``readFile`` reads
any format talkpipe has an extractor for (text, markdown, HTML, docx,
PDF, CSV, JSON, …), ``listFiles`` globs, and ``downloadURL |
htmlToText`` fetches a URL — respecting robots.txt, retrying transient
failures, and reducing HTML to its readable text. The tools only add
the agent-facing bounds: file access is read-only, listings and
file/URL contents are clipped to a stated size with an explicit
truncation notice, and ``fetch_url`` speaks only http(s). The
calculator evaluates arithmetic through an AST walker rather than
``eval``. The notepad keeps its notes in the agent's own Strands
``AgentState``, so every agent has its own pad with no host wiring.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from strands import tool as strands_tool
from strands.types.tools import ToolContext
from talkpipe.data.extraction import (
    ReadFile,
    extract_text,
    get_default_registry,
    listFiles,
)
from talkpipe.data.html import downloadURLSegment, htmlToTextSegment

from .tools import register_agent_tool

_MAX_SEARCH_FILE_BYTES = 1_000_000  # search_files skips bigger files
_DEFAULT_MAX_CHARS = 10000  # read_file / fetch_url output clip

# Every optional parameter below also accepts None as "use the default":
# models (llama3.2 among them) send null for optional arguments they
# leave unset, and Strands validates arguments against the type hints,
# so a plain ``int = 10000`` would reject such a call before the tool ran.


def _clip(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        max_chars = _DEFAULT_MAX_CHARS
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated; {len(text) - max_chars} more characters]"


# --- calculator -------------------------------------------------------

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "exp",
        "log",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "floor",
        "ceil",
        "degrees",
        "radians",
        "factorial",
    )
}
_FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max})


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 5000:
            raise ValueError("factorial argument too large")
        return _FUNCTIONS[node.func.id](*args)
    raise ValueError(
        f"unsupported syntax at {ast.unparse(node)!r}; only numbers, "
        "+ - * / // % **, parentheses, the constants "
        f"{', '.join(sorted(_CONSTANTS))} and the functions "
        f"{', '.join(sorted(_FUNCTIONS))} are allowed"
    )


@register_agent_tool
@strands_tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the result.
    Use this for any calculation instead of doing arithmetic yourself.
    Supports + - * / // % ** and parentheses, the constants pi, e, tau
    and inf, and the functions sqrt, exp, log, log2, log10, sin, cos,
    tan, asin, acos, atan, floor, ceil, degrees, radians, factorial,
    abs, round, min and max.

    Args:
        expression: The expression to evaluate, e.g. "(3 + 4) * 2" or
            "sqrt(2) / sin(pi / 4)".
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {exc.msg}") from exc
    result = _eval_node(parsed)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return str(result)


# --- dates and times --------------------------------------------------


@register_agent_tool
@strands_tool
def current_time(timezone: str | None = None) -> str:
    """Get the current date and time. Use this whenever "now", "today"
    or a current timestamp matters — do not guess the date.

    Args:
        timezone: IANA timezone name such as "UTC" or
            "America/New_York"; omit it for the local timezone.
    """
    now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
    return f"{now.isoformat(timespec='seconds')} ({now:%A})"


@register_agent_tool
@strands_tool
def days_between(start: str, end: str) -> str:
    """Count the days from one date to another exactly. The result is
    negative when end is before start. Use this instead of computing
    date differences yourself.

    Args:
        start: The start date in ISO format, e.g. "2026-01-31".
        end: The end date in ISO format, e.g. "2026-03-01".
    """
    try:
        delta = (
            datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()
        )
    except ValueError as exc:
        raise ValueError(
            f"dates must be ISO formatted, like 2026-01-31: {exc}"
        ) from exc
    return str(delta.days)


# --- text -------------------------------------------------------------


@register_agent_tool
@strands_tool
def regex_extract(pattern: str, text: str, max_matches: int | None = 20) -> str:
    """Extract every match of a regular expression from text, one match
    per line. If the pattern has capture groups, the groups are
    returned tab-separated; otherwise the whole match is returned.

    Args:
        pattern: A Python regular expression, e.g.
            r"\\b[A-Z]{2,5}\\b" or r"(\\w+)=(\\d+)".
        text: The text to search.
        max_matches: Stop after this many matches.
    """
    if max_matches is None:
        max_matches = 20
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
    matches = []
    for m in compiled.finditer(text):
        matches.append("\t".join(m.groups()) if m.groups() else m.group(0))
        if len(matches) >= max_matches:
            break
    return "\n".join(matches) if matches else "(no matches)"


# --- files (read-only) ------------------------------------------------
#
# talkpipe's ``readFile`` segment, compiled into a callable. Its default
# registry knows text, markdown, HTML, docx, PDF, CSV, JSON and more; a
# copy with ``extract_text`` as the fallback reads any other extension
# (source code, logs, extensionless files) as text, so an agent is not
# limited to the formats talkpipe names. ``skip_errors=False`` makes an
# unreadable file an error the agent sees, not a silent empty result;
# ``search_files`` keeps the default (skip and continue) so one bad
# file does not abort a search.

_REGISTRY = get_default_registry()
_REGISTRY.register_default(extract_text)
_read_file = ReadFile(registry=_REGISTRY, skip_errors=False).as_function(single_in=True)
_read_file_or_skip = ReadFile(registry=_REGISTRY).as_function(single_in=True)
_list_files = listFiles(full_path=True).as_function(single_in=True)
_list_files_recursive = listFiles(full_path=True, files_only=True).as_function(
    single_in=True
)


def _file_text(read: Callable[[str], list[Any]], path: str) -> str:
    """The text of *path* as read by a ``readFile`` callable; a format
    that yields several items (CSV rows, JSONL lines) is joined one per
    line."""
    return "\n".join(result.content for result in read(path))


@register_agent_tool
@strands_tool
def read_file(path: str, max_chars: int | None = _DEFAULT_MAX_CHARS) -> str:
    """Read a file and return its text. Text, markdown, HTML, Word
    (.docx), PDF, CSV, JSON and JSONL are understood (HTML is reduced to
    its readable text, one CSV row or JSONL record per line); any other
    extension is read as plain text. Long files are truncated with an
    explicit notice.

    Args:
        path: Path to the file ("~" is expanded).
        max_chars: Return at most this many characters.
    """
    return _clip(_file_text(_read_file, os.path.expanduser(path)), max_chars)


@register_agent_tool
@strands_tool
def list_files(directory: str | None = ".", pattern: str | None = "*") -> str:
    """List the entries of a directory, one per line — subdirectories
    with a trailing "/", files with their size in bytes.

    Args:
        directory: The directory to list: "." (the default) is the current
            directory; "~" is expanded.
        pattern: Glob pattern to filter entries, e.g. "*.py".
    """
    directory, pattern = directory or ".", pattern or "*"
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a directory: {directory}")
    lines = []
    for entry in map(Path, _list_files(str(root / pattern))):
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            lines.append(f"{entry.name}\t{entry.stat().st_size} bytes")
    if len(lines) > 200:
        lines = [*lines[:200], f"…[{len(lines) - 200} more entries]"]
    return "\n".join(lines) if lines else "(no matching entries)"


@register_agent_tool
@strands_tool
def search_files(
    pattern: str,
    directory: str | None = ".",
    file_glob: str | None = "*",
    max_matches: int | None = 50,
) -> str:
    """Search the files under a directory (recursively) for a regular
    expression and return matching lines as "path:line: text". Files
    are read as read_file does, so Word, PDF and CSV documents are
    searched by their text and line numbers match read_file's output.
    Hidden directories, unreadable files and files over 1 MB are
    skipped.

    Args:
        pattern: A Python regular expression to search for.
        directory: The directory to search under: "." (the default) is the
            current directory; "~" is expanded.
        file_glob: Glob pattern for which files to search, e.g. "*.md".
        max_matches: Stop after this many matching lines.
    """
    directory, file_glob = directory or ".", file_glob or "*"
    if max_matches is None:
        max_matches = 50
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a directory: {directory}")
    root = root.resolve()  # listFiles yields resolved paths
    lines: list[str] = []
    for path in map(Path, _list_files_recursive(str(root / "**" / file_glob))):
        if path.stat().st_size > _MAX_SEARCH_FILE_BYTES:
            continue
        rel = os.path.relpath(path, root)
        text = _file_text(_read_file_or_skip, str(path))
        for lineno, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                lines.append(f"{rel}:{lineno}: {line.strip()}")
                if len(lines) >= max_matches:
                    return "\n".join(lines) + "\n…[match limit reached]"
    return "\n".join(lines) if lines else "(no matches)"


# --- web --------------------------------------------------------------
#
# ``downloadURL | htmlToText``, compiled into a callable: the download
# honours robots.txt, sends browser-like headers (the User-Agent follows
# talkpipe's ``user_agent`` config key) and retries transient failures;
# the text extraction keeps the readable body of an HTML page and
# leaves plain text, JSON and markdown as they are.

_fetch_url = (downloadURLSegment(timeout=30) | htmlToTextSegment()).as_function(
    single_in=True, single_out=True
)


@register_agent_tool
@strands_tool
def fetch_url(url: str, max_chars: int | None = _DEFAULT_MAX_CHARS) -> str:
    """Fetch a web page or other http(s) resource and return its text.
    HTML is reduced to its readable text; long content is truncated with
    an explicit notice.

    Args:
        url: The http:// or https:// URL to fetch.
        max_chars: Return at most this many characters.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http:// and https:// URLs are supported")
    return _clip(_fetch_url(url), max_chars)


# --- notepad ----------------------------------------------------------
#
# The pad lives in the agent's Strands ``AgentState`` (JSON-serializable,
# per agent, persisted with the agent by a session manager), so each
# conversation has its own notes with no factory or host wiring:
# ``AgentChat`` gets one pad per conversation and ``llmAgent`` one per
# segment (its agents share a state, so notes carry across the items of
# a stream while the conversation does not). Note numbers are stable:
# deleting a note never renumbers the others.

_PAD_KEY = "notepad"


def _pad(tool_context: ToolContext) -> dict[str, Any]:
    return tool_context.agent.state.get(_PAD_KEY) or {"next": 1, "notes": {}}


def _save_pad(tool_context: ToolContext, pad: dict[str, Any]) -> None:
    tool_context.agent.state.set(_PAD_KEY, pad)


def _require(pad: dict[str, Any], number: int) -> str:
    key = str(number)
    if key not in pad["notes"]:
        have = ", ".join(pad["notes"]) or "none"
        raise ValueError(f"no note {number} (saved notes: {have})")
    return key


@register_agent_tool
@strands_tool(context=True)
def save_note(note: str, tool_context: ToolContext) -> str:
    """Save a short note to your scratch pad, e.g. an intermediate
    result or something to remember for later. Notes persist for the
    rest of the run and can be read back with read_notes.

    Args:
        note: The note to save, self-contained and concise.
    """
    pad = _pad(tool_context)
    number = pad["next"]
    pad["notes"][str(number)] = note
    pad["next"] = number + 1
    _save_pad(tool_context, pad)
    return f"(note {number} saved)"


@register_agent_tool
@strands_tool(context=True)
def read_notes(tool_context: ToolContext) -> str:
    """Read back all notes previously saved with save_note, in order,
    each prefixed with its number."""
    notes = _pad(tool_context)["notes"]
    if not notes:
        return "(no notes saved yet)"
    return "\n".join(f"{n}. {note}" for n, note in notes.items())


@register_agent_tool
@strands_tool(context=True)
def summarize_notes(tool_context: ToolContext) -> str:
    """Get a compact overview of the scratch pad — how many notes,
    their total size, and a clipped preview of each — without reading
    every note in full. Use this to survey a pad that may have grown
    long before deciding what to read, update or delete."""
    notes = _pad(tool_context)["notes"]
    if not notes:
        return "(no notes saved yet)"
    total = sum(len(note) for note in notes.values())
    lines = [f"{len(notes)} notes, {total} characters in total:"]
    for number, note in notes.items():
        preview = note.split("\n", 1)[0][:80]
        if len(preview) < len(note):
            preview += "…"
        lines.append(f"{number}. ({len(note)} chars) {preview}")
    return "\n".join(lines)


@register_agent_tool
@strands_tool(context=True)
def update_note(number: int, note: str, tool_context: ToolContext) -> str:
    """Replace a previously saved note with new text, keeping its
    number. Use this to correct or refine what you remember instead of
    saving a near-duplicate.

    Args:
        number: The note's number, as shown by read_notes.
        note: The replacement text.
    """
    pad = _pad(tool_context)
    pad["notes"][_require(pad, number)] = note
    _save_pad(tool_context, pad)
    return f"(note {number} updated)"


@register_agent_tool
@strands_tool(context=True)
def delete_note(number: int, tool_context: ToolContext) -> str:
    """Delete one note permanently. Use this to expunge memory that is
    stale or no longer needed; the remaining notes keep their numbers.

    Args:
        number: The note's number, as shown by read_notes.
    """
    pad = _pad(tool_context)
    del pad["notes"][_require(pad, number)]
    _save_pad(tool_context, pad)
    return f"(note {number} deleted)"


@register_agent_tool
@strands_tool(context=True)
def clear_notes(tool_context: ToolContext) -> str:
    """Delete every note, leaving the scratch pad empty. To compact
    rather than forget: read_notes, then clear_notes, then save one
    note summarizing what still matters."""
    pad = _pad(tool_context)
    deleted = len(pad["notes"])
    pad["notes"] = {}
    _save_pad(tool_context, pad)
    if not deleted:
        return "(the pad was already empty)"
    return f"({deleted} notes deleted)"
