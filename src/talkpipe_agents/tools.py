"""Agent tool registry and the generic ask-user and think tools.

Tools are ordinary Strands tools (``@strands.tool`` on a function —
docstring becomes the description, type hints the schema). Registering
one here gives it a short name chatterlang scripts can reference in
``llmAgent[tools=[...]]``; Python callers may also pass tool objects
directly. Any string that is not a registered name is handed to
Strands' own tool loader, so a script can also name a tool by its
module path (``"strands.vended_tools:http_request"``,
``"my_pkg.tools"`` for every tool in a module) or by a file path,
with no registration at all. Tool packs whose tools need runtime state
(a database handle, settings) should expose factories returning bound
tools and register the results at startup.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable
from inspect import isfunction, ismethod
from pathlib import Path
from types import ModuleType
from typing import Any

from strands import tool as strands_tool
from strands.plugins import Plugin
from strands.tools.loader import load_tool_from_string
from strands.vended_plugins.skills import AgentSkills, SkillSource
from talkpipe.chatterlang.compiler import CompileError

logger = logging.getLogger(__name__)

_tools: dict[str, Any] = {}


class UnknownAgentToolError(CompileError, ValueError):
    """A tool string that is neither a registered name nor something
    Strands' loader can resolve. A ``ValueError`` for Python callers,
    and a chatterlang ``CompileError`` so script runners report it as
    a one-line compile error rather than a traceback."""


def _as_tool(func: Any) -> Any:
    """Wrap a plain callable as a Strands tool; its docstring is the
    model's only description of it, so its absence is worth a warning."""
    if not getattr(func, "__doc__", None):
        logger.warning(
            "agent tool %r has no docstring; the model is told nothing about "
            "what it does or when to call it",
            getattr(func, "__name__", func),
        )
    return strands_tool(func)


def register_agent_tool(tool_obj: Any = None, *, name: str | None = None) -> Any:
    """Register a tool for lookup by name, so chatterlang scripts (and
    Python callers) can reference it as a string in ``tools=[...]``.

    Accepts a ``@strands.tool`` object or a plain callable, which gets
    wrapped — its docstring becomes the tool's description and its type
    hints the schema. Returns the tool, so this doubles as a decorator,
    bare or with arguments::

        @register_agent_tool
        def lookup_ticket(ticket_id: str) -> str:
            \"\"\"Look up a support ticket by id and return its text.\"\"\"
            ...

        @register_agent_tool(name="tickets")
        def lookup_ticket(ticket_id: str) -> str: ...

    ``name`` is the registry key only — the name scripts put in
    ``tools=[...]``. The tool itself keeps its own name (the function's,
    or the one given to ``@strands.tool``), which is what the model
    sees, what progress lines report, and what an offline test scripts
    with ``scripted_model.call(...)``. Pass ``name`` for a script-facing
    alias; rename the function (or use ``@strands.tool(name=...)``) to
    change what the model is told.

    Registration is last-wins so an application can deliberately replace
    a shipped tool; an accidental replacement is logged as a warning.
    """
    if tool_obj is None:

        def decorate(obj: Any) -> Any:
            return register_agent_tool(obj, name=name)

        return decorate
    if not hasattr(tool_obj, "tool_name"):
        tool_obj = _as_tool(tool_obj)
    key = name or tool_obj.tool_name
    previous = _tools.get(key)
    if previous is not None and previous is not tool_obj:
        logger.warning(
            "agent tool %r re-registered; the previous tool is replaced", key
        )
    _tools[key] = tool_obj
    return tool_obj


def agent_tool_names() -> list[str]:
    """The names currently registered with ``register_agent_tool`` (the
    shipped common tools plus whatever the application added), sorted."""
    return sorted(_tools)


class UnknownAgentSkillsError(CompileError, ValueError):
    """A skills path that does not exist. Strands would only log a
    warning and run the agent with no skills; a chatterlang script
    should fail when it compiles instead."""


def skills_plugin(skills: str | Iterable[str] | None) -> list[Plugin]:
    """Strands' ``AgentSkills`` plugin for *skills*, as an ``Agent``'s
    ``plugins=`` list — empty when *skills* is empty. Each source is a
    folder of skill subdirectories (each holding a ``SKILL.md``), one
    skill directory, a ``SKILL.md`` path, or an ``https://`` URL to raw
    ``SKILL.md`` content; a single string is one source, not iterated.
    The plugin lists the skills in the system prompt and gives the agent
    a ``skills`` tool that loads a skill's full instructions on demand.
    A filesystem source that does not exist raises
    :class:`UnknownAgentSkillsError` here, at compile time."""
    sources: list[SkillSource] = (
        [skills] if isinstance(skills, str) else list(skills or [])
    )
    if not sources:
        return []
    for source in sources:
        if (
            isinstance(source, str)
            and not source.startswith("https://")
            and not Path(source).exists()
        ):
            raise UnknownAgentSkillsError(
                f"skills path {str(source)!r} does not exist; skills= takes a "
                "folder of skill directories, a skill directory, a SKILL.md "
                "path, or an https:// URL"
            )
    return [AgentSkills(skills=sources)]


def resolve_agent_tools(specs: Iterable[Any] | str) -> list[Any]:
    """Tool objects for *specs*: strings resolve through the registry,
    then through Strands' loader (module or file paths); unknown names
    raise :class:`UnknownAgentToolError`, so chatterlang scripts fail at
    compile time. Plain callables are wrapped as Strands tools (as
    ``register_agent_tool`` does); tool objects, modules and Strands tool
    dicts pass through; anything else (a number, ``None``) raises
    ``TypeError`` here rather than being dropped by Strands at run time.
    A single string is taken as one spec, not iterated."""
    if isinstance(specs, str):
        specs = [specs]
    resolved = []
    for spec in specs:
        if not isinstance(spec, str):
            if (isfunction(spec) or ismethod(spec)) and not hasattr(spec, "tool_name"):
                spec = _as_tool(spec)
            elif not (
                callable(spec)
                or hasattr(spec, "tool_name")
                or isinstance(spec, (dict, ModuleType))
            ):
                raise TypeError(
                    f"not a tool: {spec!r} ({type(spec).__name__}); tools= takes "
                    "registered names, Strands tool paths, @strands.tool objects, "
                    "or plain functions"
                )
            resolved.append(spec)
        elif spec in _tools:
            resolved.append(_tools[spec])
        else:
            try:
                resolved.extend(load_tool_from_string(spec))
            except Exception as exc:
                # A bare name was meant for the registry, so the loader's
                # failure to import a module by that name is noise; for a
                # path-like string the loader's error is the useful part.
                is_path = any(c in spec for c in ":/\\.")
                hint = (
                    f"; Strands' loader could not resolve it ({exc}); a Strands "
                    "tool path looks like 'strands.vended_tools:http_request'"
                    if is_path
                    else "; if it is your own tool, is the module that registers "
                    "it imported? (chatterlang_script: --load-module PATH; "
                    "or give a Strands tool path such as "
                    "'strands.vended_tools:http_request')"
                )
                raise UnknownAgentToolError(
                    f"unknown agent tool {spec!r}; registered tools: "
                    f"{', '.join(agent_tool_names()) or '(none)'}{hint}",
                    bad_name=spec,
                ) from (exc if is_path else None)
    return resolved


def _default_ask(question: str) -> str:
    if sys.stdin is not None and sys.stdin.isatty():
        return input(f"{question} ")
    return "(the user is not available to answer; continue with what you have)"


def make_ask_user(ask: Callable[[str], str] | None = None) -> Any:
    """The ask-user tool, bound to *ask* — a callable that shows the
    question to the human and returns their reply. Defaults to stdin
    when interactive, a polite "user unavailable" answer when headless;
    a UI passes its own callable (e.g. a modal prompt)."""
    ask = ask or _default_ask

    @strands_tool
    def ask_user(question: str) -> str:
        """Ask the human user one specific question and return their
        answer. Use only when required information cannot be obtained
        any other way.

        Args:
            question: The question to show the user, self-contained
                and answerable in one short reply.
        """
        return ask(question)

    return ask_user


@register_agent_tool
@strands_tool
def think(thought: str) -> str:
    """Think through the problem step by step without taking any
    external action. Use between tool calls to reason iteratively:
    lay out what you have learned so far, question your current
    hypothesis, and plan what to check next. The thought is recorded
    for your own reference and nothing else happens.

    Args:
        thought: The reasoning step, written out in full.
    """
    return "(thought recorded — continue when ready)"


register_agent_tool(make_ask_user())
