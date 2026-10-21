"""Chatterlang scripts as agent tools.

``make_script_tool`` turns a chatterlang script plus its documentation
into an ordinary Strands tool; the ``scriptTool`` segment does the same
from inside chatterlang and registers the result under
``register_agent_tool``, so one script can carry its own short tooling
scripts and hand them to ``llmAgent[tools=[...]]`` further down.
Segments are constructed in statement order when a script is compiled,
so a ``scriptTool`` statement makes its name available to every later
statement — and a broken tool script fails at compile time, before
anything runs.

The tool takes one string argument, fed to the script as its input
stream (a script that starts with its own ``INPUT FROM`` ignores it).
Each call compiles a fresh instance of the script, so stateful segments
and ``@variables`` start clean every time and concurrent calls share
nothing. The tool returns the script's output items joined with
newlines.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Annotated, Any

from strands import tool as strands_tool
from talkpipe.chatterlang.compiler import CompileError
from talkpipe.chatterlang.compiler import compile as compile_script
from talkpipe.chatterlang.registry import register_segment
from talkpipe.pipe.core import AbstractSegment

from .tools import register_agent_tool

logger = logging.getLogger(__name__)

_INPUT_DOC = "The input text, fed to the script as its input stream."


def make_script_tool(
    name: str, description: str, script: str, input_description: str = _INPUT_DOC
) -> Any:
    """A Strands tool named *name* whose body is the chatterlang
    *script*. *description* documents the tool to the model (what it
    does, what to pass); *input_description* documents its single
    ``text`` argument. The script is compiled once here so mistakes
    surface at definition time, then freshly per call so no state
    leaks between calls."""
    try:
        compile_script(script)
    except CompileError as e:
        raise CompileError(
            f"the script for tool {name!r} does not compile: {e}",
            line=e.line,
            column=e.column,
            kind=e.kind,
            bad_name=e.bad_name,
        ) from None

    def run(text: str = "") -> str:
        pipeline = compile_script(script)
        return "\n".join(str(item) for item in pipeline([text]))

    run.__doc__ = f"{description}\n\nArgs:\n    text: {input_description}\n"
    return strands_tool(name=name, description=description)(run)


@register_segment("scriptTool")
class ScriptTool(AbstractSegment):
    """Define a chatterlang script as an agent tool and register it by
    name. Registration happens when the segment is constructed — at
    compile time of the enclosing script — so the definition can stand
    as its own statement (ending in ``;``) ahead of the ``llmAgent``
    that names it. In a running pipeline the segment passes items
    through unchanged."""

    def __init__(
        self,
        name: Annotated[str, "The tool name, as referenced in llmAgent[tools=[...]]"],
        description: Annotated[
            str, "What the tool does and what to pass it, addressed to the model"
        ],
        script: Annotated[str, "The chatterlang script that is the tool's body"],
        input_description: Annotated[
            str | None, "Documentation for the tool's single text argument"
        ] = None,
    ):
        super().__init__()
        self.tool = register_agent_tool(
            make_script_tool(
                name,
                description,
                script,
                input_description=input_description or _INPUT_DOC,
            )
        )

    def transform(self, input_iter: Iterable) -> Iterator:
        yield from input_iter
