"""The plugin contract: installing this package makes talkpipe itself
register the llmAgent segment — no import of talkpipe_agents needed."""

from importlib import metadata

import pytest


def test_entry_point_declared():
    eps = metadata.entry_points(group="talkpipe.plugins")
    assert any(ep.name == "agents" and ep.value == "talkpipe_agents" for ep in eps)


def test_segments_registered_via_talkpipe_import():
    # importing talkpipe triggers plugin discovery, which imports this
    # package and thereby registers the segments
    import talkpipe  # noqa: F401
    from talkpipe.chatterlang import registry

    assert "llmAgent" in registry.segment_registry.available_names
    assert "scriptTool" in registry.segment_registry.available_names


def test_chatterlang_compiles_llm_agent():
    from talkpipe.chatterlang import compiler

    # unknown tool names must fail at compile time, proving the segment
    # the registry serves is this package's (its constructor resolves
    # tool names eagerly); the error is both a ValueError and a
    # chatterlang CompileError, so chatterlang_script prints one line
    with pytest.raises(ValueError, match="no_such_tool") as info:
        compiler.compile('| llmAgent[tools=["no_such_tool"]]')
    assert isinstance(info.value, compiler.CompileError)


def test_notepad_tools_exported_with_the_rest():
    import talkpipe_agents

    notepad = (
        "save_note",
        "read_notes",
        "summarize_notes",
        "update_note",
        "delete_note",
        "clear_notes",
    )
    for name in notepad:
        assert name in talkpipe_agents.__all__
        assert getattr(talkpipe_agents, name).tool_name == name
