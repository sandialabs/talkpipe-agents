"""pytest plugin, auto-loaded through the ``pytest11`` entry point when
this package is installed: the ``scripted_model`` fixture.

Only pytest imports this module; nothing in the package does. A project
that does not want the fixture runs ``pytest -p no:talkpipe_agents``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .testing import ScriptedModel, scripted_source


@pytest.fixture
def scripted_model() -> Iterator[ScriptedModel]:
    """A fresh talkpipe_agents.testing.ScriptedModel, registered as agent
    model source "scripted" for this test. Script the model's turns with
    .call(tool, **args) and .say(text), run your segment or chat with
    source="scripted", then assert on the answer and on .tool_results.

    Example::

        def test_agent_uses_my_tool(scripted_model):
            scripted_model.call("lookup_ticket", ticket_id="T-1").say("Not urgent.")
            step = LLMAgent(source="scripted", model="any", tools=["lookup_ticket"])
            assert list(step(["How urgent is T-1?"])) == ["Not urgent."]
            assert scripted_model.tool_results == [
                ("lookup_ticket", "success", "Printer on fire, third floor")
            ]
    """
    with scripted_source() as model:
        yield model
