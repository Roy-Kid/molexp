"""``lifecycle_tools`` — the opt-in workspace cancel/harvest verbs.

The gating (default operation_mode does not mount these; ``lifecycle`` mode
does) is owned by ``test_loop.py``; the agent→harness firewall is owned by
``tests/test_agent/test_import_guard.py``. This file only locks the factory's
tool surface.
"""

from __future__ import annotations

from pathlib import Path

from molexp.agent.loops.interactive.lifecycle_tools import (
    LIFECYCLE_TOOL_NAMES,
    lifecycle_tools,
)


class TestLifecycleTools:
    def test_expose_cancel_and_harvest_verbs(self, tmp_path: Path) -> None:
        names = {tool.__name__ for tool in lifecycle_tools(workspace_root=tmp_path)}
        assert names == LIFECYCLE_TOOL_NAMES == {"cancel_run", "harvest_run"}

    def test_harvest_nonterminal_returns_error_string(self, tmp_path: Path) -> None:
        """A pending run must not raise out of the tool — the model sees the error."""
        from molexp.workspace import Workspace

        ws = Workspace(tmp_path / "ws", name="ws")
        project = ws.add_project("p")
        experiment = project.add_experiment("e")
        run = experiment.add_run(params={})
        assert run.status == "pending"

        tools = {t.__name__: t for t in lifecycle_tools(workspace_root=ws.path())}
        result = tools["harvest_run"](
            project.id,
            experiment.id,
            run.id,
            narrative="should not harvest yet",
        )
        assert isinstance(result, str)
        assert result.startswith("error: ValueError:")
        assert "pending" in result
        assert "terminal" in result
