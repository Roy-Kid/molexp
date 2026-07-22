"""Agent-session mount-context render parity (vision-loop-11).

``routes.agent._mount_context`` must produce the same context block as the CLI
one-path builder ``services.agent_context.build_mount_context`` and anchor the
session at the run dir. The context-block *content* contract lives in
``tests/test_services/test_agent_context.py``.

Spec: ``.claude/specs/vision-loop-11-mount-context.md`` (Design §3).
"""

from __future__ import annotations

from pathlib import Path

from molexp.server.routes import agent as agent_routes
from molexp.workspace import Run, Workspace


class TestMountContext:
    """``routes.agent._mount_context`` — CLI/server one-path parity (ac-004)."""

    def test_server_render_matches_cli_build_mount_context(
        self, workspace: Workspace, run: Run
    ) -> None:
        from molexp.services.agent_context import build_mount_context

        cli_block = build_mount_context(
            workspace, project_id="test-project", experiment_id="test-exp", run_id=run.id
        )
        server_block, server_anchor = agent_routes._mount_context(
            workspace, project_id="test-project", experiment_id="test-exp", run_id=run.id
        )
        assert server_block == cli_block
        assert server_anchor == Path(str(run.run_dir))
