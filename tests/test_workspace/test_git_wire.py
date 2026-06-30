"""Shared CLI≡server backend + ApprovalGate-gated push.

Spec: workspace-git-projection-04-wire. The ``molexp git`` CLI and the
``/api/git/*`` server routes call the SAME backend symbols (Python ≡ UI), and
remote ``push`` — the only outward-facing, hard-to-reverse action — is a
curation ``ToolCapability`` whose ``side_effects`` route it through the harness
``ApprovalGate``. Local materialization (checkpoint / rebuild) is ungated.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import molexp.workspace.git_projection as gp
from molexp.harness import (
    StageExecutionError,
    enforce_side_effect_approvals,
)
from molexp.harness.capabilities.curation import curation_capabilities
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace import Workspace

# ── (b) Python ≡ UI — one shared backend symbol ──────────────────────────────


class TestSharedBackend:
    def test_cli_and_server_reference_the_same_backend(self):
        from molexp.cli import git_cmd as cli_git
        from molexp.server.routes import git as route_git

        # Both the CLI group and the server route import the exact backend
        # functions — not a re-implemented copy.
        assert cli_git.checkpoint is gp.checkpoint
        assert route_git.checkpoint is gp.checkpoint
        assert cli_git.rebuild is gp.rebuild
        assert route_git.rebuild is gp.rebuild
        assert cli_git.push is gp.push
        assert route_git.push is gp.push


# ── (c) push is ApprovalGate-gated; local materialization is not ─────────────


def _git_push_capability():
    return next(c for c in curation_capabilities() if c.name == "git_push")


def _make_ctx(root: Path) -> HarnessRunContext:
    db_path = root / "events.sqlite"
    artifacts = FileArtifactStore(root=root / "artifacts")
    events = SQLiteEventLog(path=db_path)
    lineage = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-git-push",
        workspace_root=root,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


async def _deny(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="test",
        decided_at=datetime.now(tz=UTC),
        reason="denied",
    )


class TestPushGate:
    def test_git_push_is_a_destructive_capability(self):
        cap = _git_push_capability()
        assert cap.side_effects == ["push:remote"]  # non-empty → gated
        assert cap.callable_path == "molexp.workspace.git_projection.push"

    async def test_denied_gate_blocks_the_push(self, tmp_path):
        ctx = _make_ctx(tmp_path / "harness")
        # The gate rejects before any push could be dispatched.
        with pytest.raises(StageExecutionError):
            await enforce_side_effect_approvals([_git_push_capability()], ctx=ctx, approve=_deny)

    async def test_read_only_capability_bypasses_the_gate(self, tmp_path):
        ctx = _make_ctx(tmp_path / "harness")
        read_only = next(c for c in curation_capabilities() if c.name == "scan_workspace")
        # A read-only capability is never gated — local materialization is ungated.
        result = await enforce_side_effect_approvals([read_only], ctx=ctx, approve=_deny)
        assert result is None

    async def test_push_backend_mirrors_refs_to_remote(self, tmp_path):
        ws = Workspace(root=tmp_path / "lab", name="Lab")
        run = ws.add_project("demo").add_experiment("baseline", params={}).add_run(params={})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"v": 1})
        await gp.checkpoint(ws)  # local materialization (ungated) builds the refs

        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        await gp.push(ws, remote=str(remote))  # the approved-path backend pushes once

        shown = subprocess.run(
            ["git", "-C", str(remote), "show-ref"], capture_output=True, text=True
        ).stdout
        assert f"refs/molexp/runs/{run.id}" in shown
