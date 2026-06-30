"""Low-frequency git checkpoint cadence + historical materialization.

Spec: workspace-git-projection-04-wire. git is a LOW-FREQUENCY checkpoint:
commits track *settled executions*, never the high-frequency workspace write
stream. The auto-checkpoint is opt-in by the projection DB's existence and
fires exactly once per settled execution in ``run_lifecycle.exit()``. A
historical execution is materialized into a scratch worktree, never checked out
into the live, molexp-managed workspace.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.git import ensure_object_db
from molexp.workspace import Workspace
from molexp.workspace.git_projection import (
    checkpoint,
    default_object_db_path,
    materialize_run,
)


def _commit_lines(ws: Workspace, run_id: str, fmt: str = "%H") -> list[str]:
    db = default_object_db_path(ws)
    out = subprocess.run(
        ["git", "-C", str(db), "log", f"--format={fmt}", f"refs/molexp/runs/{run_id}"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def _commits(ws: Workspace, run_id: str) -> int:
    return len(_commit_lines(ws, run_id))


async def _enable_projection(ws: Workspace) -> None:
    """Opt in by initialising the object DB (existence == enabled)."""
    await ensure_object_db(default_object_db_path(ws))


def _new_run(ws_root: Path):
    ws = Workspace(root=ws_root, name="Lab")
    exp = ws.add_project("demo").add_experiment("baseline", params={})
    return ws, exp.add_run(params={"seed": 0})


# ── (a) frequency contract ───────────────────────────────────────────────────


class TestCheckpointCadence:
    async def test_n_writes_between_executions_produce_zero_commits(self, tmp_path):
        ws, run = _new_run(tmp_path / "lab")
        await _enable_projection(ws)

        with run.start() as ctx:
            ctx.artifact.save("m.json", {"v": 1})
        assert _commits(ws, run.id) == 1  # one settled execution → one commit

        # N high-frequency workspace writes that are NOT execution settles.
        for _ in range(5):
            run.update_ops(lambda s: s.model_copy(update={"heartbeat_at": datetime.now(UTC)}))
        ws.add_project("noise")  # an add_folder write
        run.experiment.add_run(params={"seed": 99})  # another entity write
        assert _commits(ws, run.id) == 1  # unchanged — no settle, no commit

        with run.start():  # a second execution (rerun) settles
            pass
        assert _commits(ws, run.id) == 2  # exactly +1 per settled execution

    async def test_failed_execution_is_a_settled_commit(self, tmp_path):
        ws, run = _new_run(tmp_path / "lab")
        await _enable_projection(ws)
        with pytest.raises(ValueError, match="boom"), run.start():
            raise ValueError("boom")
        assert _commits(ws, run.id) == 1  # FAILED is a settled terminal → one commit


# ── (b) checkpoint fires at the Execution-settled boundary ───────────────────


class TestSettleBoundary:
    async def test_no_checkpoint_until_projection_enabled(self, tmp_path):
        ws, run = _new_run(tmp_path / "lab")
        # NOT enabled → settle creates no DB and no commits (opt-in by existence).
        with run.start():
            pass
        assert not (default_object_db_path(ws) / "HEAD").exists()
        assert _commits(ws, run.id) == 0

    async def test_settle_commit_is_parent_linked(self, tmp_path):
        ws, run = _new_run(tmp_path / "lab")
        await _enable_projection(ws)
        with run.start():  # attempt 1 → root commit
            pass
        with run.start():  # attempt 2 (rerun) → new commit, parent-linked
            pass
        lines = _commit_lines(ws, run.id, fmt="%H %P")
        assert len(lines) == 2
        tip_parents = lines[0].split()[1:]
        root_hash = lines[1].split()[0]
        assert root_hash in tip_parents  # the tip attempt links back to the prior one


# ── (d) historical materialization into a scratch worktree ───────────────────


class TestHistoricalMaterialization:
    async def test_materialize_run_into_scratch_not_live_workspace(self, tmp_path):
        ws, run = _new_run(tmp_path / "lab")
        with run.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1})
        await checkpoint(ws)  # build refs/molexp/runs/<id>

        scratch = tmp_path / "scratch"
        await materialize_run(ws, run.id, scratch)

        # The historical run state is materialized into the scratch dir …
        assert (scratch / "run.json").exists()
        assert (scratch / "artifacts" / "metrics.json").exists()
        # … and the scratch dir is OUTSIDE the live workspace tree.
        ws_root = Path(str(ws.root))
        assert ws_root not in scratch.parents
        # The live run directory was never checked out into (no stray .git file).
        assert not (Path(str(run.run_dir)) / ".git").exists()
