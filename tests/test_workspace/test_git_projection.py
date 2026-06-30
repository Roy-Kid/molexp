"""``molexp.workspace.git_projection`` — workspace→git projection.

Spec: workspace-git-projection-03-map. The projection is a derived,
rebuildable view of the authoritative workspace: entities map onto the real
git objects from spec 02, refs land under ``refs/molexp/*``, and a
``rebuild`` from the authoritative files reproduces byte-identical OIDs
(proving git is a projection target, never a second truth). Hot state
(``_ops/``), ``cache/`` and derived indexes are excluded; ``molexp.ids`` and
the content-addressed cache keys are never perturbed.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from molexp.git import ensure_object_db
from molexp.ids import compute_content_hash
from molexp.workspace import Workspace
from molexp.workspace.git_projection import (
    ARTIFACT_POINTER_MARKER,
    GitProjection,
)
from molexp.workspace.models import ExecutionRecord


def _git(db_path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(db_path), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def _seed_two_runs(ws_root: Path) -> tuple[Workspace, object, object]:
    """One project / one experiment / two started runs with artifacts."""
    ws = Workspace(root=ws_root, name="Lab")
    exp = ws.add_project("demo").add_experiment("baseline", params={"lr": 1e-3})
    run_a = exp.add_run(params={"seed": 0})
    with run_a.start() as ctx:
        ctx.artifact.save("metrics.json", {"loss": 0.1})
    run_b = exp.add_run(params={"seed": 1})
    with run_b.start() as ctx:
        ctx.artifact.save("metrics.json", {"loss": 0.2})
    return ws, run_a, run_b


def _append_execution(run, execution_id: str, started: datetime, finished: datetime) -> None:
    """Seed an extra settled ExecutionRecord with fixed (deterministic) dates."""

    def _add(ops):
        rec = ExecutionRecord(
            execution_id=execution_id,
            started_at=started,
            finished_at=finished,
            status="succeeded",
        )
        return ops.model_copy(update={"executions": (*ops.executions, rec)})

    run.update_ops(_add)


# ── (a) deterministic rebuild ────────────────────────────────────────────────


class TestDeterministicRebuild:
    async def test_rebuild_reproduces_byte_identical_oids(self, tmp_path):
        ws, _run_a, _run_b = _seed_two_runs(tmp_path / "lab")
        db = await ensure_object_db(tmp_path / "odb")
        proj = GitProjection(ws, db)

        first = await proj.project()
        rebuilt = await proj.rebuild()  # erases object DB + refs, re-derives

        assert rebuilt == first  # frozen dataclasses compare OIDs by value
        assert first.workspace_tree.hex  # non-empty
        assert len(first.runs) == 2

    async def test_rebuild_from_scratch_matches_a_fresh_projection(self, tmp_path):
        ws, _a, _b = _seed_two_runs(tmp_path / "lab")
        db1 = await ensure_object_db(tmp_path / "odb1")
        db2 = await ensure_object_db(tmp_path / "odb2")
        r1 = await GitProjection(ws, db1).project()
        r2 = await GitProjection(ws, db2).project()
        # Same authoritative inputs → identical OIDs in two independent DBs.
        assert r1.workspace_tree == r2.workspace_tree


# ── (b) real-git interop ─────────────────────────────────────────────────────


class TestRealGitInterop:
    async def test_git_log_walks_the_execution_dag(self, tmp_path):
        ws = Workspace(root=tmp_path / "lab", name="Lab")
        exp = ws.add_project("demo").add_experiment("baseline", params={})
        run = exp.add_run(params={"seed": 0})
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"v": 1})
        # Two more (rerun) attempts with fixed dates → a 3-commit chain.
        _append_execution(
            run, f"exec-{run.id}-2", datetime(2026, 1, 1, 10, 0, 0), datetime(2026, 1, 1, 10, 5, 0)
        )
        _append_execution(
            run, f"exec-{run.id}-3", datetime(2026, 1, 2, 10, 0, 0), datetime(2026, 1, 2, 10, 5, 0)
        )
        db = await ensure_object_db(tmp_path / "odb")
        res = await GitProjection(ws, db).project()

        pr = res.run(run.id)
        assert pr is not None
        assert pr.ref == f"refs/molexp/runs/{run.id}"
        assert len(pr.commits) == 3  # one commit per execution
        # git log over the ref walks the whole chain (tip reaches every attempt).
        subjects = _git(db.path, "log", "--format=%H", pr.ref).split()
        assert len(subjects) == 3
        assert subjects[0] == pr.commits[-1].hex  # ref points at the tip

    async def test_git_diff_surfaces_param_deltas_between_runs(self, tmp_path):
        ws, run_a, run_b = _seed_two_runs(tmp_path / "lab")
        db = await ensure_object_db(tmp_path / "odb")
        res = await GitProjection(ws, db).project()
        tree_a = res.run(run_a.id).tree
        tree_b = res.run(run_b.id).tree
        diff = _git(db.path, "diff", tree_a.hex, tree_b.hex)
        assert "run.json" in diff
        assert "seed" in diff  # the differing param surfaces in the diff


# ── (c) exclusion of hot state + derived indexes ─────────────────────────────


class TestExclusion:
    async def test_hot_state_and_indexes_never_projected(self, tmp_path):
        ws, _a, _b = _seed_two_runs(tmp_path / "lab")
        db = await ensure_object_db(tmp_path / "odb")
        res = await GitProjection(ws, db).project()
        paths = _git(db.path, "ls-tree", "-r", "--name-only", res.workspace_tree.hex).split("\n")
        joined = "\n".join(paths)
        assert "_ops" not in joined
        assert "cache" not in joined
        assert "executions" not in joined
        # The run ENTITY file is projected (identity), the index files are not.
        assert any(p.endswith("run.json") for p in paths)
        # No children-index at a container level (e.g. demo/experiment.json).
        assert not any(p.endswith("/experiment.json") or p == "project.json" for p in paths)

    async def test_run_tree_contains_whitelisted_entries(self, tmp_path):
        ws, run_a, _b = _seed_two_runs(tmp_path / "lab")
        db = await ensure_object_db(tmp_path / "odb")
        res = await GitProjection(ws, db).project()
        names = set(_git(db.path, "ls-tree", "--name-only", res.run(run_a.id).tree.hex).split())
        assert "run.json" in names
        assert "artifacts" in names
        assert "_ops" not in names
        assert "cache" not in names


# ── (d) molexp.ids + cache keys untouched ────────────────────────────────────


class TestIdentityUntouched:
    async def test_projection_does_not_mutate_authoritative_files_or_hashes(self, tmp_path):
        ws, run_a, _b = _seed_two_runs(tmp_path / "lab")
        run_json = Path(str(run_a.run_dir)) / "run.json"
        before_bytes = run_json.read_bytes()
        before_hash = compute_content_hash(run_json)

        db = await ensure_object_db(tmp_path / "odb")
        proj = GitProjection(ws, db)
        await proj.project()
        await proj.rebuild()

        # Authoritative file is byte-for-byte unchanged; no git OID written back.
        assert run_json.read_bytes() == before_bytes
        assert compute_content_hash(run_json) == before_hash
        assert "refs/molexp" not in before_bytes.decode()


# ── (e) artifact blob-vs-pointer threshold ───────────────────────────────────


class TestArtifactThreshold:
    async def test_large_artifact_is_pointer_small_is_blob(self, tmp_path):
        ws = Workspace(root=tmp_path / "lab", name="Lab")
        exp = ws.add_project("demo").add_experiment("baseline", params={})
        run = exp.add_run(params={"seed": 0})
        big = b"X" * 4096
        with run.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1})  # small → blob
            ctx.artifact.save("traj.bin", big)  # large → pointer

        db = await ensure_object_db(tmp_path / "odb")
        res = await GitProjection(ws, db, blob_threshold_bytes=64).project()
        tree = res.run(run.id).tree

        # Small artifact: the blob holds the real bytes.
        small = _git(db.path, "cat-file", "-p", f"{tree.hex}:artifacts/metrics.json")
        assert "loss" in small

        # Large artifact: the entry is a pointer (hash + size), not the bytes.
        pointer = _git(db.path, "cat-file", "-p", f"{tree.hex}:artifacts/traj.bin")
        assert pointer.startswith(ARTIFACT_POINTER_MARKER)
        assert "sha256:" in pointer
        assert str(len(big)) in pointer
        # The 4 KiB payload never entered the object DB as a blob.
        assert "XXXX" not in pointer
