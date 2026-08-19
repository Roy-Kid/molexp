"""RegisterArtifact / register_* promotion — task outputs become run products.

Locks the user-facing vocabulary: write under ``ctx.workdir``, publish with
``register_artifact`` / ``RegisterArtifact``. Downstream binds the *run*
artifact path, never the ``.materialize`` scratch path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import (
    RegisterArtifact,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)
from molexp.workflow.cache import Caching
from molexp.workspace import Workspace


def _new_run(tmp_path: Path, params: dict | None = None):
    ws = Workspace(tmp_path / "lab")
    return ws.add_project(name="p").add_experiment(name="e").add_run(params=params or {})


class TestPromoteRegisterArtifact:
    @pytest.mark.asyncio
    async def test_returned_marker_copies_into_run_artifacts(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}
        wf = WorkflowCompiler(name="reg")

        @wf.task
        async def export(ctx: TaskContext) -> dict:
            out = ctx.workdir / "system.data"
            out.write_text("atoms")
            captured["scratch"] = out
            return {"data": RegisterArtifact(out, mime="chemical/x-lammps-data")}

        run = _new_run(tmp_path)
        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.status == "succeeded"
        promoted = Path(result.outputs["export"]["data"])
        assert promoted == Path(run.run_dir) / "artifacts" / "system.data"
        scratch = Path(captured["scratch"])
        assert scratch.parent.name == "export"
        assert "work" in scratch.parts
        assert "executions" in scratch.parts
        assert promoted.read_text() == "atoms"
        assert promoted != captured["scratch"]
        found = run.assets.query(producer_task="export", kind="artifact")
        assert any(a.name == "system.data" for a in found)

    @pytest.mark.asyncio
    async def test_scalar_marker_return_is_promoted(self, tmp_path: Path) -> None:
        wf = WorkflowCompiler(name="scalar")

        @wf.task
        async def export(ctx: TaskContext) -> RegisterArtifact:
            out = ctx.workdir / "a.txt"
            out.write_text("x")
            return RegisterArtifact(out)

        run = _new_run(tmp_path)
        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert Path(result.outputs["export"]) == Path(run.run_dir) / "artifacts" / "a.txt"

    @pytest.mark.asyncio
    async def test_ctx_register_artifact_in_return_dict(self, tmp_path: Path) -> None:
        wf = WorkflowCompiler(name="verb")

        @wf.task
        async def export(ctx: TaskContext) -> dict:
            out = ctx.workdir / "report.txt"
            out.write_text("ok")
            return {"report": ctx.register_artifact(out, mime="text/plain")}

        run = _new_run(tmp_path)
        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert (
            Path(result.outputs["export"]["report"])
            == Path(run.run_dir) / "artifacts" / "report.txt"
        )

    @pytest.mark.asyncio
    async def test_pending_only_still_registers(self, tmp_path: Path) -> None:
        wf = WorkflowCompiler(name="pending")

        @wf.task
        async def export(ctx: TaskContext) -> int:
            out = ctx.workdir / "side.txt"
            out.write_text("side")
            ctx.register_artifact(out)
            return 1

        run = _new_run(tmp_path)
        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.outputs["export"] == 1
        assert (Path(run.run_dir) / "artifacts" / "side.txt").read_text() == "side"

    @pytest.mark.asyncio
    async def test_register_metric_appends_wal(self, tmp_path: Path) -> None:
        wf = WorkflowCompiler(name="metric")

        @wf.task
        async def measure(ctx: TaskContext) -> dict:
            return {"n": ctx.register_metric("n_atoms", 42.0)}

        run = _new_run(tmp_path)
        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.outputs["measure"]["n"] == 42.0
        wal = Path(run.run_dir) / "metrics.mlp.jsonl"
        assert wal.exists()
        assert "n_atoms" in wal.read_text()

    @pytest.mark.asyncio
    async def test_pending_only_reregisters_on_cache_hit(self, tmp_path: Path) -> None:
        counters = {"export": 0}
        wf = WorkflowCompiler(name="cached-reg")

        @wf.task
        async def export(ctx: TaskContext) -> int:
            counters["export"] += 1
            out = ctx.workdir / "cached.txt"
            out.write_text("cached")
            ctx.register_artifact(out)
            return 7

        compiled = wf.compile()
        cache = Caching(store_dir=tmp_path / "shared-cache")

        # Same workspace so content-hash re-registration can see the first run's
        # bytes; two experiments keep the runs distinct.
        proj = Workspace(tmp_path / "lab").add_project(name="p")
        run1 = proj.add_experiment(name="e1").add_run()
        with run1.start() as ctx1:
            r1 = await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache)
        run2 = proj.add_experiment(name="e2").add_run()
        with run2.start() as ctx2:
            r2 = await WorkflowRuntime().execute(compiled, run_context=ctx2, cache=cache)

        assert counters["export"] == 1
        assert r1.outputs["export"] == r2.outputs["export"] == 7
        found = run2.assets.query(producer_task="export", kind="artifact")
        assert any(a.name == "cached.txt" for a in found)

    def test_task_context_workdir_has_no_ambient_authority(self) -> None:
        ctx = TaskContext(inputs={}, workdir=Path("/tmp"))
        assert not hasattr(ctx.workdir, "folder")
        assert not hasattr(ctx.workdir, "artifact")
        assert not hasattr(ctx.workdir, "run")
