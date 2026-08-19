"""Tests for ``RunContext`` — the run-execution context manager and its
typed asset accessors (``LogAccessor`` /
``CheckpointAccessor``) exposed on the facade.

Scope is the RunContext surface only: lifecycle status resolution, in-context
result/artifact/log/checkpoint I/O, working-dir guards, and the sync/async
context-manager protocols. Manifest *scanning* (``scan_assets``) is owned by
``test_asset_scan`` / ``test_assets``; failure-recovery / no-op resolution by
``test_run_lifecycle_recovery``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.assets import ArtifactAsset, CheckpointAsset
from molexp.workspace.run import RunStatus


class TestRunContextLifecycle:
    def test_enter_sets_running(self, run):
        with run.start():
            assert run.status == "running"

    def test_clean_exit_marks_succeeded(self, run):
        with run.start():
            pass
        assert run.status == RunStatus.SUCCEEDED

    def test_exception_marks_failed_and_records_error(self, experiment):
        run = experiment.add_run()
        with pytest.raises(ValueError), run.start():
            raise ValueError("boom")
        assert run.status == RunStatus.FAILED
        assert run.metadata.error is not None
        assert run.metadata.error.type == "ValueError"
        assert run.metadata.error.message == "boom"

    def test_exception_writes_error_txt_trace(self, experiment):
        """The exception-propagation exit path lands a physical ``error.txt``
        trace under the execution dir (distinct from the engine-swallowed path
        owned by ``test_run_lifecycle_recovery``)."""
        run = experiment.add_run()
        ctx_ref: dict[str, object] = {}
        with pytest.raises(RuntimeError), run.start() as ctx:
            ctx_ref["ctx"] = ctx
            raise RuntimeError("detailed error")
        ctx = ctx_ref["ctx"]
        error_txt = ctx.run_dir / "executions" / ctx._execution_id / "error.txt"
        assert error_txt.exists()
        assert "RuntimeError" in error_txt.read_text()


class TestRunContextResults:
    def test_set_result_then_get_result_round_trips(self, run):
        with run.start() as ctx:
            ctx.set_result("acc", 0.95)
            assert ctx.get_result("acc") == 0.95


class TestRegisterArtifact:
    def test_register_artifact_writes_and_returns_readable_asset(self, run):
        with run.start() as ctx:
            asset = ctx.register_artifact({"key": "value"}, name="data.json")
            assert isinstance(asset, ArtifactAsset)
            assert asset.absolute_path(ctx.run_dir).exists()
            assert asset.read_json(ctx.run_dir) == {"key": "value"}

    def test_register_artifact_from_path_defaults_name(self, run):
        with run.start() as ctx:
            src = ctx.workdir / "report.txt"
            src.write_text("ok")
            asset = ctx.register_artifact(src)
            assert asset.name == "report.txt"
            assert (ctx.run_dir / "artifacts" / "report.txt").read_text() == "ok"

    def test_register_artifact_memory_payload_requires_name(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="name is required"):
            ctx.register_artifact({"key": "value"})

    def test_register_artifact_stamps_producer_run_and_execution_id(self, run):
        with run.start() as ctx:
            asset = ctx.register_artifact({"a": 1}, name="m.json")
            assert asset.producer is not None
            assert asset.producer.run_id == run.id
            assert asset.producer.execution_id == ctx._execution_id

    def test_register_metric_writes_wal(self, run):
        with run.start() as ctx:
            ctx.register_metric("score", 0.87, step=1)
        wal = run.run_dir / "metrics.mlp.jsonl"
        assert wal.exists()
        assert "score" in wal.read_text()


class TestLogAccessor:
    def test_append_then_tail_returns_lines(self, run):
        with run.start() as ctx:
            log = ctx.log("train")
            log.append("epoch 1")
            log.append("epoch 2")
            assert log.tail() == ["epoch 1", "epoch 2"]


class TestCheckpointAccessor:
    def test_checkpoint_saves_and_loads_payload(self, run):
        with run.start() as ctx:
            asset = ctx.checkpoint("mid-run", data={"step": 5})
            assert isinstance(asset, CheckpointAsset)
            assert asset.ckpt_id.startswith("ckpt_")
            assert asset.absolute_path(ctx.run_dir).exists()
            assert asset.load(ctx.run_dir)["data"] == {"step": 5}

    def test_checkpoints_chain_parent_ids(self, run):
        with run.start() as ctx:
            first = ctx.checkpoint("a", data={"s": 1})
            second = ctx.checkpoint("b", data={"s": 2})
            assert first.parent_ckpt_id is None
            assert second.parent_ckpt_id == first.ckpt_id


class TestGetDataDir:
    def test_fallback_creates_missing_dir(self, run):
        with run.start() as ctx:
            data_dir = ctx.get_data_dir("nonexistent", fallback="data/qm9")
            assert data_dir.is_dir()
            assert isinstance(data_dir, Path)

    def test_missing_without_fallback_raises(self, run):
        with run.start() as ctx, pytest.raises(FileNotFoundError, match="not found"):
            ctx.get_data_dir("nonexistent")


class TestAsyncRunContext:
    """``async with run.start()`` protocol + the ``with run as ctx`` sugar."""

    @pytest.mark.asyncio
    async def test_async_with_start_marks_running_then_succeeded(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        run = ws.add_project(name="p").add_experiment(name="e").add_run()
        async with run.start() as ctx:
            assert ctx.run_dir.exists()
            assert run.status == "running"
        assert run.status == RunStatus.SUCCEEDED

    def test_run_as_context_manager_sugar_succeeds(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        run = ws.add_project(name="p").add_experiment(name="e").add_run()
        with run as ctx:
            assert ctx.run_dir.exists()
        assert run.status == RunStatus.SUCCEEDED


class TestRunContextWorkdir:
    def test_workdir_is_under_execution_work(self, run):
        with run.start() as ctx:
            d = ctx.workdir
            assert d.is_dir()
            assert d.name == "work"
            assert d.relative_to(ctx.run_dir).parts[0] == "executions"

    def test_requires_active_execution(self, run):
        ctx = run.start()  # constructed but not entered → no execution yet
        with pytest.raises(RuntimeError, match="active execution"):
            _ = ctx.workdir

    def test_removed_spellings_are_gone(self, run):
        with run.start() as ctx:
            assert not hasattr(ctx, "work_dir")
            assert not hasattr(ctx, "artifact")
            assert not hasattr(ctx, "folder")
            assert not hasattr(ctx, "register_asset")

    def test_platform_json_writes_go_through_filestore(self, run, monkeypatch):
        from molexp.workspace.file_store import FileStore

        seen: list[str] = []
        orig = FileStore.put

        def spy(self, relpath, data):
            seen.append(Path(relpath).as_posix())
            return orig(self, relpath, data)

        monkeypatch.setattr(FileStore, "put", spy)
        with run.start():
            pass
        assert "run.json" in seen
        assert "ops/run.json" in seen

    def test_register_catalogs_without_rewriting(self, run):
        with run.start() as ctx:
            dest = ctx.files.put("artifacts/keep.txt", "payload")
            before = dest.stat().st_mtime
            asset = ctx.register(dest, kind="artifact")
            assert dest.read_text() == "payload"
            assert dest.stat().st_mtime == before
            assert asset.name == "keep.txt"
