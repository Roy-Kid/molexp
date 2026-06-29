"""Tests for RunContext lifecycle and typed asset accessors."""

from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.assets import ArtifactAsset, CheckpointAsset
from molexp.workspace.run import RunStatus


class TestRunContextLifecycle:
    def test_enter_sets_running(self, run):
        with run.start():
            assert run.status == "running"

    def test_exit_success(self, run):
        with run.start():
            pass
        assert run.status == RunStatus.SUCCEEDED

    def test_exit_failure(self, experiment):
        run = experiment.add_run()
        try:
            with run.start():
                raise ValueError("boom")
        except ValueError:
            pass
        assert run.status == RunStatus.FAILED
        assert run.metadata.error is not None
        assert run.metadata.error.type == "ValueError"
        assert run.metadata.error.message == "boom"

    def test_finished_at_set(self, run):
        with run.start():
            pass
        assert run.finished_at is not None

    def test_creates_work_dir(self, run):
        with run.start() as ctx:
            assert ctx.work_dir.exists()


class TestRunContextResults:
    def test_set_and_get_result(self, run):
        with run.start() as ctx:
            ctx.set_result("acc", 0.95)
            assert ctx.get_result("acc") == 0.95

    def test_missing_result_returns_none(self, run):
        with run.start() as ctx:
            assert ctx.get_result("missing") is None


class TestArtifactAccessor:
    def test_save_dict_registers_asset(self, run):
        with run.start() as ctx:
            asset = ctx.artifact.save("data.json", {"key": "value"})
            assert isinstance(asset, ArtifactAsset)
            path = asset.absolute_path(ctx.work_dir)
            assert path.exists()
            assert asset.read_json(ctx.work_dir) == {"key": "value"}

    def test_save_bytes(self, run):
        with run.start() as ctx:
            asset = ctx.artifact.save("binary.bin", b"\x00\x01\x02")
            assert asset.read_bytes(ctx.work_dir) == b"\x00\x01\x02"
            assert asset.size == 3

    def test_save_text(self, run):
        with run.start() as ctx:
            asset = ctx.artifact.save("log.txt", "hello world")
            path = asset.absolute_path(ctx.work_dir)
            assert path.read_text() == "hello world"

    def test_producer_captures_run_id(self, run):
        with run.start() as ctx:
            asset = ctx.artifact.save("m.json", {"a": 1})
            assert asset.producer is not None
            assert asset.producer.run_id == run.id
            assert asset.producer.execution_id == ctx._execution_id

    def test_asset_visible_in_catalog(self, run):
        with run.start() as ctx:
            ctx.artifact.save("m.json", {"a": 1})
        from molexp.workspace.assets import scan

        artifacts = scan.scan_assets(
            run.experiment.project.workspace.root, kind="artifact", producer_run=run.id
        )
        assert len(artifacts) == 1
        assert artifacts[0].name == "m.json"


class TestLogAccessor:
    def test_append_and_tail(self, run):
        with run.start() as ctx:
            log = ctx.log("train")
            log.append("epoch 1")
            log.append("epoch 2")
            assert log.tail() == ["epoch 1", "epoch 2"]

    def test_returns_same_bound_log(self, run):
        with run.start() as ctx:
            assert ctx.log("x") is ctx.log("x")

    def test_log_asset_registered(self, run):
        with run.start() as ctx:
            ctx.log("custom").append("msg")
        from molexp.workspace.assets import scan

        logs = scan.scan_assets(
            run.experiment.project.workspace.root, kind="log", producer_run=run.id
        )
        names = {a.name for a in logs}
        # "run" log is created automatically by lifecycle; "custom" by user
        assert "custom" in names
        assert "run" in names

    def test_run_log_contains_lifecycle_messages(self, run):
        with run.start() as ctx:
            run_log = ctx.log("run")
            tail = run_log.tail()
            assert any("execution started" in line for line in tail)


class TestCheckpointAccessor:
    def test_saves_and_registers(self, run):
        with run.start() as ctx:
            asset = ctx.checkpoint("mid-run", data={"step": 5})
            assert isinstance(asset, CheckpointAsset)
            assert asset.ckpt_id.startswith("ckpt_")
            path = asset.absolute_path(ctx.work_dir)
            assert path.exists()
            payload = asset.load(ctx.work_dir)
            assert payload["data"] == {"step": 5}

    def test_parent_chain(self, run):
        with run.start() as ctx:
            first = ctx.checkpoint("a", data={"s": 1})
            second = ctx.checkpoint("b", data={"s": 2})
            assert first.parent_ckpt_id is None
            assert second.parent_ckpt_id == first.ckpt_id


class TestRunContextParams:
    def test_params_shortcut(self, experiment):
        run = experiment.add_run(params={"lr": 1e-4, "batch": 32})
        with run.start() as ctx:
            assert ctx.params == {"lr": 1e-4, "batch": 32}
            assert ctx.params is ctx.run.parameters

    def test_profile_defaults_none(self, run):
        with run.start() as ctx:
            assert ctx.config.name is None
            assert run.metadata.profile is None


class TestGetDataDir:
    def test_fallback_creates_dir(self, run):
        with run.start() as ctx:
            data_dir = ctx.get_data_dir("nonexistent", fallback="data/qm9")
            assert data_dir.exists()
            assert data_dir.is_dir()
            assert isinstance(data_dir, Path)

    def test_no_fallback_raises(self, run):
        with run.start() as ctx, pytest.raises(FileNotFoundError, match="not found"):
            ctx.get_data_dir("nonexistent")


class TestErrorTraceAsset:
    def test_error_file_created_and_registered(self, experiment):
        run = experiment.add_run()
        ctx_ref = {}
        try:
            with run.start() as ctx:
                ctx_ref["ctx"] = ctx
                raise RuntimeError("detailed error")
        except RuntimeError:
            pass
        ctx = ctx_ref["ctx"]
        # Physical file
        error_txt = ctx.work_dir / "executions" / ctx._execution_id / "error.txt"
        assert error_txt.exists()
        assert "RuntimeError" in error_txt.read_text()

        # Error-trace asset recorded in the manifest
        from molexp.workspace.assets import scan

        traces = scan.scan_assets(
            run.experiment.project.workspace.root, kind="error_trace", producer_run=run.id
        )
        assert len(traces) == 1
        assert traces[0].exception_type == "RuntimeError"


# ── Async context-manager protocol + Run sugar ─────────────────────────────


class TestAsyncRunContext:
    """``async with`` support on ``RunContext`` and the ``Run`` sugar shape."""

    @pytest.mark.asyncio
    async def test_async_with_run_start(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="p").add_experiment(name="e")
        run = exp.add_run()
        async with run.start() as ctx:
            assert ctx.work_dir.exists()
            assert run.status == "running"
        assert run.status == RunStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_async_with_run_sugar(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="p").add_experiment(name="e")
        run = exp.add_run()
        async with run as ctx:
            assert ctx.work_dir.exists()
        assert run.status == RunStatus.SUCCEEDED

    def test_sync_with_run_sugar(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="p").add_experiment(name="e")
        run = exp.add_run()
        with run as ctx:
            assert ctx.work_dir.exists()
        assert run.status == RunStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_async_failure_propagates_and_records_failed(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        exp = ws.add_project(name="p").add_experiment(name="e")
        run = exp.add_run()
        with pytest.raises(ValueError, match="boom"):
            async with run.start():
                raise ValueError("boom")
        assert run.status == RunStatus.FAILED


class TestRunContextFolder:
    def test_folder_creates_under_execution(self, run):
        with run.start() as ctx:
            d = ctx.folder("scratch/CAT")
            assert d.is_dir()
            # …/runs/<run>/executions/<exec>/scratch/CAT
            assert d.parent.name == "scratch"
            assert d.parent.parent.parent.name == "executions"
            assert d.relative_to(ctx.work_dir).parts[0] == "executions"

    def test_folder_is_idempotent(self, run):
        with run.start() as ctx:
            assert ctx.folder("scratch/CAT") == ctx.folder("scratch/CAT")

    def test_folder_nested_and_distinct(self, run):
        with run.start() as ctx:
            a = ctx.folder("scratch/CAT")
            b = ctx.folder("output")
            assert a != b
            assert a.is_dir() and b.is_dir()

    def test_folder_rejects_absolute(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="relative"):
            ctx.folder("/etc")

    def test_folder_rejects_escape(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="escapes"):
            ctx.folder("../../escape")

    def test_folder_requires_active_execution(self, run):
        ctx = run.start()  # constructed but not entered → no execution yet
        with pytest.raises(RuntimeError, match="active execution"):
            ctx.folder("scratch")
