"""Tests for RunContext lifecycle."""

import json
from pathlib import Path

import pytest

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
        run = experiment.run()
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
        assert run.metadata.finished_at is not None

    def test_creates_directories(self, run):
        with run.start() as ctx:
            assert ctx.work_dir.exists()
            assert ctx.artifacts_dir.exists()
            assert ctx.logs_dir.exists()


class TestRunContextResults:
    def test_set_and_get_result(self, run):
        with run.start() as ctx:
            ctx.set_result("acc", 0.95)
            assert ctx.get_result("acc") == 0.95

    def test_missing_result_returns_none(self, run):
        with run.start() as ctx:
            assert ctx.get_result("missing") is None


class TestRunContextArtifacts:
    def test_save_dict_as_json(self, run):
        with run.start() as ctx:
            path = ctx.save_artifact("data.json", {"key": "value"})
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["key"] == "value"

    def test_save_bytes(self, run):
        with run.start() as ctx:
            path = ctx.save_artifact("binary.bin", b"\x00\x01\x02")
            assert path.exists()
            assert path.read_bytes() == b"\x00\x01\x02"

    def test_save_text(self, run):
        with run.start() as ctx:
            path = ctx.save_artifact("log.txt", "hello world")
            assert path.exists()
            assert path.read_text() == "hello world"

    def test_get_artifact_path(self, run):
        with run.start() as ctx:
            expected = ctx.artifacts_dir / "file.txt"
            assert ctx.get_artifact_path("file.txt") == expected


class TestRunContextCheckpoint:
    def test_checkpoint_creates_file(self, run):
        with run.start() as ctx:
            ckpt_id = ctx.checkpoint("mid-run")
            assert ckpt_id.startswith("ckpt_")
            ckpt_dir = ctx.work_dir / ".ckpt"
            assert ckpt_dir.exists()
            assert any(ckpt_dir.iterdir())


class TestRunContextParams:
    def test_params_shortcut(self, experiment):
        run = experiment.run(parameters={"lr": 1e-4, "batch": 32})
        with run.start() as ctx:
            assert ctx.params == {"lr": 1e-4, "batch": 32}
            assert ctx.params is ctx.run.parameters

    def test_profile_defaults_none(self, run):
        with run.start() as ctx:
            assert ctx.config.name is None
            assert run.metadata.profile is None


class TestRunContextGetDataDir:
    def test_fallback_creates_dir(self, run):
        with run.start() as ctx:
            data_dir = ctx.get_data_dir("nonexistent", fallback="data/qm9")
            assert data_dir.exists()
            assert data_dir.is_dir()
            assert isinstance(data_dir, Path)

    def test_no_fallback_raises(self, run):
        with run.start() as ctx:
            with pytest.raises(FileNotFoundError, match="not found"):
                ctx.get_data_dir("nonexistent")

    def test_returns_path_type(self, run):
        with run.start() as ctx:
            result = ctx.get_data_dir("missing", fallback="data/test")
            assert isinstance(result, Path)


class TestRunContextErrorDetails:
    def test_error_log_created(self, experiment):
        run = experiment.run()
        try:
            with run.start() as ctx:
                raise RuntimeError("detailed error")
        except RuntimeError:
            pass
        # error.txt now lives in the per-execution subdir alongside workflow.json
        error_txt = ctx.work_dir / "execution" / ctx._execution_id / "error.txt"
        assert error_txt.exists(), f"error.txt not found at {error_txt}"
        content = error_txt.read_text()
        assert "RuntimeError" in content
        assert "detailed error" in content
        assert not (ctx.artifacts_dir / "error.txt").exists()
        assert not (ctx.artifacts_dir / "error.json").exists()
