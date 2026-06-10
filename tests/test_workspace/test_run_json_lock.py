"""Lost-update protection for the ``run.json`` read-modify-write cycle.

``run.json`` is written by several uncoordinated writers (server process,
foreground CLI, detached workers). ``Run._update_metadata`` must reload
the on-disk state and apply the partial update under an advisory file
lock, so two writers touching *distinct* fields never drop each other's
updates.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace._file_lock import FileLockTimeoutError, file_lock


def _read_run_json(run) -> dict:
    return json.loads(Path(str(run.run_dir / "run.json")).read_text())


def _second_handle(tmp_path, run):
    """Load an independent Run handle for the same on-disk run."""
    ws = Workspace(root=tmp_path, name="Test Lab")
    project = ws.list_projects()[0]
    experiment = project.list_experiments()[0]
    other = next(r for r in experiment.list_runs() if r.id == run.id)
    assert other is not run
    return other


class TestFileLock:
    def test_exclusive_between_holders(self, tmp_path) -> None:
        lock_path = tmp_path / "x.lock"
        order: list[str] = []
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with file_lock(lock_path):
                order.append("holder-in")
                entered.set()
                release.wait(timeout=5.0)
                order.append("holder-out")

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=5.0)
        release.set()
        with file_lock(lock_path, timeout=5.0):
            order.append("second-in")
        t.join(timeout=5.0)
        assert order == ["holder-in", "holder-out", "second-in"]

    def test_timeout_raises_clear_error(self, tmp_path) -> None:
        lock_path = tmp_path / "x.lock"
        entered = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with file_lock(lock_path):
                entered.set()
                release.wait(timeout=5.0)

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=5.0)
        try:
            with (
                pytest.raises(FileLockTimeoutError, match=r"x\.lock"),
                file_lock(lock_path, timeout=0.2),
            ):
                pass
        finally:
            release.set()
            t.join(timeout=5.0)

    def test_degrades_without_fcntl(self, tmp_path, monkeypatch) -> None:
        import molexp.workspace._file_lock as fl

        monkeypatch.setattr(fl, "_HAS_FCNTL", False)
        with file_lock(tmp_path / "x.lock"):
            pass  # no-op, no error

    def test_degrades_on_unwritable_path(self, tmp_path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        # Parent "directory" is a file → lock creation fails → no-op.
        with file_lock(blocker / "x.lock"):
            pass


class TestRunMetadataRmw:
    def test_update_reloads_other_writers_fields(self, tmp_path, run) -> None:
        run.materialize()
        other = _second_handle(tmp_path, run)

        run._update_metadata(script="train.py")
        other._update_metadata(target="cluster-a")  # stale handle, distinct field

        data = _read_run_json(run)
        assert data["script"] == "train.py"  # not clobbered by the stale handle
        assert data["target"] == "cluster-a"

    def test_concurrent_writers_lose_no_updates(self, tmp_path, run) -> None:
        run.materialize()
        other = _second_handle(tmp_path, run)
        iterations = 40
        errors: list[BaseException] = []

        def writer(handle, field: str, prefix: str) -> None:
            try:
                for i in range(iterations):
                    handle._update_metadata(**{field: f"{prefix}{i}"})
                    time.sleep(0)  # encourage interleaving
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=(run, "script", "s"))
        t2 = threading.Thread(target=writer, args=(other, "target", "t"))
        t1.start()
        t2.start()
        t1.join(timeout=60.0)
        t2.join(timeout=60.0)
        assert not errors, errors

        data = _read_run_json(run)
        assert data["script"] == f"s{iterations - 1}"
        assert data["target"] == f"t{iterations - 1}"

    def test_lock_sidecar_lives_next_to_run_json(self, tmp_path, run) -> None:
        run.materialize()
        run._update_metadata(script="x.py")
        assert Path(str(run.run_dir / "run.json.lock")).exists()
