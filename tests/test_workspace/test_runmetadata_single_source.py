"""``RunMetadata`` is identity/provenance only — hot state lives in ``_ops`` (wsokf-10).

wsokf-07 made ``_ops/run.json`` (:class:`RunOpsState`) the *read* source for a
Run's hot machine state while keeping the same fields mirrored in
``RunMetadata`` (``run.json``). wsokf-10 completes the single-source-of-truth
law: the hot-state fields (``status`` / ``finished_at`` / ``execution_history`` /
``labels``) are **removed** from ``RunMetadata`` so ``_ops/run.json`` is the
sole source. ``run.json`` keeps identity / provenance (params, config, profile,
workflow snapshot, source snapshot, target, executor_info) plus the terminal
``error`` diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

from molexp.workspace.models import RunMetadata
from molexp.workspace.run import Run
from molexp.workspace.run_ops import RunOpsState


def _read_run_json(run: Run) -> dict:
    return json.loads(Path(str(run.run_dir / "run.json")).read_text())


def _read_ops_json(run: Run) -> dict:
    return json.loads(Path(str(run.run_dir / "_ops" / "run.json")).read_text())


# ── (a) field removal + extra=ignore ────────────────────────────────────────


class TestFieldRemoval:
    def test_hot_state_fields_are_not_model_fields(self) -> None:
        fields = RunMetadata.model_fields
        for removed in ("status", "finished_at", "execution_history", "labels"):
            assert removed not in fields, f"{removed!r} must be removed from RunMetadata"

    def test_identity_fields_remain(self) -> None:
        fields = RunMetadata.model_fields
        for kept in (
            "id",
            "parameters",
            "created_at",
            "config",
            "config_hash",
            "profile",
            "workflow_snapshot",
            "workflow_id",
            "workflow_version",
            "source_snapshot",
            "script",
            "submit_cwd",
            "target",
            "executor_info",
            "error",
        ):
            assert kept in fields, f"{kept!r} must remain on RunMetadata"

    def test_extra_is_ignore(self) -> None:
        assert RunMetadata.model_config["extra"] == "ignore"

    def test_legacy_run_json_with_removed_keys_round_trips(self) -> None:
        legacy = {
            "id": "r",
            "status": "failed",
            "finished_at": "2026-01-01T00:00:00",
            "labels": {"pid": "1", "host": "h", "heartbeat": "2026-01-01T00:00:00"},
            "execution_history": [
                {
                    "execution_id": "exec-r",
                    "started_at": "2026-01-01T00:00:00",
                    "status": "failed",
                }
            ],
        }
        meta = RunMetadata.model_validate(legacy)
        assert meta.id == "r"
        for removed in ("status", "finished_at", "execution_history", "labels"):
            assert not hasattr(meta, removed)


# ── (d) the dual-write mirror is gone ───────────────────────────────────────


class TestMirrorDeleted:
    def test_run_has_no_mirror_method(self) -> None:
        assert not hasattr(Run, "_mirror_hot_state_to_ops")


# ── (b) cancel writes hot-state to _ops only ────────────────────────────────


class TestCancelOpsOnly:
    def test_cancel_persists_only_to_ops(self, run) -> None:
        run.materialize()
        run.cancel()

        ops = _read_ops_json(run)
        assert ops["status"] == "cancelled"
        assert ops["finished_at"] is not None
        assert ops["owner_pid"] is None
        assert ops["owner_host"] is None

        assert run.status == "cancelled"
        assert run.read_ops().status.value == "cancelled"

        on_disk = _read_run_json(run)
        for removed in ("status", "finished_at", "labels", "execution_history"):
            assert removed not in on_disk, f"run.json must not carry {removed!r}"


# ── (c) full lifecycle writes hot-state to _ops, error to run.json ──────────


class TestLifecycleOpsOnly:
    def test_success_lifecycle_writes_ops_only(self, run) -> None:
        with run.start():
            running = _read_ops_json(run)
            assert running["status"] == "running"
            assert running["owner_pid"] is not None
            assert running["heartbeat_at"] is not None
            assert running["current_execution_id"] is not None

        after = run.read_ops()
        assert after.status.value == "succeeded"
        assert after.finished_at is not None
        assert after.owner_pid is None
        assert [r.status for r in after.executions] == ["succeeded"]

        on_disk = _read_run_json(run)
        for removed in ("status", "finished_at", "labels", "execution_history"):
            assert removed not in on_disk

    def test_failed_lifecycle_writes_error_to_run_json_and_status_to_ops(self, run) -> None:
        try:
            with run.start():
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert run.read_ops().status.value == "failed"
        on_disk = _read_run_json(run)
        assert "status" not in on_disk
        # error stays in run.json (identity/diagnostic).
        assert on_disk["error"] is not None
        assert on_disk["error"]["type"] == "RuntimeError"
        assert run.metadata.error is not None
        assert run.metadata.error.message == "boom"


# ── error stays out of RunOpsState ──────────────────────────────────────────


class TestErrorNotInOps:
    def test_run_ops_state_has_no_error_field(self) -> None:
        assert "error" not in RunOpsState.model_fields


# ── _update_metadata rejects hot-state keys ─────────────────────────────────


class TestUpdateMetadataGuards:
    def test_update_metadata_rejects_hot_state(self, run) -> None:
        run.materialize()
        for key in ("status", "finished_at", "execution_history", "labels"):
            try:
                run._update_metadata(**{key: None})
            except ValueError:
                continue
            raise AssertionError(f"_update_metadata must reject hot-state key {key!r}")

    def test_update_metadata_accepts_identity(self, run) -> None:
        run.materialize()
        run._update_metadata(profile="p1")
        assert run.metadata.profile == "p1"
