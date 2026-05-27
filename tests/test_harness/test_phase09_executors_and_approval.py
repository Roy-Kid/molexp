"""Phase-9 tests: CommandSpec/CommandResult schemas + DryRunExecutor +
LocalExecutor + ApprovalGate stage."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

# ============================================================ schemas


def _ref():
    from molexp.harness.schemas.artifact import ArtifactRef

    return ArtifactRef(
        id="r0123456",
        kind="stdout",
        uri="file:///tmp/r",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="x",
    )


def test_command_spec_defaults_and_round_trip() -> None:
    from molexp.harness.schemas.command import CommandSpec

    s = CommandSpec(cmd=["echo", "hi"], cwd="/tmp")
    # None default → executor inherits parent env. Explicit empty dict
    # would mean "run with empty env". See schema docstring.
    assert s.env is None
    assert s.timeout_s == 3600
    assert s.expected_outputs == []
    assert s.metadata == {}
    s2 = CommandSpec.model_validate_json(s.model_dump_json())
    assert s2 == s


def test_command_spec_frozen() -> None:
    from molexp.harness.schemas.command import CommandSpec

    s = CommandSpec(cmd=["x"], cwd="/")
    with pytest.raises(ValidationError):
        s.cwd = "/etc"  # type: ignore[misc]


def test_command_result_round_trip() -> None:
    from molexp.harness.schemas.command import CommandResult

    r = CommandResult(
        exit_code=0,
        started_at=datetime(2026, 5, 26, tzinfo=UTC),
        ended_at=datetime(2026, 5, 26, tzinfo=UTC),
        stdout_artifact=_ref(),
        stderr_artifact=_ref(),
    )
    assert r.output_artifacts == []
    assert r.metadata == {}
    r2 = CommandResult.model_validate_json(r.model_dump_json())
    assert r2 == r


# ============================================================ DryRunExecutor


@pytest.fixture()
def artifact_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


def test_dry_run_executor_satisfies_protocol() -> None:
    from molexp.harness.executors.dry_run import DryRunExecutor
    from molexp.harness.executors.executor import Executor

    assert isinstance(DryRunExecutor(), Executor)


def test_dry_run_executor_returns_success(artifact_store) -> None:
    from molexp.harness.executors.dry_run import DryRunExecutor
    from molexp.harness.schemas.command import CommandSpec

    result = asyncio.run(
        DryRunExecutor().execute(
            CommandSpec(cmd=["nope"], cwd="/tmp"),
            artifact_store=artifact_store,
        )
    )
    assert result.exit_code == 0
    assert result.metadata["dry_run"] == "true"
    # stdout/stderr artifacts persisted
    assert artifact_store.get_ref(result.stdout_artifact.id).kind == "stdout"
    assert artifact_store.get_ref(result.stderr_artifact.id).kind == "stderr"


# ============================================================ LocalExecutor


def test_local_executor_runs_command(artifact_store, tmp_path: Path) -> None:
    from molexp.harness.executors.local import LocalExecutor
    from molexp.harness.schemas.command import CommandSpec

    result = asyncio.run(
        LocalExecutor().execute(
            CommandSpec(
                cmd=[sys.executable, "-c", "print('hello')"],
                cwd=str(tmp_path),
            ),
            artifact_store=artifact_store,
        )
    )
    assert result.exit_code == 0
    stdout_bytes = artifact_store.get(result.stdout_artifact.id)
    assert b"hello" in stdout_bytes


def test_local_executor_captures_exit_code(artifact_store, tmp_path: Path) -> None:
    from molexp.harness.executors.local import LocalExecutor
    from molexp.harness.schemas.command import CommandSpec

    result = asyncio.run(
        LocalExecutor().execute(
            CommandSpec(
                cmd=[sys.executable, "-c", "import sys; sys.exit(7)"],
                cwd=str(tmp_path),
            ),
            artifact_store=artifact_store,
        )
    )
    assert result.exit_code == 7


def test_local_executor_collects_expected_outputs(artifact_store, tmp_path: Path) -> None:
    """expected_outputs that exist after the command runs are persisted."""
    from molexp.harness.executors.local import LocalExecutor
    from molexp.harness.schemas.command import CommandSpec

    script = "import pathlib; pathlib.Path('out.txt').write_text('payload')"
    result = asyncio.run(
        LocalExecutor().execute(
            CommandSpec(
                cmd=[sys.executable, "-c", script],
                cwd=str(tmp_path),
                expected_outputs=["out.txt", "missing.txt"],
            ),
            artifact_store=artifact_store,
        )
    )
    assert result.exit_code == 0
    # Only the existing output is hashed + ingested.
    assert len(result.output_artifacts) == 1
    assert result.output_artifacts[0].kind == "output_file"
    # The missing one is recorded for downstream audit.
    assert result.metadata.get("missing_outputs") == "missing.txt"


def test_local_executor_refuses_path_traversal(artifact_store, tmp_path: Path) -> None:
    """expected_outputs that resolve outside cwd are rejected, not ingested."""
    from molexp.harness.executors.local import LocalExecutor
    from molexp.harness.schemas.command import CommandSpec

    # Pre-create a file OUTSIDE cwd that an attacker might want to slurp in.
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    cwd = tmp_path / "work"
    cwd.mkdir()
    try:
        result = asyncio.run(
            LocalExecutor().execute(
                CommandSpec(
                    cmd=[sys.executable, "-c", "pass"],
                    cwd=str(cwd),
                    expected_outputs=["../outside.txt"],
                ),
                artifact_store=artifact_store,
            )
        )
        assert result.output_artifacts == []
        assert result.metadata.get("escaped_outputs") == "../outside.txt"
    finally:
        outside.unlink(missing_ok=True)


def test_local_executor_timeout(artifact_store, tmp_path: Path) -> None:
    from molexp.harness.executors.local import LocalExecutor
    from molexp.harness.schemas.command import CommandSpec

    result = asyncio.run(
        LocalExecutor().execute(
            CommandSpec(
                cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=str(tmp_path),
                timeout_s=1,
            ),
            artifact_store=artifact_store,
        )
    )
    assert result.exit_code == -1
    assert result.metadata.get("timeout") == "true"


# ============================================================ ApprovalGate


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-gate",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


def _req(intent: str = "full_execution"):
    from molexp.harness.schemas.approval import ApprovalRequest

    return ApprovalRequest(
        id=f"req-{intent}",
        intent=intent,  # type: ignore[arg-type]
        reason="x",
        triggered_by_policy="require_for_" + intent,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )


def _decision(req, *, granted: bool):
    from molexp.harness.schemas.approval import ApprovalDecision

    return ApprovalDecision(
        request_id=req.id,
        granted=granted,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
    )


def test_approval_gate_name_and_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.approval_gate import ApprovalGate

    assert ApprovalGate.name == "approval_gate"
    assert issubclass(ApprovalGate, Stage)


def test_approval_gate_passes_when_all_granted(ctx) -> None:
    from molexp.harness.stages.approval_gate import ApprovalGate

    r1 = _req("hpc_submission")
    r2 = _req("full_execution")
    stage = ApprovalGate(
        decisions=[
            (r1, _decision(r1, granted=True)),
            (r2, _decision(r2, granted=True)),
        ]
    )
    ref = asyncio.run(stage.run(ctx))
    assert ref.kind == "analysis_result"


def test_approval_gate_raises_on_any_rejected(ctx) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.approval_gate import ApprovalGate

    r1 = _req("hpc_submission")
    r2 = _req("full_execution")
    stage = ApprovalGate(
        decisions=[
            (r1, _decision(r1, granted=True)),
            (r2, _decision(r2, granted=False)),
        ]
    )
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx))
    assert "full_execution" in str(exc.value)


def test_approval_gate_records_decision_events_before_failing(ctx) -> None:
    """Every decision MUST land on the event log even when the gate aborts.

    Regression: the previous implementation raised before recording, so
    the audit row for the rejection that aborted the run was missing.
    """
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.approval_gate import ApprovalGate

    r1 = _req("hpc_submission")
    r2 = _req("full_execution")
    stage = ApprovalGate(
        decisions=[
            (r1, _decision(r1, granted=True)),
            (r2, _decision(r2, granted=False)),
        ]
    )
    with pytest.raises(StageExecutionError):
        asyncio.run(stage.run(ctx))
    timeline = ctx.event_log.list_events(ctx.run_id)
    types = [e.type for e in timeline]
    # Both decisions present; the granted one BEFORE the rejected one.
    assert "approval_granted" in types
    assert "approval_rejected" in types


def test_approval_gate_summary_carries_subject_artifact_ids(ctx) -> None:
    """``subject_artifact_ids`` flows to ``parent_ids`` on the summary."""
    from molexp.harness.stages.approval_gate import ApprovalGate

    parent = ctx.artifact_store.put_json(
        kind="bound_workflow", obj={"id": "bw"}, created_by="seed", parent_ids=[]
    )
    r1 = _req("hpc_submission")
    stage = ApprovalGate(
        decisions=[(r1, _decision(r1, granted=True))],
        subject_artifact_ids=[parent.id],
    )
    ref = asyncio.run(stage.run(ctx))
    assert parent.id in ref.parent_ids


# ============================================================ surface


def test_phase09_public_surface() -> None:
    from molexp.harness import (  # noqa: F401
        ApprovalGate,
        CommandResult,
        CommandSpec,
        DryRunExecutor,
        Executor,
        LocalExecutor,
    )
