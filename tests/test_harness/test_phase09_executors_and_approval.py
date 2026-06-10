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
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-gate",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
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


def _scripted_approver(verdicts: dict[str, bool]):
    """Approver answering each request from a ``request.id -> granted`` map."""
    from datetime import datetime as _datetime

    from molexp.harness.schemas.approval import ApprovalDecision

    async def approve(request):
        return ApprovalDecision(
            request_id=request.id,
            granted=verdicts[request.id],
            decided_by="alice",
            decided_at=_datetime.now(tz=UTC),
        )

    return approve


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
        requests=[r1, r2],
        approve=_scripted_approver({r1.id: True, r2.id: True}),
    )
    ref = asyncio.run(stage.run(ctx))
    assert ref.kind == "analysis_result"


def test_approval_gate_default_approver_auto_grants(ctx) -> None:
    """Without an explicit approver the gate auto-grants — at run time.

    The decision is produced when the gate runs (``decided_at`` is the
    decision moment, ``decided_by`` names the auto-approver), and recorded
    on the event log like any human decision.
    """
    from molexp.harness.stages.approval_gate import ApprovalGate

    stage = ApprovalGate(requests=[_req("final_report")])
    ref = asyncio.run(stage.run(ctx))

    assert ref.kind == "analysis_result"
    granted = [e for e in ctx.event_log.list_events(ctx.run_id) if e.type == "approval_granted"]
    assert len(granted) == 1
    assert granted[0].payload["decided_by"] == "auto-approver"


def test_approval_gate_raises_on_any_rejected(ctx) -> None:
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.approval_gate import ApprovalGate

    r1 = _req("hpc_submission")
    r2 = _req("full_execution")
    stage = ApprovalGate(
        requests=[r1, r2],
        approve=_scripted_approver({r1.id: True, r2.id: False}),
    )
    with pytest.raises(StageExecutionError) as exc:
        asyncio.run(stage.run(ctx))
    assert "full_execution" in str(exc.value)


def test_approval_gate_records_request_and_decision_events_before_failing(ctx) -> None:
    """Every ask AND answer MUST land on the event log even when the gate aborts.

    Regression: the previous implementation raised before recording, so
    the audit row for the rejection that aborted the run was missing.
    """
    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages.approval_gate import ApprovalGate

    r1 = _req("hpc_submission")
    r2 = _req("full_execution")
    stage = ApprovalGate(
        requests=[r1, r2],
        approve=_scripted_approver({r1.id: True, r2.id: False}),
    )
    with pytest.raises(StageExecutionError):
        asyncio.run(stage.run(ctx))
    types = [e.type for e in ctx.event_log.list_events(ctx.run_id)]
    # Both asks and both answers present, request before its decision.
    assert types.count("approval_requested") == 2
    assert "approval_granted" in types
    assert "approval_rejected" in types


def test_approval_gate_rejects_mismatched_decision(ctx) -> None:
    """An approver answering the wrong request id is refused fail-fast."""
    from datetime import datetime as _datetime

    from molexp.harness.errors import StageExecutionError
    from molexp.harness.schemas.approval import ApprovalDecision
    from molexp.harness.stages.approval_gate import ApprovalGate

    async def confused_approver(request):
        return ApprovalDecision(
            request_id="someone-else",
            granted=True,
            decided_by="alice",
            decided_at=_datetime.now(tz=UTC),
        )

    stage = ApprovalGate(requests=[_req("overwrite")], approve=confused_approver)
    with pytest.raises(StageExecutionError, match="mismatched"):
        asyncio.run(stage.run(ctx))


def test_approval_gate_summary_carries_subject_artifact_ids(ctx) -> None:
    """``subject_artifact_ids`` flows to ``parent_ids`` on the summary."""
    from molexp.harness.stages.approval_gate import ApprovalGate

    parent = ctx.artifact_store.put_json(
        kind="bound_workflow", obj={"id": "bw"}, created_by="seed", parent_ids=[]
    )
    r1 = _req("hpc_submission")
    stage = ApprovalGate(
        requests=[r1],
        approve=_scripted_approver({r1.id: True}),
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
