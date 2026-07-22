"""Executor subprocess contract (``DryRunExecutor`` / ``LocalExecutor``) plus
the two ``ApprovalGate`` behaviors not owned by ``test_approval_gate.py``:
mismatched-decision rejection and subject-artifact-id parentage. (The gate's
grant/reject/pending paths + audit ordering live in ``test_approval_gate.py``.)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture()
def artifact_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


class TestDryRunExecutor:
    def test_returns_success_and_persists_stdout_stderr(self, artifact_store) -> None:
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
        assert artifact_store.get_ref(result.stdout_artifact.id).kind == "stdout"
        assert artifact_store.get_ref(result.stderr_artifact.id).kind == "stderr"


class TestLocalExecutor:
    def test_runs_the_command_and_captures_stdout(self, artifact_store, tmp_path: Path) -> None:
        from molexp.harness.executors.local import LocalExecutor
        from molexp.harness.schemas.command import CommandSpec

        result = asyncio.run(
            LocalExecutor().execute(
                CommandSpec(cmd=[sys.executable, "-c", "print('hello')"], cwd=str(tmp_path)),
                artifact_store=artifact_store,
            )
        )
        assert result.exit_code == 0
        assert b"hello" in artifact_store.get(result.stdout_artifact.id)

    def test_propagates_the_subprocess_exit_code(self, artifact_store, tmp_path: Path) -> None:
        from molexp.harness.executors.local import LocalExecutor
        from molexp.harness.schemas.command import CommandSpec

        result = asyncio.run(
            LocalExecutor().execute(
                CommandSpec(
                    cmd=[sys.executable, "-c", "import sys; sys.exit(7)"], cwd=str(tmp_path)
                ),
                artifact_store=artifact_store,
            )
        )
        assert result.exit_code == 7

    def test_ingests_existing_expected_outputs_and_records_missing(
        self, artifact_store, tmp_path: Path
    ) -> None:
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

    def test_refuses_expected_outputs_that_escape_cwd(self, artifact_store, tmp_path: Path) -> None:
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

    def test_marks_a_timed_out_command(self, artifact_store, tmp_path: Path) -> None:
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


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    return HarnessRunContext(
        run_id="run-gate",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=SQLiteEventLog(path=db),
        lineage_store=SQLiteArtifactLineageStore(path=db, artifact_store=a),
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
    from molexp.harness.schemas.approval import ApprovalDecision

    async def approve(request):
        return ApprovalDecision(
            request_id=request.id,
            granted=verdicts[request.id],
            decided_by="alice",
            decided_at=datetime.now(tz=UTC),
        )

    return approve


class TestApprovalGate:
    def test_rejects_a_decision_answering_the_wrong_request_id(self, ctx) -> None:
        """An approver answering the wrong request id is refused fail-fast."""
        from molexp.harness.errors import StageExecutionError
        from molexp.harness.schemas.approval import ApprovalDecision
        from molexp.harness.stages.approval_gate import ApprovalGate

        async def confused_approver(request):
            return ApprovalDecision(
                request_id="someone-else",
                granted=True,
                decided_by="alice",
                decided_at=datetime.now(tz=UTC),
            )

        stage = ApprovalGate(requests=[_req("overwrite")], approve=confused_approver)
        with pytest.raises(StageExecutionError, match="mismatched"):
            asyncio.run(stage.run(ctx))

    def test_summary_carries_subject_artifact_ids_as_parents(self, ctx) -> None:
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
