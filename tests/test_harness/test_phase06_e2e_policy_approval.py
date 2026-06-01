"""End-to-end Phase-6 policy → approval flow integration test (ac-013).

Constructs a "dangerous" BoundWorkflow that triggers ≥4 intents at once,
evaluates the policy, records mixed grant/reject decisions, and verifies
the event log carries the full sequence with correct payloads.

Also re-confirms the Phase-6 public surface + Phase-1..5 regression.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def _dangerous_workflow():
    """Slurm backend + 48h runtime + agent_inferred param + overwrite flag."""
    from molexp.harness.schemas.bound_workflow import (
        BoundTask,
        BoundWorkflow,
        ExecutionEnvironment,
        ResourcePolicy,
    )
    from molexp.harness.schemas.parameter import ParameterValue

    t1 = BoundTask(
        id="b1",
        ir_task_id="t1",
        capability_id="molpy.md.NEMDRunner",
        package="molpy",
        callable="molpy.md.NEMDRunner.run",
        parameters={
            "field_strength": ParameterValue(value=1e6, source="agent_inferred"),
        },
        inputs={"field_strength": "wf:field"},
        outputs={"trajectory": "traj.dcd"},
    )
    return BoundWorkflow(
        id="bw-dangerous",
        workflow_ir_id="wf-water-nemd",
        tasks=[t1],
        edges=[],
        execution_backend="slurm",
        environment=ExecutionEnvironment(python_version="3.12"),
        resource_policy=ResourcePolicy(
            backend="slurm",
            max_runtime_s=172800,  # 48h → triggers large_resource_request
            max_memory_gb=128.0,
            denied_paths=["/", "~/.ssh"],
        ),
        review_flags=["overwrite"],
    )


def test_dangerous_workflow_triggers_multiple_intents(tmp_path: Path) -> None:
    from molexp.harness import (
        ApprovalDecision,
        ApprovalPolicy,
        SQLiteEventLog,
        evaluate_approval_policy,
        record_approval_decision,
        record_approval_request,
    )

    bw = _dangerous_workflow()
    policy = ApprovalPolicy()  # all six require_for_* default True
    requests = evaluate_approval_policy(policy, bw=bw)

    # Should hit hpc_submission + full_execution + large_resource_request
    # + overwrite + agent_inferred_scientific_parameters = 5 intents.
    intents = {r.intent for r in requests}
    assert intents == {
        "hpc_submission",
        "full_execution",
        "large_resource_request",
        "overwrite",
        "agent_inferred_scientific_parameters",
    }
    assert len(requests) >= 4  # spec floor

    # Log all five requests, grant the first, reject the second.
    event_log = SQLiteEventLog(path=tmp_path / "events.sqlite")
    run_id = "run-p6-e2e"
    for req in requests:
        record_approval_request(event_log, run_id, req)

    granted_decision = ApprovalDecision(
        request_id=requests[0].id,
        granted=True,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
        reason="OK to proceed",
    )
    rejected_decision = ApprovalDecision(
        request_id=requests[1].id,
        granted=False,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
        reason="Postpone full run",
    )
    record_approval_decision(event_log, run_id, requests[0], granted_decision)
    record_approval_decision(event_log, run_id, requests[1], rejected_decision)

    # Verify event log carries the full sequence.
    timeline = event_log.get_timeline(run_id)
    types = [e.type for e in timeline]
    # Expect: [approval_requested * N, approval_granted, approval_rejected]
    expected_request_count = len(requests)
    assert types[:expected_request_count] == ["approval_requested"] * expected_request_count
    assert types[expected_request_count] == "approval_granted"
    assert types[expected_request_count + 1] == "approval_rejected"

    # ApprovalRequest.id is NOT an artifact-store id — it lives in payload
    # rather than artifact_ids (see record_approval_request docstring).
    for i, req in enumerate(requests):
        assert timeline[i].artifact_ids == []
        assert timeline[i].payload["request_id"] == req.id
    # The decision events also carry the originating request id in payload.
    assert timeline[expected_request_count].artifact_ids == []
    assert timeline[expected_request_count].payload["request_id"] == requests[0].id
    assert timeline[expected_request_count + 1].artifact_ids == []
    assert timeline[expected_request_count + 1].payload["request_id"] == requests[1].id


# ----------------------------------------- public-surface invariants


def test_phase06_public_symbols_importable_from_top_level() -> None:
    from molexp.harness import (  # noqa: F401
        ApprovalDecision,
        ApprovalIntent,
        ApprovalPolicy,
        ApprovalRequest,
        PathPolicy,
        ToolPolicy,
        evaluate_approval_policy,
        make_final_report_approval_request,
        record_approval_decision,
        record_approval_request,
    )


def test_phase01_to_phase05_surface_still_intact() -> None:
    """Regression: every Phase-1..5 export still importable."""
    from molexp.harness import (  # noqa: F401
        ArtifactRef,
        BoundWorkflow,
        CapabilityRegistry,
        ExperimentReport,
        SaveUserPlan,
        TestSpec,
        ToolCapability,
        UserPlan,
        WorkflowIR,
        validate_bound_workflow,
        validate_provenance,
        validate_test_spec,
        validate_workflow_ir,
    )
