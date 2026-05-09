"""Round-trip + manifest persistence tests for :class:`PlanRunHandoff`.

Covers acceptance criteria ac-001 (JSON / YAML round-trip) and
ac-002 (HumanReview persists the handoff section into manifest.yaml).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from molexp.agent.modes.plan import (
    PlanManifest,
    PlanRunHandoff,
    PlanWorkspaceHandle,
    ValidationReport,
)
from molexp.agent.modes.plan.policy import PlanModelPolicy
from molexp.agent.modes.plan.protocols import (
    AutoApproveGatePolicy,
    PlanDeps,
)
from molexp.agent.modes.plan.schemas import ApprovalDecision
from molexp.workspace import Workspace

from .conftest import FakeProvider


def _sample_handoff(plan_id: str = "plan_xyz") -> PlanRunHandoff:
    manifest = PlanManifest(
        plan_id=plan_id,
        created_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        report_source="report/original.md",
        workflow_ir_path=Path("ir/workflow.yaml"),
        task_ir_paths=(Path("ir/tasks/a.yaml"),),
        status="approved",
    )
    report = ValidationReport(passed=True, summary="all green")
    return PlanRunHandoff(
        plan_id=plan_id,
        experiment_workspace_path=Path("/tmp/ws/" + plan_id),
        workflow_yaml_path=Path("/tmp/ws/ir/workflow.yaml"),
        task_ir_paths=(Path("/tmp/ws/ir/tasks/a.yaml"),),
        entrypoint_module="experiment.workflow",
        entrypoint_symbol="create_workflow",
        manifest_snapshot=manifest,
        validation_report_snapshot=report,
        created_at=datetime(2026, 5, 9, 12, 5, 0, tzinfo=UTC),
    )


# ── ac-001 JSON round-trip ────────────────────────────────────────────────


def test_handoff_json_roundtrip() -> None:
    original = _sample_handoff()
    rebuilt = PlanRunHandoff.model_validate_json(original.model_dump_json())
    assert rebuilt == original


def test_handoff_is_frozen() -> None:
    from pydantic import ValidationError

    handoff = _sample_handoff()
    with pytest.raises(ValidationError):
        handoff.plan_id = "other"  # type: ignore[misc]


# ── ac-001 YAML round-trip ────────────────────────────────────────────────


def test_handoff_yaml_roundtrip() -> None:
    """YAML form goes through json.loads(model_dump_json()) — no custom encoders."""
    original = _sample_handoff()
    yaml_text = yaml.safe_dump(json.loads(original.model_dump_json()))
    parsed_dict = yaml.safe_load(yaml_text)
    rebuilt = PlanRunHandoff.model_validate(parsed_dict)
    assert rebuilt == original


# ── ac-002 manifest persistence ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_manifest_handoff_persisted(tmp_path: Path) -> None:
    """Driving PLAN_WORKFLOW end-to-end against AutoApproveGatePolicy
    persists the handoff into manifest.yaml; reloading via yaml.safe_load
    round-trips back into a PlanRunHandoff with field-equal result."""
    from molexp.agent.modes.plan import PLAN_WORKFLOW

    workspace = Workspace(tmp_path / "ws")
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="manifest_handoff")
    fake_provider = FakeProvider()
    deps = PlanDeps(
        provider=fake_provider,
        policy=PlanModelPolicy(),
        workspace_handle=handle,
        gate_policy=AutoApproveGatePolicy(),
    )
    result = await PLAN_WORKFLOW.execute(
        config={"user_input": "Investigate Suzuki coupling at varying temperatures."},
        deps=deps,
    )
    assert result.status == "completed", result

    manifest_yaml = handle.manifest_path()
    assert manifest_yaml.exists()
    raw = yaml.safe_load(manifest_yaml.read_text())
    assert isinstance(raw, dict)
    assert raw["status"] == "ready_for_run"
    assert raw["plan_mode"]["status"] == "ready_for_run"
    assert raw["plan_mode"]["ready_for_run"] is True
    assert "handoff" in raw, "manifest.yaml missing handoff block after approval"

    rebuilt = PlanRunHandoff.model_validate(raw["handoff"])
    assert rebuilt.plan_id == "manifest_handoff"
    assert rebuilt.entrypoint_module == "experiment.workflow"
    assert rebuilt.entrypoint_symbol == "create_workflow"


# ── HumanReview rejection branch ───────────────────────────────────────────


class _RejectingGate:
    async def human_review(self, _view: object) -> ApprovalDecision:
        return ApprovalDecision(approved=False, reason="rejected for testing")


@pytest.mark.asyncio
async def test_human_review_rejection_persists_non_runnable_handoff(tmp_path: Path) -> None:
    from molexp.agent.modes.plan import PLAN_WORKFLOW

    workspace = Workspace(tmp_path / "ws")
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="rejected_plan")
    deps = PlanDeps(
        provider=FakeProvider(),
        policy=PlanModelPolicy(),
        workspace_handle=handle,
        gate_policy=_RejectingGate(),  # type: ignore[arg-type]
    )
    result = await PLAN_WORKFLOW.execute(config={"user_input": "report"}, deps=deps)
    # On rejection: the workspace remains reviewable but is not ready for RunMode.
    assert result.status == "completed"
    raw = yaml.safe_load(handle.manifest_path().read_text())
    assert raw.get("status") == "ready_for_review"
    assert raw["plan_mode"]["ready_for_run"] is False
    assert raw["plan_mode"]["approved"] is False
    assert "handoff" in raw
