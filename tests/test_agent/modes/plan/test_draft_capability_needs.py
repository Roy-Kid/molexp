"""Tests for ``DraftCapabilityNeeds`` (Phase 4 capability-discovery node).

Covers acceptance criterion ``PYDA-14`` (the ``draft_needs`` half):
the three-state branching of the
:class:`~molexp.agent.modes.plan.protocols.CapabilityProbe`'s
``draft_needs`` return value (empty + ``discovery_required=True``,
empty + ``discovery_required=False``, non-empty needs), the resulting
workspace persistence at ``capability/needs.yaml``, and the
:class:`NullCapabilityProbe` fallback path.

The tests inject a stub probe through :class:`PlanDeps`, never hit
pydantic-ai or any real MCP server.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
)
from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    PlanBriefResult,
    TaskIRBrief,
    TaskIRResult,
    WorkflowContract,
    WorkflowIRResult,
)
from molexp.agent.modes.plan.tasks_capability import (
    DraftCapabilityNeeds,
    NullCapabilityProbe,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle

from .conftest import FakeRouter, canned_presets

# ── Stub probe ─────────────────────────────────────────────────────────────


class StubCapabilityProbe:
    """In-memory probe that returns a canned report from ``draft_needs``.

    ``discover`` is not exercised by the DraftCapabilityNeeds tests but
    has a sensible default so the Protocol is satisfied at runtime.
    """

    def __init__(self, report: CapabilityNeedReport) -> None:
        self._report = report
        self.draft_calls: list[tuple[object, object, tuple[TaskIRBrief, ...]]] = []

    async def draft_needs(
        self,
        *,
        plan_brief: object,
        contract: object,
        briefs: Sequence[TaskIRBrief],
    ) -> CapabilityNeedReport:
        recorded = (plan_brief, contract, tuple(briefs))
        self.draft_calls.append(recorded)
        return self._report

    async def discover(self, report: CapabilityNeedReport) -> CapabilityEvidenceBatch:
        del report
        return CapabilityEvidenceBatch(discovery_skipped=True)


# ── Fixtures + helpers ─────────────────────────────────────────────────────


@pytest.fixture
def workspace_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    from molexp.workspace import Workspace

    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="dcap_plan")


def _make_deps(handle: PlanWorkspaceHandle, probe: object) -> PlanDeps:
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    return PlanDeps(
        router=FakeRouter(),
        policy=STANDARD_PLAN_POLICY,
        workspace_handle=handle,
        capability_probe=probe,  # type: ignore[arg-type]
    )


def _make_inputs() -> dict[str, object]:
    presets = canned_presets()
    plan_brief = presets[PlanBrief]
    contract = presets[WorkflowContract]
    assert isinstance(plan_brief, PlanBrief)
    assert isinstance(contract, WorkflowContract)
    plan_brief_result = PlanBriefResult(
        plan_path=Path("plan/implementation_plan.md"),
        plan_brief=plan_brief,
    )
    workflow_ir_result = WorkflowIRResult(
        workflow_yaml_path=Path("ir/workflow.yaml"),
        contract=contract,
    )
    briefs_dict = presets[TaskIRBrief]
    assert isinstance(briefs_dict, dict)
    task_ir_result = TaskIRResult(
        task_ir_paths=tuple(Path(f"ir/tasks/{tid}.yaml") for tid in briefs_dict),
        briefs=tuple(briefs_dict.values()),
    )
    return {
        "DraftImplementationPlan": plan_brief_result,
        "CompileWorkflowIR": workflow_ir_result,
        "CompileTaskIR": task_ir_result,
    }


def _ctx_stub(deps: PlanDeps, inputs: dict[str, object]) -> object:
    """Build the minimal :class:`TaskContext`-shaped object the node needs.

    The node only reads ``ctx.deps`` and ``ctx.inputs`` so a plain
    dataclass-style object is enough; we don't need the full pydantic
    workflow plumbing for these unit tests.
    """

    class _Ctx:
        pass

    obj = _Ctx()
    obj.deps = deps  # type: ignore[attr-defined]
    obj.inputs = inputs  # type: ignore[attr-defined]
    return obj


# ── Three-state branching ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_draft_needs_three_state_empty_with_discovery_required(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """Empty needs + discovery_required=True is preserved verbatim."""
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=(),
        rationale_summary="needs unknown; force discovery",
    )
    probe = StubCapabilityProbe(report)
    deps = _make_deps(workspace_handle, probe)
    node = DraftCapabilityNeeds()

    result = await node.execute(_ctx_stub(deps, _make_inputs()))  # type: ignore[arg-type]

    assert result.discovery_required is True
    assert result.needs == ()
    needs_path = workspace_handle.capability_dir() / "needs.yaml"
    assert needs_path.exists()
    persisted = yaml.safe_load(needs_path.read_text())
    assert persisted["discovery_required"] is True
    assert len(probe.draft_calls) == 1


@pytest.mark.asyncio
async def test_draft_needs_three_state_empty_with_discovery_skipped(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """Empty needs + discovery_required=False short-circuits downstream discovery."""
    report = CapabilityNeedReport(
        discovery_required=False,
        needs=(),
        rationale_summary="pure stdlib",
    )
    probe = StubCapabilityProbe(report)
    deps = _make_deps(workspace_handle, probe)
    node = DraftCapabilityNeeds()

    result = await node.execute(_ctx_stub(deps, _make_inputs()))  # type: ignore[arg-type]

    assert result.discovery_required is False
    persisted = yaml.safe_load((workspace_handle.capability_dir() / "needs.yaml").read_text())
    assert persisted["discovery_required"] is False


@pytest.mark.asyncio
async def test_draft_needs_three_state_non_empty(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """Non-empty needs round-trip through ``capability/needs.yaml``."""
    needs = (
        CapabilityNeed(
            task_id="prepare",
            capability="construct peptide",
            rationale="needs amino-acid builder",
            expected_kind="class",
        ),
        CapabilityNeed(
            task_id="couple",
            capability="dispatch coupling reaction",
            rationale="needs catalysis runner",
            expected_kind="callable",
        ),
    )
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=needs,
        rationale_summary="needs Molcrafts symbols",
    )
    probe = StubCapabilityProbe(report)
    deps = _make_deps(workspace_handle, probe)
    node = DraftCapabilityNeeds()

    result = await node.execute(_ctx_stub(deps, _make_inputs()))  # type: ignore[arg-type]

    assert result == report
    persisted = yaml.safe_load((workspace_handle.capability_dir() / "needs.yaml").read_text())
    assert len(persisted["needs"]) == 2
    assert persisted["needs"][0]["task_id"] == "prepare"


# ── NullCapabilityProbe behavior ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_null_probe_default_returns_discovery_not_required() -> None:
    """:class:`NullCapabilityProbe` always reports discovery_required=False."""
    probe = NullCapabilityProbe()
    presets = canned_presets()
    plan_brief = presets[PlanBrief]
    contract = presets[WorkflowContract]
    briefs_dict = presets[TaskIRBrief]
    assert isinstance(plan_brief, PlanBrief)
    assert isinstance(contract, WorkflowContract)
    assert isinstance(briefs_dict, dict)
    report = await probe.draft_needs(
        plan_brief=plan_brief,
        contract=contract,
        briefs=tuple(briefs_dict.values()),
    )
    assert report.discovery_required is False
    assert report.needs == ()


@pytest.mark.asyncio
async def test_null_probe_blocks_codegen_when_discovery_required() -> None:
    """``NullCapabilityProbe.discover`` raises when discovery is required.

    This is the key safety invariant: codegen MUST be stopped when
    discovery cannot proceed — silently passing an empty evidence
    batch downstream would let the LLM make up Molcrafts API calls.
    """
    probe = NullCapabilityProbe()
    forced = CapabilityNeedReport(
        discovery_required=True,
        needs=(CapabilityNeed(task_id="prepare", capability="x"),),
        rationale_summary="forced",
    )
    with pytest.raises(CapabilityDiscoveryRequired) as excinfo:
        await probe.discover(forced)
    assert excinfo.value.reason == "no_probe"


@pytest.mark.asyncio
async def test_null_probe_returns_skipped_batch_for_pure_stdlib() -> None:
    """When discovery_required=False, NullCapabilityProbe returns a skipped batch."""
    probe = NullCapabilityProbe()
    pure = CapabilityNeedReport(discovery_required=False)
    batch = await probe.discover(pure)
    assert batch.discovery_skipped is True
    assert batch.evidence == ()


@pytest.mark.asyncio
async def test_draft_capability_needs_falls_back_to_null_probe_when_unset(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """No probe configured → DraftCapabilityNeeds calls NullCapabilityProbe.

    ``capability_probe=None`` is the default ``PlanDeps`` shape; the
    node must transparently install a NullCapabilityProbe at run time
    rather than crashing.
    """
    deps = _make_deps(workspace_handle, probe=None)  # type: ignore[arg-type]
    deps = replace(deps, capability_probe=None)
    node = DraftCapabilityNeeds()

    result = await node.execute(_ctx_stub(deps, _make_inputs()))  # type: ignore[arg-type]

    assert result.discovery_required is False
    assert result.rationale_summary == "no probe configured"
