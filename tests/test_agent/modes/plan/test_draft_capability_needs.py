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
)
from molexp.agent.modes.plan.tasks_capability import (
    DraftCapabilityNeeds,
    NullCapabilityProbe,
)
from molexp.agent.modes.plan.plan_folder import PlanFolder

from .conftest import FakeRouter, canned_presets

# ── Stub probe ─────────────────────────────────────────────────────────────


class StubCapabilityProbe:
    """In-memory probe that returns a canned report from ``draft_needs``.

    ``discover`` is not exercised by the DraftCapabilityNeeds tests but
    has a sensible default so the Protocol is satisfied at runtime.
    """

    def __init__(self, report: CapabilityNeedReport) -> None:
        self._report = report
        self.draft_calls: list[PlanBrief] = []

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
        repair_context: object | None = None,
    ) -> CapabilityNeedReport:
        del repair_context
        self.draft_calls.append(plan_brief)
        return self._report

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: object | None = None,
    ) -> CapabilityEvidenceBatch:
        del report, repair_context
        return CapabilityEvidenceBatch(discovery_skipped=True)


# ── Fixtures + helpers ─────────────────────────────────────────────────────


@pytest.fixture
def workspace_handle(tmp_path: Path) -> PlanFolder:
    from molexp.workspace import Workspace

    return Workspace(tmp_path / "ws").add_folder(PlanFolder(name="dcap_plan"))


def _make_deps(handle: PlanFolder, probe: object) -> PlanDeps:
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    return PlanDeps(
        router=FakeRouter(),
        policy=STANDARD_PLAN_POLICY,
        plan_folder=handle,
        capability_probe=probe,  # type: ignore[arg-type]
    )


def _make_inputs() -> PlanBriefResult:
    """Return the single bare upstream input the reordered pipeline supplies.

    DraftCapabilityNeeds now runs immediately after DraftImplementationPlan
    so its only upstream is the :class:`PlanBriefResult`.
    """
    presets = canned_presets()
    plan_brief = presets[PlanBrief]
    assert isinstance(plan_brief, PlanBrief)
    return PlanBriefResult(
        plan_path=Path("plan/implementation_plan.md"),
        plan_brief=plan_brief,
    )


def _ctx_stub(deps: PlanDeps, inputs: object, *, user_input: str = "") -> object:
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
    obj.config = {"user_input": user_input}  # type: ignore[attr-defined]
    return obj


# ── Three-state branching ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_draft_needs_three_state_empty_with_discovery_required(
    workspace_handle: PlanFolder,
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
    workspace_handle: PlanFolder,
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
    workspace_handle: PlanFolder,
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
    assert isinstance(plan_brief, PlanBrief)
    report = await probe.draft_needs(plan_brief=plan_brief)
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
    workspace_handle: PlanFolder,
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


@pytest.mark.asyncio
async def test_raw_user_input_required_namespace_forces_discovery(
    workspace_handle: PlanFolder,
) -> None:
    """Explicit raw-input constraints survive PlanBrief abstraction."""
    report = CapabilityNeedReport(
        discovery_required=False,
        needs=(),
        rationale_summary="planner abstracted implementation details",
    )
    probe = StubCapabilityProbe(report)
    deps = _make_deps(workspace_handle, probe)
    node = DraftCapabilityNeeds()

    result = await node.execute(
        _ctx_stub(
            deps,
            _make_inputs(),
            user_input="You need to explicitly use molpy for chain construction.",
        )  # type: ignore[arg-type]
    )

    assert result.discovery_required is True
    assert any(h.namespace == "molpy" and h.strength == "required" for h in result.hints)
    assert any("molpy" in need.query_hints for need in result.needs)
