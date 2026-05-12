"""Tests for ``DiscoverCapabilities`` (Phase 4 capability-discovery node).

Covers acceptance criterion ``PYDA-14`` (the ``discover`` half):

* stub-probe evidence aggregation lands in ``capability/evidence.yaml``;
* missing-capability rows render to ``capability/missing.md``;
* the ``discovery_skipped=True`` short-circuit is preserved verbatim;
* :class:`NullCapabilityProbe` is the default fallback when no probe is
  configured, and re-raising :class:`CapabilityDiscoveryRequired`
  immediately blocks codegen.

The tests inject a stub probe through :class:`PlanDeps`, never hit
pydantic-ai or any real MCP server.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from molexp.agent.modes.plan.capability import (
    CapabilityEvidence,
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
    MissingCapability,
)
from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.tasks_capability import (
    DiscoverCapabilities,
    NullCapabilityProbe,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle

from .conftest import FakeRouter

# ── Stub probe for discovery ────────────────────────────────────────────────


class StubDiscoveryProbe:
    """Returns a canned :class:`CapabilityEvidenceBatch` from ``discover``.

    ``draft_needs`` is unused by these tests but provided so the
    Protocol is satisfied at runtime.
    """

    def __init__(self, batch: CapabilityEvidenceBatch) -> None:
        self._batch = batch
        self.discover_calls: list[CapabilityNeedReport] = []

    async def draft_needs(
        self,
        *,
        plan_brief: object,
        repair_context: object | None = None,
    ) -> CapabilityNeedReport:
        del plan_brief, repair_context
        return CapabilityNeedReport(discovery_required=False)

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: object | None = None,
    ) -> CapabilityEvidenceBatch:
        del repair_context
        self.discover_calls.append(report)
        return self._batch


class RaisingProbe:
    """Probe whose ``discover`` always raises :class:`CapabilityDiscoveryRequired`."""

    async def draft_needs(
        self,
        *,
        plan_brief: object,
        repair_context: object | None = None,
    ) -> CapabilityNeedReport:
        del plan_brief, repair_context
        return CapabilityNeedReport(discovery_required=True)

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: object | None = None,
    ) -> CapabilityEvidenceBatch:
        del report, repair_context
        raise CapabilityDiscoveryRequired(
            "Forced for test", reason="test", detail="raise from discover"
        )


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


def _ctx_stub(deps: PlanDeps, report: CapabilityNeedReport) -> object:
    class _Ctx:
        pass

    obj = _Ctx()
    obj.deps = deps  # type: ignore[attr-defined]
    obj.inputs = report  # type: ignore[attr-defined]
    return obj


def _evidence(api_ref: str = "molpy.builders.peptide.PeptideBuilder") -> CapabilityEvidence:
    module, _, symbol = api_ref.rpartition(".")
    return CapabilityEvidence(
        need_fingerprint="prepare:construct peptide",
        source="molmcp",
        package=module.split(".", 1)[0],
        module=module,
        symbol=symbol,
        kind="class",
        signature=f"class {symbol}:",
        doc_summary="Build a peptide from amino-acid codes.",
        api_ref=api_ref,
        confidence=0.95,
    )


# ── Aggregation paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_aggregates_evidence_into_workspace(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """Evidence rows land in capability/evidence.yaml verbatim."""
    batch = CapabilityEvidenceBatch(
        evidence=(
            _evidence("molpy.builders.peptide.PeptideBuilder"),
            _evidence("molexp.workflow.Task"),
        ),
        missing=(),
        discovery_skipped=False,
    )
    probe = StubDiscoveryProbe(batch)
    deps = _make_deps(workspace_handle, probe)
    node = DiscoverCapabilities()
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=(CapabilityNeed(task_id="prepare", capability="construct peptide"),),
    )
    result = await node.execute(_ctx_stub(deps, report))  # type: ignore[arg-type]

    assert result == batch
    persisted = yaml.safe_load((workspace_handle.capability_dir() / "evidence.yaml").read_text())
    assert len(persisted["evidence"]) == 2
    refs = {e["api_ref"] for e in persisted["evidence"]}
    assert "molpy.builders.peptide.PeptideBuilder" in refs


@pytest.mark.asyncio
async def test_discover_aggregates_missing_into_markdown(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """Missing rows render to capability/missing.md as a table."""
    needs = (CapabilityNeed(task_id="prepare", capability="construct peptide"),)
    batch = CapabilityEvidenceBatch(
        evidence=(),
        missing=(
            MissingCapability(
                need=needs[0],
                reason="mcp_no_match",
                detail="no result for 'construct peptide'",
                repairable=True,
            ),
        ),
        discovery_skipped=False,
    )
    probe = StubDiscoveryProbe(batch)
    deps = _make_deps(workspace_handle, probe)
    node = DiscoverCapabilities()
    report = CapabilityNeedReport(discovery_required=True, needs=needs)

    await node.execute(_ctx_stub(deps, report))  # type: ignore[arg-type]

    missing_md = (workspace_handle.capability_dir() / "missing.md").read_text()
    assert "mcp_no_match" in missing_md
    assert "prepare: construct peptide" in missing_md


@pytest.mark.asyncio
async def test_discover_skipped_short_circuit_persists(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """``discovery_skipped=True`` is preserved through write back."""
    batch = CapabilityEvidenceBatch(
        evidence=(),
        missing=(),
        discovery_skipped=True,
    )
    probe = StubDiscoveryProbe(batch)
    deps = _make_deps(workspace_handle, probe)
    node = DiscoverCapabilities()
    report = CapabilityNeedReport(discovery_required=False)

    result = await node.execute(_ctx_stub(deps, report))  # type: ignore[arg-type]
    assert result.discovery_skipped is True

    persisted = yaml.safe_load((workspace_handle.capability_dir() / "evidence.yaml").read_text())
    assert persisted["discovery_skipped"] is True


# ── Failure-path: NullCapabilityProbe blocks codegen ───────────────────────


@pytest.mark.asyncio
async def test_null_probe_blocks_when_node_invoked_with_discovery_required(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """``DiscoverCapabilities`` re-raises CapabilityDiscoveryRequired.

    The repair loop catches this exception and re-runs the
    discovery pair; here we just assert the node propagates rather
    than swallows.
    """
    deps = _make_deps(workspace_handle, NullCapabilityProbe())
    node = DiscoverCapabilities()
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=(CapabilityNeed(task_id="prepare", capability="x"),),
    )

    with pytest.raises(CapabilityDiscoveryRequired):
        await node.execute(_ctx_stub(deps, report))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_discover_propagates_probe_capability_required(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """A probe that itself raises CapabilityDiscoveryRequired bubbles up."""
    deps = _make_deps(workspace_handle, RaisingProbe())
    node = DiscoverCapabilities()
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=(CapabilityNeed(task_id="prepare", capability="x"),),
    )

    with pytest.raises(CapabilityDiscoveryRequired):
        await node.execute(_ctx_stub(deps, report))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_discover_falls_back_to_null_probe_when_deps_unset(
    workspace_handle: PlanWorkspaceHandle,
) -> None:
    """``deps.capability_probe=None`` resolves to NullCapabilityProbe at run time."""
    deps = _make_deps(workspace_handle, probe=None)  # type: ignore[arg-type]
    deps = replace(deps, capability_probe=None)
    node = DiscoverCapabilities()
    pure = CapabilityNeedReport(discovery_required=False)
    result = await node.execute(_ctx_stub(deps, pure))  # type: ignore[arg-type]
    assert result.discovery_skipped is True
