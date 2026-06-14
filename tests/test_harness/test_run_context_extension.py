"""Tests for HarnessRunContext extension (Phase 7).

Three new optional service fields: capability_registry / agent_gateway /
approval_policy. All default None; existing 5-arg construction still works.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def stores(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return a, e, p


def test_construct_with_only_phase1_args_backward_compatible(tmp_path: Path, stores) -> None:
    """Phase-1..6 callers: 5 kwargs, no new ones. Must still work."""
    from molexp.harness.core.run_context import HarnessRunContext

    a, e, p = stores
    ctx = HarnessRunContext(
        run_id="run-x",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )
    assert ctx.run_id == "run-x"
    assert ctx.capability_registry is None
    assert ctx.agent_gateway is None
    assert ctx.approval_policy is None


def test_construct_with_all_new_services(tmp_path: Path, stores) -> None:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.gateways.stub import StubAgentGateway
    from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
    from molexp.harness.schemas.policy import ApprovalPolicy

    a, e, p = stores
    reg = InMemoryCapabilityRegistry()
    gw = StubAgentGateway(artifact_store=a)
    pol = ApprovalPolicy()
    ctx = HarnessRunContext(
        run_id="run-x",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
        capability_registry=reg,
        agent_gateway=gw,
        approval_policy=pol,
    )
    assert ctx.capability_registry is reg
    assert ctx.agent_gateway is gw
    assert ctx.approval_policy is pol


def test_new_fields_frozen_post_construction(tmp_path: Path, stores) -> None:
    from molexp.harness.core.run_context import HarnessRunContext

    a, e, p = stores
    ctx = HarnessRunContext(
        run_id="run-x",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )
    with pytest.raises(AttributeError):
        ctx.capability_registry = "mutated"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctx.agent_gateway = "mutated"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctx.approval_policy = "mutated"  # type: ignore[misc]
