"""In-process tests for the shared workspace-curation flow (link 05).

Drives :func:`molexp.server.curate_runtime.flow.run_curation_flow` directly from
Python (the route + CLI tests live elsewhere). The flow is the single backend
code path both ``molexp curate`` and ``POST /curate`` will share: it persists the
rendered capability catalog, asks the ``curation_planner`` agent for a structured
:class:`CurationInvocation`, validates the chosen capability against the merged
registry, reconstructs the curation function's live-object kwargs, gates any
destructive capability through the ``side_effects`` -> ``ApprovalGate`` rule, and
finally invokes the resolved callable in-process.

These tests are **RED until the production module lands** — the top-level import
of ``molexp.server.curate_runtime.flow`` fails at collection
(``ModuleNotFoundError``) because that package does not exist yet.

Determinism: no wall-clock assertions, no network, no FS writes outside
``tmp_path``. The merged-registry seam is monkeypatched to a stub returning the
built-in curation catalog, so molmcp is never spawned.

Async style: this module uses ``@pytest.mark.asyncio`` (pytest-asyncio in strict
mode — there is no ``asyncio_mode = "auto"`` in ``pyproject.toml``, so each
coroutine test must carry the marker), mirroring
``tests/test_harness/test_router_backed_gateway.py`` and
``tests/test_harness/test_sqlite_thread_safety.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness import (
    Approver,
    InMemoryCapabilityRegistry,
    StageExecutionError,
    auto_grant_approver,
)
from molexp.harness.capabilities import curation_capabilities
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.server.curate_runtime.flow import (
    CurationInvocation,
    CurationResult,
    resolve_curation_arguments,
    run_curation_flow,
)
from molexp.workspace import Experiment, Run, Workspace

# ─────────────────────────────────────────────────────────── fixtures / helpers


@dataclass(frozen=True)
class CurationEnv:
    """A materialized workspace with a source/target experiment pair.

    Attributes:
        ws: The live workspace (what a ``workspace`` curation arg resolves to).
        source_exp: Experiment hosting the curate ``run`` and the ``subject_run``.
        target_exp: Sibling experiment a ``move_run`` relocates the subject into.
        run: The curate run whose ``run_dir`` hosts harness artifacts + the DB.
        subject_run: A throwaway run under ``source_exp`` to be moved.
    """

    ws: Workspace
    source_exp: Experiment
    target_exp: Experiment
    run: Run
    subject_run: Run


@pytest.fixture
def env(tmp_path: Path) -> CurationEnv:
    """Build a materialized workspace with the source/target experiment pair."""
    ws = Workspace(tmp_path / "ws", "curation-ws")
    ws.materialize()
    proj = ws.add_project("p")
    source_exp = proj.add_experiment("source-exp")
    target_exp = proj.add_experiment("target-exp")
    run = source_exp.add_run({"mode": "curate"}, id="curate-run")
    subject_run = source_exp.add_run({"seed": 0}, id="subject")
    return CurationEnv(
        ws=ws,
        source_exp=source_exp,
        target_exp=target_exp,
        run=run,
        subject_run=subject_run,
    )


def _gateway_with_planner(
    run: Run,
    *,
    capability_id: str,
    references: dict[str, str],
    reason: str = "",
) -> StubAgentGateway:
    """Build a stub gateway whose ``curation_planner`` returns a fixed invocation.

    The stub writes through a ``FileArtifactStore`` rooted at the same
    ``run_dir/artifacts`` directory the flow uses, so the planner's output
    artifact lands beside the flow's own artifacts.
    """
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    gateway = StubAgentGateway(store)
    gateway.register(
        "curation_planner",
        {"capability_id": capability_id, "references": references, "reason": reason},
        output_kind="curation_invocation",
    )
    return gateway


async def _stub_registry(workspace_root: str) -> InMemoryCapabilityRegistry:
    """Stand in for the merged-registry seam: built-in curation catalog only."""
    return InMemoryCapabilityRegistry(curation_capabilities())


@pytest.fixture
def patched_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the merged-registry seam so molmcp is never spawned."""
    monkeypatch.setattr(
        "molexp.server.curate_runtime.flow.aresolve_curation_capability_registry",
        _stub_registry,
    )


async def denying_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Reject every approval request — proves the gate aborts before mutation."""
    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="test",
        decided_at=datetime.now(tz=UTC),
    )


# A type-checked binding: the static checker confirms ``denying_approver``
# conforms to the ``Approver`` callback contract.
_DENYING_APPROVER: Approver = denying_approver


# ─────────────────────────────────────────────── basics · CurationInvocation schema
# Category: basics (the planner's structured-output contract: defaults).


def test_curation_invocation_defaults() -> None:
    """``CurationInvocation`` defaults ``references`` to ``{}`` and ``reason`` to ''."""
    invocation = CurationInvocation(capability_id="molexp.curation.scan_workspace")

    assert invocation.capability_id == "molexp.curation.scan_workspace"
    assert invocation.references == {}
    assert invocation.reason == ""


# ─────────────────────────────────── ac-002 · happy path, read-only capability
# Category: basics + integration (planner -> validate -> resolve -> invoke ->
# persist, no approval needed for a read-only capability).


@pytest.mark.asyncio
async def test_read_only_flow_scans_and_persists_artifacts(
    env: CurationEnv,
    patched_registry: None,
) -> None:
    """A read-only ``scan_workspace`` flow returns granted and persists both the
    capability catalog and the invocation-result artifact (no approver needed)."""
    gateway = _gateway_with_planner(
        env.run,
        capability_id="molexp.curation.scan_workspace",
        references={},
        reason="scan",
    )

    result = await run_curation_flow(
        "inventory the workspace",
        workspace=env.ws,
        experiment=env.source_exp,
        run=env.run,
        gateway=gateway,
    )

    assert isinstance(result, CurationResult)
    assert result.capability_id == "molexp.curation.scan_workspace"
    assert result.granted is True

    store = FileArtifactStore(root=Path(env.run.run_dir) / "artifacts")
    assert store.list_by_kind("capability_catalog"), "catalog artifact not persisted"
    assert store.list_by_kind("capability_invocation_result"), "result artifact not persisted"


# ─────────────────────────────────── ac-002b · resolve_curation_arguments (unit)
# Category: basics + edge case (workspace injection; live-object reconstruction).


def test_resolve_arguments_injects_live_workspace(env: CurationEnv) -> None:
    """A ``workspace`` parameter resolves to the injected live workspace, ignoring
    references entirely."""
    args = resolve_curation_arguments(
        "molexp.curation.scan_workspace",
        {},
        workspace=env.ws,
        experiment=env.source_exp,
    )

    assert args == {"workspace": env.ws}
    assert args["workspace"] is env.ws


def test_resolve_arguments_reconstructs_run_and_target_experiment(env: CurationEnv) -> None:
    """``run`` resolves to a live ``Run`` and ``target_experiment`` to a live
    ``Experiment`` — both reconstructed from JSON id references plus context."""
    args = resolve_curation_arguments(
        "molexp.curation.move_run",
        {"run": "subject", "target_experiment": "target-exp"},
        workspace=env.ws,
        experiment=env.source_exp,
    )

    assert set(args) == {"run", "target_experiment"}

    run_arg = args["run"]
    assert isinstance(run_arg, Run)
    assert run_arg.id == "subject"

    target_arg = args["target_experiment"]
    assert isinstance(target_arg, Experiment)
    assert target_arg.id == "target-exp"


# ──────────────────────────── ac-004 (denied) · destructive gate aborts the move
# Category: edge case + immutability (denial raises; no mutation occurs).


@pytest.mark.asyncio
async def test_destructive_denied_raises_and_does_not_mutate(
    env: CurationEnv,
    patched_registry: None,
) -> None:
    """A denied ``move_run`` raises ``StageExecutionError`` before any mutation —
    the subject run stays under the source experiment, never reaches the target."""
    gateway = _gateway_with_planner(
        env.run,
        capability_id="molexp.curation.move_run",
        references={"run": "subject", "target_experiment": "target-exp"},
        reason="relocate",
    )

    with pytest.raises(StageExecutionError):
        await run_curation_flow(
            "move subject run to target-exp",
            workspace=env.ws,
            experiment=env.source_exp,
            run=env.run,
            gateway=gateway,
            approve=_DENYING_APPROVER,
        )

    source_ids = {run.id for run in env.source_exp.list_runs()}
    target_ids = {run.id for run in env.target_exp.list_runs()}
    assert "subject" in source_ids, "denied move must leave the run in its source"
    assert "subject" not in target_ids, "denied move must not relocate the run"


# ──────────────────────────── ac-004 (granted) · destructive proceeds, run moves
# Category: integration + lifecycle (grant -> invoke -> mutation observed).


@pytest.mark.asyncio
async def test_destructive_granted_proceeds_and_moves_run(
    env: CurationEnv,
    patched_registry: None,
) -> None:
    """An auto-granted ``move_run`` returns granted and relocates the subject run:
    it now lives under the target experiment and is gone from the source."""
    gateway = _gateway_with_planner(
        env.run,
        capability_id="molexp.curation.move_run",
        references={"run": "subject", "target_experiment": "target-exp"},
        reason="relocate",
    )

    result = await run_curation_flow(
        "move subject run to target-exp",
        workspace=env.ws,
        experiment=env.source_exp,
        run=env.run,
        gateway=gateway,
        approve=auto_grant_approver,
    )

    assert result.granted is True
    assert result.capability_id == "molexp.curation.move_run"

    source_ids = {run.id for run in env.source_exp.list_runs()}
    target_ids = {run.id for run in env.target_exp.list_runs()}
    assert "subject" not in source_ids, "granted move must remove the run from its source"
    assert "subject" in target_ids, "granted move must relocate the run to the target"
