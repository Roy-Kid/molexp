"""Unit + integration tests for the shared curation flow.

Mirrors :mod:`molexp.services.curate_runtime.flow` — the single backend code
path both ``molexp curate`` and the ``curate-tasks`` route delegate to. Drives
:func:`run_curation_flow` and :func:`resolve_curation_arguments` directly from
Python (this suite owns the flow's own behavior; the CLI/route thin-adapter
wiring lives in ``test_cli``/``test_approvals_routes``).

Determinism: no wall-clock assertions, no network, no FS writes outside
``tmp_path``. The merged-registry seam is monkeypatched to a stub returning the
built-in curation catalog, so molmcp is never spawned.

Async style: ``@pytest.mark.asyncio`` (pytest-asyncio strict mode — there is no
``asyncio_mode = "auto"`` in ``pyproject.toml``, so each coroutine test carries
the marker).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.capabilities import curation_capabilities
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.registry import InMemoryCapabilityRegistry
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.stages import auto_grant_approver
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.services.curate_runtime.flow import (
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
    """Build a stub gateway whose ``curation_planner`` returns a fixed invocation."""
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
        "molexp.services.curate_runtime.flow.aresolve_curation_capability_registry",
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


class TestRunCurationFlow:
    """The three branches of :func:`run_curation_flow`."""

    @pytest.mark.asyncio
    async def test_read_only_capability_persists_artifacts_without_proposal(
        self, env: CurationEnv, patched_registry: None
    ) -> None:
        """A read-only ``scan_workspace`` flow returns granted and persists the
        catalog + invocation-result artifacts, and creates NO ChangeProposal."""
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
        assert not store.list_by_kind("change_proposal"), "read-only must not create a proposal"

    @pytest.mark.asyncio
    async def test_destructive_grant_moves_run_and_persists_proposal_artifacts(
        self, env: CurationEnv, patched_registry: None
    ) -> None:
        """An auto-granted ``move_run`` returns granted, relocates the subject run,
        and persists the §8 change_proposal + proposal_action_result artifacts."""
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

        store = FileArtifactStore(root=Path(env.run.run_dir) / "artifacts")
        assert store.list_by_kind("change_proposal"), "change_proposal artifact not persisted"
        assert store.list_by_kind("proposal_action_result"), "action result artifact not persisted"

    @pytest.mark.asyncio
    async def test_destructive_denial_records_granted_false_without_mutation(
        self, env: CurationEnv, patched_registry: None
    ) -> None:
        """A denied ``move_run`` is *recorded* (granted=False), never raised (§8.3):
        the subject run stays under the source experiment, never reaches the target."""
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
            approve=denying_approver,
        )

        assert result.granted is False
        source_ids = {run.id for run in env.source_exp.list_runs()}
        target_ids = {run.id for run in env.target_exp.list_runs()}
        assert "subject" in source_ids, "denied move must leave the run in its source"
        assert "subject" not in target_ids, "denied move must not relocate the run"


class TestResolveCurationArguments:
    """Live-object reconstruction from the planner's JSON reference handles."""

    def test_workspace_parameter_injects_the_live_workspace(self, env: CurationEnv) -> None:
        """A ``workspace`` parameter resolves to the injected live workspace,
        ignoring references entirely."""
        args = resolve_curation_arguments(
            "molexp.curation.scan_workspace",
            {},
            workspace=env.ws,
            experiment=env.source_exp,
        )

        assert args == {"workspace": env.ws}
        assert args["workspace"] is env.ws

    def test_reference_handles_reconstruct_run_and_target_experiment(
        self, env: CurationEnv
    ) -> None:
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
