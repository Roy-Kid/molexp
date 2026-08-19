"""Lock the shrunk ``molexp.harness`` public surface.

The harness top level exports only the orchestration surface: the Stage
machinery, the one shipped pipeline (``run_plan``), the stores,
the executor seam, the approval gate, and the agent gateway — plus the symbols
the public
docs (``docs/guide/plan-mode.md`` / ``docs/architecture/plan-mode.md``) cite
as harness API (``RouterBackedAgentGateway``, ``CapabilityRegistry``,
``StageExecutionError``, ``SQLiteEventLog``, ``SQLiteArtifactLineageStore``,
``replay_metadata``).

Stages, schemas, and validators are NOT re-exported at the top level; their
canonical import paths are ``molexp.harness.stages`` / ``.schemas`` /
``.validators``. Any regrowth of the top-level surface must be a deliberate
edit to this ledger.
"""

from __future__ import annotations

import molexp.harness as harness

# The frozen public surface: 12 baseline symbols + 6 doc-cited symbols
# + 2 approval-inbox symbols (vision-loop-01: the suspend signal and the
# persisted-decision store the server decide-route constructs).
EXPECTED_ALL = {
    # baseline — Stage machinery + the shipped pipeline
    "Stage",
    "StageRunner",
    "HarnessRunContext",
    "ChatMode",
    "run_plan",
    "ModeResult",
    "chat_loop_config",
    # baseline — stores, executors, gate, gateway
    "ArtifactStore",
    "FileArtifactStore",
    "Executor",
    "LocalExecutor",
    "DryRunExecutor",
    "ApprovalGate",
    "AgentGateway",
    # doc-cited (docs/guide/plan-mode.md + docs/architecture/plan-mode.md)
    "RouterBackedAgentGateway",
    "CapabilityRegistry",
    "StageExecutionError",
    "SQLiteEventLog",
    "SQLiteArtifactLineageStore",
    "replay_metadata",
    # approval inbox (vision-loop-01)
    "ApprovalPendingError",
    "SQLiteApprovalStore",
}


def test_all_is_exactly_the_frozen_ledger() -> None:
    assert set(harness.__all__) == EXPECTED_ALL


def test_every_exported_symbol_resolves() -> None:
    for name in harness.__all__:
        assert getattr(harness, name) is not None


def test_stages_schemas_validators_not_reexported_at_top_level() -> None:
    """The long tail lives at its submodule path, not on the package root."""
    for demoted in (
        "SaveUserPlan",  # stage → molexp.harness.stages
        "ExperimentSpec",  # schema → molexp.harness.schemas
        "PlanWorkflowIR",  # schema → molexp.harness.schemas
        "PlanArtifactRef",  # schema → molexp.harness.schemas
        "PlanValidationReport",  # schema → molexp.harness.schemas
        "WorkflowIRValidator",  # validator → molexp.harness.validators
        "InMemoryCapabilityRegistry",  # registry impl → molexp.harness.registry
        "auto_grant_approver",  # helper → molexp.harness.stages
        "summarize_workspace",  # copilot → molexp.harness.copilot
    ):
        assert demoted not in harness.__all__, demoted


def test_submodule_paths_still_serve_the_demoted_symbols() -> None:
    from molexp.harness.copilot import summarize_workspace
    from molexp.harness.registry import InMemoryCapabilityRegistry
    from molexp.harness.schemas import (
        ExperimentSpec,
        PlanArtifactRef,
        PlanTaskIR,
        PlanValidationReport,
        PlanWorkflowIR,
    )
    from molexp.harness.stages import SaveUserPlan, auto_grant_approver
    from molexp.harness.validators import WorkflowIRValidator

    for obj in (
        ExperimentSpec,
        PlanArtifactRef,
        PlanTaskIR,
        PlanValidationReport,
        PlanWorkflowIR,
        SaveUserPlan,
        WorkflowIRValidator,
        InMemoryCapabilityRegistry,
        auto_grant_approver,
        summarize_workspace,
    ):
        assert obj is not None
