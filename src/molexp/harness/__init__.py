"""molexp.harness — lineage-first scientific workflow harness.

Top-level package, sibling of :mod:`molexp.workspace`, :mod:`molexp.workflow`,
and :mod:`molexp.agent`. Owns state, artifact lineage, validation, approval,
execution, and audit for harness-driven runs; agents are demoted to
proposal generators behind hard boundaries defined in
``.claude/notes/harness-goal.md``.

Provenance split (one owner per concern): the harness records
**pipeline-artifact lineage only** — which stage of which run produced which
artifact, derived from which prior artifact (:class:`ArtifactLineageStore`)
— plus the audit event timeline. **Run-level provenance** (params, merged
config + ``config_hash``, profile, workflow identity, execution history,
environment) is owned by :mod:`molexp.workspace` (``RunMetadata`` /
``AssetCatalog``); code-version and environment capture belong there, never
here.

The state substrate: typed artifact references, an append-only event log, an
artifact-lineage edge store, and a Stage / StageRunner wrapper that brackets
each execution unit with events + lineage edges; on top sit WorkflowIR,
BoundWorkflow, CapabilityRegistry, TestSpec, executors, approval gates,
replay, and the ``molexp plan`` CLI path.

Dependency direction: ``molexp.harness → molexp.workspace`` plus a single
sanctioned edge ``molexp.harness → molexp.agent.router`` (the SDK-free
Protocol module — see spec ``harness-as-mode-substrate-03a``).
Imports from ``molexp.workflow``, ``molexp.plugins``, ``molexp.server``,
``molexp.cli``, or ``molexp.sweep`` are forbidden, and ``pydantic_ai`` /
``pydantic_graph`` must not be transitively pulled in — see
``tests/test_harness/test_import_guard.py``.
"""

from __future__ import annotations

from molexp.harness.audit import (
    find_last_successful_stage,
    generate_audit_report,
    replay_metadata,
)
from molexp.harness.core import HarnessRunContext, Stage, StageRunner
from molexp.harness.errors import (
    AgentResponseNotRegisteredError,
    ArtifactNotFoundError,
    CapabilityAlreadyRegisteredError,
    CapabilityCallValidationError,
    CapabilityNotFoundError,
    EventSeqConflictError,
    HarnessError,
    StageExecutionError,
    StagePersistedFailureError,
)
from molexp.harness.executors import DryRunExecutor, Executor, LocalExecutor
from molexp.harness.gateways import AgentGateway, RouterBackedAgentGateway
from molexp.harness.mode import Mode
from molexp.harness.modes import PlanMode, RunMode
from molexp.harness.policy import (
    evaluate_approval_policy,
    make_final_report_approval_request,
    record_approval_decision,
    record_approval_request,
)
from molexp.harness.registry import CapabilityRegistry, InMemoryCapabilityRegistry
from molexp.harness.schemas import (
    WELL_KNOWN_ARTIFACT_KINDS,
    AgentCallResult,
    AgentCallSpec,
    ApprovalDecision,
    ApprovalIntent,
    ApprovalPolicy,
    ApprovalRequest,
    ArtifactKind,
    ArtifactRef,
    AuditReport,
    BoundTask,
    BoundWorkflow,
    CommandResult,
    CommandSpec,
    DependencyEdge,
    EventType,
    ExecutionEnvironment,
    ExecutionResult,
    ExpectedOutput,
    ExperimentReport,
    FinalReport,
    HarnessEvent,
    ModeResult,
    ParameterSource,
    ParameterValue,
    PathPolicy,
    ResourcePolicy,
    TaskIR,
    TestKind,
    TestResult,
    TestSource,
    TestSpec,
    TestStatus,
    ToolCapability,
    ToolPolicy,
    UserPlan,
    ValidationReport,
    ValidationViolation,
    WorkflowIR,
    WorkflowSource,
)
from molexp.harness.stages import (
    ApprovalGate,
    Approver,
    BindMolcraftsTasks,
    ExecuteTests,
    ExecuteWorkflow,
    ExtractWorkflowIR,
    GenerateAuditReport,
    GenerateExperimentReport,
    GenerateFinalReport,
    GenerateTestCode,
    GenerateTestSpec,
    GenerateWorkflowSource,
    MaterializeExecution,
    SaveUserPlan,
    ValidateBoundWorkflow,
    ValidateTestSource,
    ValidateTestSpec,
    ValidateWorkflowIR,
    ValidateWorkflowSource,
    auto_grant_approver,
)
from molexp.harness.store import (
    ArtifactLineageStore,
    ArtifactStore,
    EventLog,
    FileArtifactStore,
    SQLiteArtifactLineageStore,
    SQLiteEventLog,
)
from molexp.harness.validators import (
    validate_bound_workflow,
    validate_provenance,
    validate_test_source,
    validate_test_spec,
    validate_workflow_ir,
    validate_workflow_source,
)

__all__ = [
    "WELL_KNOWN_ARTIFACT_KINDS",
    "AgentCallResult",
    "AgentCallSpec",
    "AgentGateway",
    "AgentResponseNotRegisteredError",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalIntent",
    "ApprovalPolicy",
    "ApprovalRequest",
    "Approver",
    "ArtifactKind",
    "ArtifactLineageStore",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactStore",
    "AuditReport",
    "BindMolcraftsTasks",
    "BoundTask",
    "BoundWorkflow",
    "CapabilityAlreadyRegisteredError",
    "CapabilityCallValidationError",
    "CapabilityNotFoundError",
    "CapabilityRegistry",
    "CommandResult",
    "CommandSpec",
    "DependencyEdge",
    "DryRunExecutor",
    "EventLog",
    "EventSeqConflictError",
    "EventType",
    "ExecuteTests",
    "ExecuteWorkflow",
    "ExecutionEnvironment",
    "ExecutionResult",
    "Executor",
    "ExpectedOutput",
    "ExperimentReport",
    "ExtractWorkflowIR",
    "FileArtifactStore",
    "FinalReport",
    "GenerateAuditReport",
    "GenerateExperimentReport",
    "GenerateFinalReport",
    "GenerateTestCode",
    "GenerateTestSpec",
    "GenerateWorkflowSource",
    "HarnessError",
    "HarnessEvent",
    "HarnessRunContext",
    "InMemoryCapabilityRegistry",
    "LocalExecutor",
    "MaterializeExecution",
    "Mode",
    "ModeResult",
    "ParameterSource",
    "ParameterValue",
    "PathPolicy",
    "PlanMode",
    "ResourcePolicy",
    "RouterBackedAgentGateway",
    "RunMode",
    "SQLiteArtifactLineageStore",
    "SQLiteEventLog",
    "SaveUserPlan",
    "Stage",
    "StageExecutionError",
    "StagePersistedFailureError",
    "StageRunner",
    "TaskIR",
    "TestKind",
    "TestResult",
    "TestSource",
    "TestSpec",
    "TestStatus",
    "ToolCapability",
    "ToolPolicy",
    "UserPlan",
    "ValidateBoundWorkflow",
    "ValidateTestSource",
    "ValidateTestSpec",
    "ValidateWorkflowIR",
    "ValidateWorkflowSource",
    "ValidationReport",
    "ValidationViolation",
    "WorkflowIR",
    "WorkflowSource",
    "auto_grant_approver",
    "evaluate_approval_policy",
    "find_last_successful_stage",
    "generate_audit_report",
    "make_final_report_approval_request",
    "record_approval_decision",
    "record_approval_request",
    "replay_metadata",
    "validate_bound_workflow",
    "validate_provenance",
    "validate_test_source",
    "validate_test_spec",
    "validate_workflow_ir",
    "validate_workflow_source",
]
