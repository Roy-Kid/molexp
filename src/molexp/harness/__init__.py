"""molexp.harness — provenance-first scientific workflow harness.

Top-level package, sibling of :mod:`molexp.workspace`, :mod:`molexp.workflow`,
and :mod:`molexp.agent`. Owns state, provenance, validation, approval,
execution, and audit for harness-driven runs; agents are demoted to
proposal generators behind hard boundaries defined in
``.claude/notes/harness-goal.md``.

Phase 1 (this module's current content) ships only the **state substrate**:
typed artifact references, an append-only event log, a provenance edge
store, and a Stage / StageRunner wrapper that brackets each execution unit
with events + lineage. Later Phases add WorkflowIR, BoundWorkflow,
CapabilityRegistry, TestSpec, executors, approval gates, replay, and CLI.

Dependency direction is one-way: ``molexp.harness → molexp.workspace``.
Imports from ``molexp.workflow``, ``molexp.agent``, ``molexp.plugins``,
``molexp.server``, ``molexp.cli``, or ``molexp.sweep`` are forbidden — see
``tests/test_harness/test_import_guard.py``.
"""

from __future__ import annotations

from molexp.harness.agents import AgentGateway
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
    ExpectedOutput,
    ExperimentReport,
    HarnessEvent,
    ParameterSource,
    ParameterValue,
    PathPolicy,
    ResourcePolicy,
    TaskIR,
    TestKind,
    TestResult,
    TestSpec,
    TestStatus,
    ToolCapability,
    ToolPolicy,
    UserPlan,
    ValidationReport,
    ValidationViolation,
    WorkflowIR,
)
from molexp.harness.stages import (
    ApprovalGate,
    BindMolcraftsTasks,
    ExtractWorkflowIR,
    GenerateExperimentReport,
    GenerateTestSpec,
    SaveUserPlan,
    ValidateBoundWorkflow,
    ValidateWorkflowIR,
)
from molexp.harness.store import (
    ArtifactStore,
    EventLog,
    FileArtifactStore,
    ProvenanceStore,
    SQLiteEventLog,
    SQLiteProvenanceStore,
)
from molexp.harness.validators import (
    validate_bound_workflow,
    validate_provenance,
    validate_test_spec,
    validate_workflow_ir,
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
    "ArtifactKind",
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
    "ExecutionEnvironment",
    "Executor",
    "ExpectedOutput",
    "ExperimentReport",
    "ExtractWorkflowIR",
    "FileArtifactStore",
    "GenerateExperimentReport",
    "GenerateTestSpec",
    "HarnessError",
    "HarnessEvent",
    "HarnessRunContext",
    "InMemoryCapabilityRegistry",
    "LocalExecutor",
    "ParameterSource",
    "ParameterValue",
    "PathPolicy",
    "ProvenanceStore",
    "ResourcePolicy",
    "SQLiteEventLog",
    "SQLiteProvenanceStore",
    "SaveUserPlan",
    "Stage",
    "StageExecutionError",
    "StagePersistedFailureError",
    "StageRunner",
    "TaskIR",
    "TestKind",
    "TestResult",
    "TestSpec",
    "TestStatus",
    "ToolCapability",
    "ToolPolicy",
    "UserPlan",
    "ValidateBoundWorkflow",
    "ValidateWorkflowIR",
    "ValidationReport",
    "ValidationViolation",
    "WorkflowIR",
    "evaluate_approval_policy",
    "find_last_successful_stage",
    "generate_audit_report",
    "make_final_report_approval_request",
    "record_approval_decision",
    "record_approval_request",
    "replay_metadata",
    "validate_bound_workflow",
    "validate_provenance",
    "validate_test_spec",
    "validate_workflow_ir",
]
