"""Frozen pydantic schemas for ``molexp.harness``.

Pure data: every model has ``frozen=True``. Discriminator fields are
``typing.Literal[...]`` aliases, not ``enum.Enum``, matching the workspace
convention.
"""

from __future__ import annotations

from molexp.harness.schemas.agent_call import AgentCallResult, AgentCallSpec
from molexp.harness.schemas.approval import (
    ApprovalDecision,
    ApprovalIntent,
    ApprovalRequest,
)
from molexp.harness.schemas.artifact import (
    WELL_KNOWN_ARTIFACT_KINDS,
    ArtifactKind,
    PlanArtifactRef,
)
from molexp.harness.schemas.audit_report import AuditReport
from molexp.harness.schemas.bound_workflow import (
    BoundTask,
    BoundWorkflow,
    ExecutionEnvironment,
    ResourcePolicy,
)
from molexp.harness.schemas.capability import ToolCapability
from molexp.harness.schemas.capability_invocation import CapabilityInvocationResult
from molexp.harness.schemas.capability_selection import (
    CapabilitySelection,
    SelectedCapability,
)
from molexp.harness.schemas.change_proposal import (
    CHANGE_PROPOSAL_KIND,
    AgentIntent,
    ApprovalLevel,
    ChangeProposal,
    ChangeSpec,
    HighRiskOp,
    ObjectRef,
    ProposalOutcome,
    ProposalStatus,
    Reversibility,
    StateSnapshot,
)
from molexp.harness.schemas.command import CommandResult, CommandSpec
from molexp.harness.schemas.event import EventType, HarnessEvent
from molexp.harness.schemas.execution_report import ExecutionReport
from molexp.harness.schemas.execution_result import ExecutionResult
from molexp.harness.schemas.experiment_report import ExperimentReport
from molexp.harness.schemas.experiment_spec import (
    ExperimentSpec,
    ResolvedQuestion,
    SpecCondition,
    SpecVariable,
)
from molexp.harness.schemas.final_report import FinalReport
from molexp.harness.schemas.input_set import InputSet, SweepAxis, SweepStrategy
from molexp.harness.schemas.mode_result import ModeResult, StageTiming
from molexp.harness.schemas.parameter import ParameterSource, ParameterValue
from molexp.harness.schemas.plan_review import PlanReview, PlanReviewFinding
from molexp.harness.schemas.policy import ApprovalPolicy, PathPolicy, ToolPolicy
from molexp.harness.schemas.step_audit import (
    FindingResolution,
    FormArtifactRefField,
    FormBooleanField,
    FormDocument,
    FormField,
    FormKeyValueField,
    FormMarkdownField,
    FormMultiSelectField,
    FormNumberField,
    FormSelectField,
    FormSelectOption,
    FormTableColumn,
    FormTableField,
    FormTextAreaField,
    FormTextField,
    ReviewDecision,
    ReviewFinding,
    ReviewPack,
    StepAuditPolicy,
    approval_decision_to_review,
    review_decision_to_approval,
)
from molexp.harness.schemas.test_source import TestSource
from molexp.harness.schemas.test_spec import (
    TestKind,
    TestResult,
    TestSpec,
    TestSpecBundle,
    TestStatus,
)
from molexp.harness.schemas.user_plan import UserPlan
from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import (
    DependencyEdge,
    ExpectedOutput,
    PlanTaskIR,
    PlanWorkflowIR,
)
from molexp.harness.schemas.workflow_source import GeneratedFile, WorkflowSource

__all__ = [
    # ChangeProposal / AgentIntent — high-risk-mutation guard (integration P0.5)
    "CHANGE_PROPOSAL_KIND",
    "WELL_KNOWN_ARTIFACT_KINDS",
    "AgentCallResult",
    "AgentCallSpec",
    "AgentIntent",
    "ApprovalDecision",
    "ApprovalIntent",
    "ApprovalLevel",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ArtifactKind",
    "AuditReport",
    "BoundTask",
    "BoundWorkflow",
    "CapabilityInvocationResult",
    "CapabilitySelection",
    "ChangeProposal",
    "ChangeSpec",
    "CommandResult",
    "CommandSpec",
    "DependencyEdge",
    "EventType",
    "ExecutionEnvironment",
    "ExecutionReport",
    "ExecutionResult",
    "ExpectedOutput",
    "ExperimentReport",
    "ExperimentSpec",
    "FinalReport",
    "FindingResolution",
    "FormArtifactRefField",
    "FormBooleanField",
    "FormDocument",
    "FormField",
    "FormKeyValueField",
    "FormMarkdownField",
    "FormMultiSelectField",
    "FormNumberField",
    "FormSelectField",
    "FormSelectOption",
    "FormTableColumn",
    "FormTableField",
    "FormTextAreaField",
    "FormTextField",
    "GeneratedFile",
    "HarnessEvent",
    "HighRiskOp",
    "InputSet",
    "ModeResult",
    "StageTiming",
    "ObjectRef",
    "ParameterSource",
    "ParameterValue",
    "PathPolicy",
    "PlanArtifactRef",
    "PlanReview",
    "PlanReviewFinding",
    "PlanTaskIR",
    "PlanValidationReport",
    "PlanWorkflowIR",
    "ProposalOutcome",
    "ProposalStatus",
    "ResolvedQuestion",
    "ResourcePolicy",
    "Reversibility",
    "ReviewDecision",
    "ReviewFinding",
    "ReviewPack",
    "SelectedCapability",
    "SpecCondition",
    "SpecVariable",
    "StateSnapshot",
    "StepAuditPolicy",
    "SweepAxis",
    "SweepStrategy",
    "TestKind",
    "TestResult",
    "TestSource",
    "TestSpec",
    "TestSpecBundle",
    "TestStatus",
    "ToolCapability",
    "ToolPolicy",
    "UserPlan",
    "ValidationViolation",
    "WorkflowSource",
    "approval_decision_to_review",
    "review_decision_to_approval",
]
