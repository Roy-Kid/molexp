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
    ArtifactRef,
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
from molexp.harness.schemas.mode_result import ModeResult
from molexp.harness.schemas.parameter import ParameterSource, ParameterValue
from molexp.harness.schemas.plan_review import PlanReview, PlanReviewFinding
from molexp.harness.schemas.policy import ApprovalPolicy, PathPolicy, ToolPolicy
from molexp.harness.schemas.test_source import TestSource
from molexp.harness.schemas.test_spec import (
    TestKind,
    TestResult,
    TestSpec,
    TestSpecBundle,
    TestStatus,
)
from molexp.harness.schemas.user_plan import UserPlan
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import (
    DependencyEdge,
    ExpectedOutput,
    TaskIR,
    WorkflowIR,
)
from molexp.harness.schemas.workflow_source import GeneratedFile, WorkflowSource

__all__ = [
    "WELL_KNOWN_ARTIFACT_KINDS",
    "AgentCallResult",
    "AgentCallSpec",
    "ApprovalDecision",
    "ApprovalIntent",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ArtifactKind",
    "ArtifactRef",
    "AuditReport",
    "BoundTask",
    "BoundWorkflow",
    "CapabilityInvocationResult",
    "CapabilitySelection",
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
    "GeneratedFile",
    "HarnessEvent",
    "InputSet",
    "ModeResult",
    "ParameterSource",
    "ParameterValue",
    "PathPolicy",
    "PlanReview",
    "PlanReviewFinding",
    "ResolvedQuestion",
    "ResourcePolicy",
    "SelectedCapability",
    "SpecCondition",
    "SpecVariable",
    "SweepAxis",
    "SweepStrategy",
    "TaskIR",
    "TestKind",
    "TestResult",
    "TestSource",
    "TestSpec",
    "TestSpecBundle",
    "TestStatus",
    "ToolCapability",
    "ToolPolicy",
    "UserPlan",
    "ValidationReport",
    "ValidationViolation",
    "WorkflowIR",
    "WorkflowSource",
]
