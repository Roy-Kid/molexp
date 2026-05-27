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
from molexp.harness.schemas.artifact import ArtifactKind, ArtifactRef
from molexp.harness.schemas.audit_report import AuditReport
from molexp.harness.schemas.bound_workflow import (
    BoundTask,
    BoundWorkflow,
    ExecutionEnvironment,
    ResourcePolicy,
)
from molexp.harness.schemas.capability import ToolCapability
from molexp.harness.schemas.command import CommandResult, CommandSpec
from molexp.harness.schemas.event import EventType, HarnessEvent
from molexp.harness.schemas.experiment_report import ExperimentReport
from molexp.harness.schemas.parameter import ParameterSource, ParameterValue
from molexp.harness.schemas.policy import ApprovalPolicy, PathPolicy, ToolPolicy
from molexp.harness.schemas.test_spec import TestKind, TestResult, TestSpec, TestStatus
from molexp.harness.schemas.user_plan import UserPlan
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import (
    DependencyEdge,
    ExpectedOutput,
    TaskIR,
    WorkflowIR,
)

__all__ = [
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
    "CommandResult",
    "CommandSpec",
    "DependencyEdge",
    "EventType",
    "ExecutionEnvironment",
    "ExpectedOutput",
    "ExperimentReport",
    "HarnessEvent",
    "ParameterSource",
    "ParameterValue",
    "PathPolicy",
    "ResourcePolicy",
    "TaskIR",
    "TestKind",
    "TestResult",
    "TestSpec",
    "TestStatus",
    "ToolCapability",
    "ToolPolicy",
    "UserPlan",
    "ValidationReport",
    "ValidationViolation",
    "WorkflowIR",
]
