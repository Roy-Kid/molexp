"""Concrete :class:`molexp.harness.Stage` subclasses for the §3 pipeline.

Phase 2 ships the first two stages — :class:`SaveUserPlan` and
:class:`GenerateExperimentReport`. Each is a thin wrapper that constructs
typed inputs, drives the right artifact-store / agent-gateway call, and
returns a single :class:`ArtifactRef`. The :class:`StageRunner` handles
event emission and provenance wiring around them.
"""

from __future__ import annotations

from molexp.harness.stages.approval_gate import ApprovalGate
from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks
from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR
from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport
from molexp.harness.stages.generate_test_spec import GenerateTestSpec
from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource
from molexp.harness.stages.save_user_plan import SaveUserPlan
from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow
from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR
from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

__all__ = [
    "ApprovalGate",
    "BindMolcraftsTasks",
    "ExtractWorkflowIR",
    "GenerateExperimentReport",
    "GenerateTestSpec",
    "GenerateWorkflowSource",
    "SaveUserPlan",
    "ValidateBoundWorkflow",
    "ValidateWorkflowIR",
    "ValidateWorkflowSource",
]
