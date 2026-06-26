"""Concrete :class:`molexp.harness.Stage` subclasses for the §3 pipeline.

Each stage is a thin wrapper that constructs typed inputs, drives the right
artifact-store / agent-gateway call, and returns a single
:class:`ArtifactRef`. The audit bracket
(:func:`~molexp.harness.core.stage_runner.run_stage_bracketed`) handles
event emission and artifact-lineage wiring around them.
"""

from __future__ import annotations

from molexp.harness.stages.approval_gate import ApprovalGate, Approver, auto_grant_approver
from molexp.harness.stages.bind_molcrafts_tasks import BindMolcraftsTasks
from molexp.harness.stages.compile_workflow import CompileWorkflow
from molexp.harness.stages.execute_tests import ExecuteTests
from molexp.harness.stages.execute_workflow import ExecuteWorkflow
from molexp.harness.stages.extract_workflow_ir import ExtractWorkflowIR
from molexp.harness.stages.generate_audit_report import GenerateAuditReport
from molexp.harness.stages.generate_execution_report import GenerateExecutionReport
from molexp.harness.stages.generate_experiment_report import GenerateExperimentReport
from molexp.harness.stages.generate_experiment_spec import GenerateExperimentSpec
from molexp.harness.stages.generate_final_report import GenerateFinalReport
from molexp.harness.stages.generate_input_set import GenerateInputSet
from molexp.harness.stages.generate_test_code import GenerateTestCode
from molexp.harness.stages.generate_test_spec import GenerateTestSpec
from molexp.harness.stages.generate_workflow_source import GenerateWorkflowSource
from molexp.harness.stages.materialize_execution import MaterializeExecution
from molexp.harness.stages.repair_loop import RepairLoop
from molexp.harness.stages.resolve_capabilities import ResolveCapabilities
from molexp.harness.stages.review_plan import ReviewPlan
from molexp.harness.stages.save_user_plan import SaveUserPlan
from molexp.harness.stages.validate_bound_workflow import ValidateBoundWorkflow
from molexp.harness.stages.validate_experiment_spec import ValidateExperimentSpec
from molexp.harness.stages.validate_input_set import ValidateInputSet
from molexp.harness.stages.validate_test_source import ValidateTestSource
from molexp.harness.stages.validate_test_spec import ValidateTestSpec
from molexp.harness.stages.validate_workflow_ir import ValidateWorkflowIR
from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

__all__ = [
    "ApprovalGate",
    "Approver",
    "BindMolcraftsTasks",
    "CompileWorkflow",
    "ExecuteTests",
    "ExecuteWorkflow",
    "ExtractWorkflowIR",
    "GenerateAuditReport",
    "GenerateExecutionReport",
    "GenerateExperimentReport",
    "GenerateExperimentSpec",
    "GenerateFinalReport",
    "GenerateInputSet",
    "GenerateTestCode",
    "GenerateTestSpec",
    "GenerateWorkflowSource",
    "MaterializeExecution",
    "RepairLoop",
    "ResolveCapabilities",
    "ReviewPlan",
    "SaveUserPlan",
    "ValidateBoundWorkflow",
    "ValidateExperimentSpec",
    "ValidateInputSet",
    "ValidateTestSource",
    "ValidateTestSpec",
    "ValidateWorkflowIR",
    "ValidateWorkflowSource",
    "auto_grant_approver",
]
