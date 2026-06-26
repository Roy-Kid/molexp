"""Shared registry of the plan pipeline's LLM agents.

The CLI (``cli/plan_cmd.py``) and the server (``server/plan_runtime/gateway.py``)
each build their own :class:`RouterBackedAgentGateway`, but they MUST agree on
*which* agents exist, what schema each returns, what artifact kind its output
is persisted under, and which system prompt drives it. This module is the one
source of those three maps so the two shells can never drift — the Python
(CLI) path and the UI (server) path register identical agents.

Keys are the ``agent_name``s the harness planning/codegen stages set on their
:class:`AgentCallSpec`. Schema imports are pure pydantic; prompt imports are
plain strings — importing this module keeps the harness import-guard green.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["plan_agent_responses", "plan_output_kinds", "plan_system_prompts"]


def plan_agent_responses() -> dict[str, type[BaseModel]]:
    """Map each plan ``agent_name`` to the pydantic schema it must return."""
    from molexp.harness.schemas import (
        BoundWorkflow,
        CapabilitySelection,
        ExperimentReport,
        ExperimentSpec,
        FinalReport,
        InputSet,
        PlanReview,
        TestSource,
        TestSpecBundle,
        WorkflowIR,
        WorkflowSource,
    )

    return {
        "experiment_report_writer": ExperimentReport,
        "experiment_spec_generator": ExperimentSpec,
        "capability_selector": CapabilitySelection,
        "workflow_ir_extractor": WorkflowIR,
        "bound_workflow_binder": BoundWorkflow,
        "workflow_source_writer": WorkflowSource,
        "input_set_generator": InputSet,
        "plan_reviewer": PlanReview,
        "test_spec_writer": TestSpecBundle,
        "test_code_writer": TestSource,
        "final_report_writer": FinalReport,
    }


def plan_output_kinds() -> dict[str, str]:
    """Map each plan ``agent_name`` to the artifact kind its output is stored as."""
    return {
        "experiment_report_writer": "experiment_report",
        "experiment_spec_generator": "experiment_spec",
        "capability_selector": "capability_selection",
        "workflow_ir_extractor": "workflow_ir",
        "bound_workflow_binder": "bound_workflow",
        "workflow_source_writer": "workflow_source",
        "input_set_generator": "input_set",
        "plan_reviewer": "plan_review",
        "test_spec_writer": "test_spec",
        "test_code_writer": "test_source",
        "final_report_writer": "final_report",
    }


def plan_system_prompts() -> dict[str, str]:
    """Map each plan ``agent_name`` to its shipped system prompt."""
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )

    return {
        **prompts_by_agent(),
        "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
    }
