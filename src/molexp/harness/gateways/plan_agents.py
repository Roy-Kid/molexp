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

    from molexp.agent.router import ModelTier

__all__ = [
    "plan_agent_mcp_servers",
    "plan_agent_responses",
    "plan_agent_tiers",
    "plan_output_kinds",
    "plan_system_prompts",
]


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
        PlanWorkflowIR,
        TestSource,
        TestSpecBundle,
        WorkflowSource,
    )

    return {
        "experiment_report_writer": ExperimentReport,
        "experiment_spec_generator": ExperimentSpec,
        # Agentic planner (call_mode="agentic"): molmcp-grounded, studies the
        # toolchain then emits a concrete plan. Pinned to ExperimentSpec for now.
        "create_experiment_plan": ExperimentSpec,
        # Renders an approved frozen plan into a human-readable design report.
        "plan_report_renderer": ExperimentReport,
        "capability_selector": CapabilitySelection,
        "workflow_ir_extractor": PlanWorkflowIR,
        "bound_workflow_binder": BoundWorkflow,
        "workflow_source_writer": WorkflowSource,
        # Per-task variant: GenerateWorkflowSource fans one call per bound task
        # so each pro codegen stays small and consults molmcp for just that task.
        "workflow_source_file_writer": WorkflowSource,
        "input_set_generator": InputSet,
        "plan_reviewer": PlanReview,
        "test_spec_writer": TestSpecBundle,
        "test_code_writer": TestSource,
        # Per-file variant: GenerateTestCode fans one call per task so each
        # v4-pro codegen stays small (the whole-suite call times out on pro).
        "test_code_file_writer": TestSource,
        "final_report_writer": FinalReport,
    }


def plan_agent_tiers() -> dict[str, ModelTier]:
    """Map each plan ``agent_name`` to the model tier it must run at.

    **No gateway fallback.** Every agent registered in
    :func:`plan_agent_responses` MUST appear here; the gateway refuses
    unmapped agents instead of silently using a default tier.

    **Policy:** everything that *writes* plan content runs **HEAVY (pro)** —
    report, spec, IR, bind, codegen (first pass *and* repair), input set,
    final report. Only the **critic** (``plan_reviewer``) is DEFAULT.
    Validators and structural checks are not LLM agents.
    """
    from molexp.agent.router import ModelTier

    heavy = ModelTier.HEAVY
    default = ModelTier.DEFAULT
    return {
        "experiment_report_writer": heavy,
        "experiment_spec_generator": heavy,
        "create_experiment_plan": heavy,
        "plan_report_renderer": heavy,
        "capability_selector": heavy,
        "workflow_ir_extractor": heavy,
        "bound_workflow_binder": heavy,
        "input_set_generator": heavy,
        "test_spec_writer": heavy,
        "final_report_writer": heavy,
        # Codegen authors — always HEAVY (including first pass).
        "workflow_source_writer": heavy,
        "workflow_source_file_writer": heavy,
        "test_code_writer": heavy,
        "test_code_file_writer": heavy,
        # Critic only → DEFAULT.
        "plan_reviewer": default,
    }


def plan_agent_mcp_servers() -> dict[str, tuple[str, ...]]:
    """Map each plan ``agent_name`` to the MCP servers it **requires** while running.

    Code-writing agents must consult **molmcp** (``molcrafts_search`` /
    ``molcrafts_describe`` / ``molcrafts_explore`` / ``molcrafts_usage``) so
    they resolve real molcrafts APIs mid-generation. The application layer
    resolves each name to a concrete stdio spec; a missing molmcp is a
    **hard preflight failure** for these agents (no silent catalog-only
    fallback — guessing APIs is how plans invent symbols).
    """
    return {
        # Agentic planner grounds its plan on the live toolchain.
        "create_experiment_plan": ("molmcp",),
        "workflow_source_writer": ("molmcp",),
        "workflow_source_file_writer": ("molmcp",),
        "test_code_writer": ("molmcp",),
        "test_code_file_writer": ("molmcp",),
    }


def plan_output_kinds() -> dict[str, str]:
    """Map each plan ``agent_name`` to the artifact kind its output is stored as."""
    return {
        "experiment_report_writer": "experiment_report",
        "experiment_spec_generator": "experiment_spec",
        "create_experiment_plan": "experiment_plan",
        "plan_report_renderer": "plan_report",
        "capability_selector": "capability_selection",
        "workflow_ir_extractor": "workflow_ir",
        "bound_workflow_binder": "bound_workflow",
        "workflow_source_writer": "workflow_source",
        # Per-task partials land under their own kind; the merged workflow_source
        # GenerateWorkflowSource assembles stays the single latest.
        "workflow_source_file_writer": "workflow_source_file",
        "input_set_generator": "input_set",
        "plan_reviewer": "plan_review",
        "test_spec_writer": "test_spec",
        "test_code_writer": "test_source",
        # Per-file partials land under their own kind so the merged
        # ``test_source`` GenerateTestCode assembles stays the single latest.
        "test_code_file_writer": "test_source_file",
        "final_report_writer": "final_report",
    }


def plan_system_prompts() -> dict[str, str]:
    """Map each plan ``agent_name`` to its shipped system prompt."""
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.prompts.create_experiment_plan import (
        SYSTEM_PROMPT as CREATE_EXPERIMENT_PLAN_SYSTEM_PROMPT,
    )
    from molexp.harness.prompts.plan_report import SYSTEM_PROMPT as PLAN_REPORT_SYSTEM_PROMPT
    from molexp.harness.prompts.test_code import SYSTEM_PROMPT as TEST_CODE_SYSTEM_PROMPT
    from molexp.harness.prompts.test_code_file import (
        SYSTEM_PROMPT as TEST_CODE_FILE_SYSTEM_PROMPT,
    )
    from molexp.harness.prompts.workflow_source import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_SYSTEM_PROMPT,
    )
    from molexp.harness.prompts.workflow_source_file import (
        SYSTEM_PROMPT as WORKFLOW_SOURCE_FILE_SYSTEM_PROMPT,
    )

    return {
        **prompts_by_agent(),
        "create_experiment_plan": CREATE_EXPERIMENT_PLAN_SYSTEM_PROMPT,
        "plan_report_renderer": PLAN_REPORT_SYSTEM_PROMPT,
        "workflow_source_writer": WORKFLOW_SOURCE_SYSTEM_PROMPT,
        # Per-task writers: short system + YAML user payload (codegen_prompt).
        "workflow_source_file_writer": WORKFLOW_SOURCE_FILE_SYSTEM_PROMPT,
        "test_code_writer": TEST_CODE_SYSTEM_PROMPT,
        "test_code_file_writer": TEST_CODE_FILE_SYSTEM_PROMPT,
    }
