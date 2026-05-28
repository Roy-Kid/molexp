"""PlanMode's five first-class stage classes + thread-state carrier.

The substrate's :class:`~molexp.agent.pipeline.execute_pipeline`
walks the five Stage subclasses below in declared order, threading a
typed :class:`PlanThreadState` carrier between them. PlanState
lifecycle transitions are driven by :class:`~molexp.agent.mode.ModePipeline`'s
``lifecycle_validator`` (interpreted by PlanMode's
``_build_lifecycle_validator``); repair routing is driven by registered
:class:`~molexp.agent.repair.RepairPolicy`\\ s on the pipeline.

After the ``plan-mode-pydanticai-rewrite`` collapse, the previous three
LLM-driven stages (``ExploreCapabilities`` + ``SynthesizeCandidates`` +
``SelectPlan``) are replaced by a single
:class:`~molexp.agent.modes.plan.stages.research_and_plan.ResearchAndPlan`
stage that drives one MCP-attached pydantic-ai agent end-to-end.
"""

from __future__ import annotations

from molexp.agent.modes.plan.stages.clarify_intent import ClarifyIntent
from molexp.agent.modes.plan.stages.emit_approved_plan import EmitApprovedPlan
from molexp.agent.modes.plan.stages.preflight_plan_graph import PreflightPlanGraph
from molexp.agent.modes.plan.stages.research_and_plan import ResearchAndPlan
from molexp.agent.modes.plan.stages.synthesize_intent import SynthesizeIntent
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState

__all__ = [
    "ClarifyIntent",
    "EmitApprovedPlan",
    "PlanThreadState",
    "PreflightPlanGraph",
    "ResearchAndPlan",
    "SynthesizeIntent",
]
