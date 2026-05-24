"""PlanMode's seven first-class stage classes + thread-state carrier.

The substrate's :class:`~molexp.agent.harness.pipeline.execute_pipeline`
walks the seven Stage subclasses below in declared order, threading a
typed :class:`PlanThreadState` carrier between them. PlanState
lifecycle transitions are driven by :class:`~molexp.agent.mode.ModePipeline`'s
``lifecycle_validator`` (interpreted by PlanMode's
``_build_lifecycle_validator``); repair routing is driven by registered
:class:`~molexp.agent.harness.repair.RepairPolicy`\\ s on the pipeline.

The seven stages replace what was a hand-written 7-stage ``while`` loop
in PlanMode's ``_mode.py`` plus the mutable ``_StageOutcome`` scratchpad
plus the manual repair-loop counter — all now declarative.
"""

from __future__ import annotations

from molexp.agent.modes.plan.stages.clarify_intent import ClarifyIntent
from molexp.agent.modes.plan.stages.emit_approved_plan import EmitApprovedPlan
from molexp.agent.modes.plan.stages.explore_capabilities import ExploreCapabilities
from molexp.agent.modes.plan.stages.preflight_plan_graph import PreflightPlanGraph
from molexp.agent.modes.plan.stages.select_plan import SelectPlan
from molexp.agent.modes.plan.stages.synthesize_candidates import SynthesizeCandidates
from molexp.agent.modes.plan.stages.synthesize_intent import SynthesizeIntent
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState

__all__ = [
    "ClarifyIntent",
    "EmitApprovedPlan",
    "ExploreCapabilities",
    "PlanThreadState",
    "PreflightPlanGraph",
    "SelectPlan",
    "SynthesizeCandidates",
    "SynthesizeIntent",
]
