"""PlanMode — the read-only typed planner.

PlanMode turns a user report into an approved typed ``PlanGraph`` and
ends at the ``approve_direction`` gate, emitting an
:class:`ApprovedPlanHandoff`. It writes **no** executable code — only
typed plan artefacts persisted through :class:`PlanFolder`.

After the ``plan-mode-pydanticai-rewrite`` collapse, the pipeline is
**5 stages, 2 LLM operations**:

1. ``SynthesizeIntent`` — one structured call → ``IntentSpec``.
2. ``ClarifyIntent`` — pure routing.
3. ``ResearchAndPlan`` — one MCP-attached agentic call → ``PlanGraph``
   with ``api_refs`` + ``composition_notes`` inline on each step.
4. ``PreflightPlanGraph`` — pure structural check.
5. ``EmitApprovedPlan`` — approval gate + handoff.

Public surface:

- :class:`PlanMode` / :class:`PlanModeConfig` — the harness-based mode.
- :class:`ApprovedPlanHandoff` — PlanMode's sole terminal output.
- :class:`PlanFolder` — the ``kind="agent.plan"`` plan workspace.
- :class:`PlanGraphPreflightReport` — the structural-preflight verdict.
"""

from __future__ import annotations

from molexp.agent.modes.plan._mode import PlanMode, PlanModeConfig
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import AGENT_PLAN_KIND, PlanFolder
from molexp.agent.modes.plan.plan_graph_preflight import (
    PlanGraphPreflightReport,
    preflight_plan_graph,
)

__all__ = [
    "AGENT_PLAN_KIND",
    "ApprovedPlanHandoff",
    "PlanFolder",
    "PlanGraphPreflightReport",
    "PlanMode",
    "PlanModeConfig",
    "preflight_plan_graph",
]
