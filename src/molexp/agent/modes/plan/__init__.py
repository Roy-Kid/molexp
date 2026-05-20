"""PlanMode — the read-only typed planner (sub-spec 03).

PlanMode turns a user report into an approved typed ``PlanGraph`` and
ends at the ``approve_direction`` gate, emitting an
:class:`ApprovedPlanHandoff`. It writes **no** executable code — only
typed plan artefacts persisted through :class:`PlanFolder`.

Public surface:

- :class:`PlanMode` / :class:`PlanModeConfig` — the harness-based mode.
- :class:`ApprovedPlanHandoff` — PlanMode's sole terminal output.
- :class:`PlanFolder` — the ``kind="agent.plan"`` plan workspace.
- :class:`PlanGraphPreflightReport` — the structural-preflight verdict.
- :class:`CapabilityProbe` / :class:`ProbeResult` — the capability-probe
  protocol and its flat result; :class:`NullCapabilityProbe` is the
  fail-closed fallback impl.
"""

from __future__ import annotations

from molexp.agent.modes.plan._mode import PlanMode, PlanModeConfig
from molexp.agent.modes.plan.capability_probe_null import NullCapabilityProbe
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import AGENT_PLAN_KIND, PlanFolder
from molexp.agent.modes.plan.plan_graph_preflight import (
    PlanGraphPreflightReport,
    run_plan_graph_preflight,
)
from molexp.agent.modes.plan.protocols import CapabilityProbe, ProbeResult

__all__ = [
    "AGENT_PLAN_KIND",
    "ApprovedPlanHandoff",
    "CapabilityProbe",
    "NullCapabilityProbe",
    "PlanFolder",
    "PlanGraphPreflightReport",
    "PlanMode",
    "PlanModeConfig",
    "ProbeResult",
    "run_plan_graph_preflight",
]
