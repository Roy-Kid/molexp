"""PlanMode application-service runtime, shared by the CLI and the server.

Runs the harness ``PlanMode`` pipeline as a background ``asyncio.Task`` driven
by the server's ``plan-tasks`` routes, and provides the gateway builder +
agent-stack preflight ``molexp plan`` drives directly. On completion the
generated workflow is persisted onto the experiment so the existing UI
workflow-graph renderer shows it.
"""

from __future__ import annotations

from molexp.services.plan_runtime.decide import decide_plan_review, map_review_action_to_approval
from molexp.services.plan_runtime.drive import drive_plan_mode
from molexp.services.plan_runtime.gateway import (
    PlanPreflightError,
    build_plan_gateway,
    preflight_plan_router,
    reset_plan_gateway_factory,
    set_plan_gateway_factory,
)
from molexp.services.plan_runtime.materialize import (
    PlanFailure,
    PlanRecordError,
    PlanRecordOutcome,
    materialize_plan_records,
)
from molexp.services.plan_runtime.persist import persist_plan_workflow_to_experiment
from molexp.services.plan_runtime.preview import (
    build_review_pack,
    render_approval_preview,
    render_review_pack,
)
from molexp.services.plan_runtime.registry import PlanTaskRegistry
from molexp.services.plan_runtime.targets import resolve_plan_compute_target
from molexp.services.plan_runtime.task import PlanTask, PlanTaskStatus

__all__ = [
    "PlanFailure",
    "PlanPreflightError",
    "PlanRecordError",
    "PlanRecordOutcome",
    "PlanTask",
    "PlanTaskRegistry",
    "PlanTaskStatus",
    "build_plan_gateway",
    "build_review_pack",
    "decide_plan_review",
    "drive_plan_mode",
    "map_review_action_to_approval",
    "materialize_plan_records",
    "persist_plan_workflow_to_experiment",
    "preflight_plan_router",
    "render_approval_preview",
    "render_review_pack",
    "reset_plan_gateway_factory",
    "resolve_plan_compute_target",
    "set_plan_gateway_factory",
]
