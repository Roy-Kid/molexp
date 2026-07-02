"""PlanMode application-service runtime, shared by the CLI and the server.

Runs the harness ``PlanMode`` pipeline as a background ``asyncio.Task`` driven
by the server's ``plan-tasks`` routes, and provides the gateway builder +
agent-stack preflight ``molexp plan`` drives directly. On completion the
generated workflow is persisted onto the experiment so the existing UI
workflow-graph renderer shows it.
"""

from __future__ import annotations

from molexp.services.plan_runtime.drive import drive_plan_mode
from molexp.services.plan_runtime.gateway import (
    PlanPreflightError,
    build_plan_gateway,
    preflight_plan_router,
    reset_plan_gateway_factory,
    set_plan_gateway_factory,
)
from molexp.services.plan_runtime.materialize import materialize_plan_records
from molexp.services.plan_runtime.persist import persist_plan_workflow_to_experiment
from molexp.services.plan_runtime.record import record_plan_outputs
from molexp.services.plan_runtime.registry import PlanTaskRegistry
from molexp.services.plan_runtime.task import PlanTask, PlanTaskStatus

__all__ = [
    "PlanPreflightError",
    "PlanTask",
    "PlanTaskRegistry",
    "PlanTaskStatus",
    "build_plan_gateway",
    "drive_plan_mode",
    "materialize_plan_records",
    "persist_plan_workflow_to_experiment",
    "preflight_plan_router",
    "record_plan_outputs",
    "reset_plan_gateway_factory",
    "set_plan_gateway_factory",
]
