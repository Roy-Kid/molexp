"""Server-side PlanMode background-task runtime.

Runs the harness ``PlanMode`` pipeline as a background ``asyncio.Task`` driven
by the ``plan-tasks`` routes, mirroring the ``agent_runtime`` package. On
completion the generated workflow is persisted onto the experiment so the
existing UI workflow-graph renderer shows it.
"""

from __future__ import annotations

from molexp.server.plan_runtime.gateway import (
    build_plan_gateway,
    reset_plan_gateway_factory,
    set_plan_gateway_factory,
)
from molexp.server.plan_runtime.materialize import materialize_plan_records
from molexp.server.plan_runtime.persist import persist_plan_workflow_to_experiment
from molexp.server.plan_runtime.record import record_plan_outputs
from molexp.server.plan_runtime.registry import PlanTaskRegistry
from molexp.server.plan_runtime.task import PlanTask, PlanTaskStatus

__all__ = [
    "PlanTask",
    "PlanTaskRegistry",
    "PlanTaskStatus",
    "build_plan_gateway",
    "materialize_plan_records",
    "persist_plan_workflow_to_experiment",
    "record_plan_outputs",
    "reset_plan_gateway_factory",
    "set_plan_gateway_factory",
]
