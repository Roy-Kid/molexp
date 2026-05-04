"""Native workflow tools: run lifecycle + workflow-IR binding.

Mutating tools (``submit_run``, ``retry_run``, ``set_workflow_from_ir``,
``execute_run``, ``create_experiment``) default to ``mutates=True`` so
:class:`ToolPolicy.needs_approval` gates them in plan mode.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from molexp.agent.tools.native._helpers import (
    err,
    get_experiment,
    get_project,
    get_run,
    ok,
)
from molexp.agent.tools.native._templates import TEMPLATES, list_templates
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec


_TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled", "error"}


def _read_run_results(run: Any) -> dict[str, Any]:
    """Pull ``context.results`` off the on-disk ``run.json`` for a freshly-loaded Run."""

    payload = json.loads((run.run_dir / "run.json").read_text())
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        return {}
    results = context.get("results")
    return dict(results) if isinstance(results, dict) else {}


def _status_payload(run: Any) -> dict[str, Any]:
    meta = run.metadata
    error_payload: dict[str, str] | None = None
    if meta.error is not None:
        error_payload = {"type": meta.error.type, "message": meta.error.message}
    return {
        "run_id": run.id,
        "status": str(run.status),
        "started_at": meta.created_at.isoformat(),
        "finished_at": meta.finished_at.isoformat() if meta.finished_at else None,
        "error": error_payload,
    }


# ── Run lifecycle ────────────────────────────────────────────────────────────


@native_tool(ToolSpec(
    name="native:submit_run",
    description="Create a new run with parameters in an experiment.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "parameters": {"type": "object"},
        },
        "required": ["project_id", "experiment_id", "parameters"],
    },
    category="workflow",
    mutates=True,
))
async def submit_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment, failure = get_experiment(ctx, args["project_id"], args["experiment_id"])
    if failure is not None:
        return failure
    parameters = dict(args.get("parameters") or {})
    run = experiment.run(parameters=parameters)
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": parameters,
        }
    )


@native_tool(ToolSpec(
    name="native:get_run_status",
    description="Read the current status of a run from on-disk metadata.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
        },
        "required": ["project_id", "experiment_id", "run_id"],
    },
    category="workflow",
    mutates=False,
))
async def get_run_status(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    run, failure = get_run(
        ctx, args["project_id"], args["experiment_id"], args["run_id"]
    )
    if failure is not None:
        return failure
    return ok(_status_payload(run))


@native_tool(ToolSpec(
    name="native:wait_for_run",
    description="Poll a run until it reaches a terminal status or timeout.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
            "timeout_seconds": {"type": "number"},
            "poll_interval": {"type": "number"},
        },
        "required": ["project_id", "experiment_id", "run_id"],
    },
    category="workflow",
    mutates=False,
))
async def wait_for_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    timeout_seconds = float(args.get("timeout_seconds", 300.0))
    poll_interval = max(0.1, float(args.get("poll_interval", 2.0)))
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    experiment, failure = get_experiment(
        ctx, args["project_id"], args["experiment_id"]
    )
    if failure is not None:
        return failure
    run_id = args["run_id"]
    while True:
        run = experiment.get_run(run_id)
        if run is None:
            return err(f"Run '{run_id}' not found")
        payload = _status_payload(run)
        status = payload["status"].lower()
        if status in _TERMINAL_STATUSES:
            payload["timed_out"] = False
            return ok(payload)
        if time.monotonic() >= deadline:
            payload["timed_out"] = True
            return ok(payload)
        await asyncio.sleep(poll_interval)


@native_tool(ToolSpec(
    name="native:retry_run",
    description="Clone an existing run's parameters into a fresh run.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
        },
        "required": ["project_id", "experiment_id", "run_id"],
    },
    category="workflow",
    mutates=True,
))
async def retry_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment, failure = get_experiment(ctx, args["project_id"], args["experiment_id"])
    if failure is not None:
        return failure
    run = experiment.get_run(args["run_id"])
    if run is None:
        return err(f"Run '{args['run_id']}' not found")
    new_run = experiment.run(parameters=dict(run.parameters or {}))
    return ok(
        {
            "source_run_id": run.id,
            "new_run_id": new_run.id,
            "status": str(new_run.status),
        }
    )


@native_tool(ToolSpec(
    name="native:get_run_results",
    description="Read the final ctx.set_result payload of a run.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
        },
        "required": ["project_id", "experiment_id", "run_id"],
    },
    category="workflow",
    mutates=False,
))
async def get_run_results(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    run, failure = get_run(
        ctx, args["project_id"], args["experiment_id"], args["run_id"]
    )
    if failure is not None:
        return failure
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
            "results": _read_run_results(run),
        }
    )


# ── Workflow / experiment binding ────────────────────────────────────────────


@native_tool(ToolSpec(
    name="native:list_workflow_templates",
    description="Return the catalog of built-in workflow templates the agent can attach.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    category="workflow",
    mutates=False,
))
async def list_workflow_templates(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ok(list_templates())


@native_tool(ToolSpec(
    name="native:create_experiment",
    description="Create an experiment, optionally binding a built-in workflow template.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "name": {"type": "string"},
            "template": {"type": "string"},
        },
        "required": ["project_id", "name"],
    },
    category="workflow",
    mutates=True,
))
async def create_experiment(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    project, failure = get_project(ctx, args["project_id"])
    if failure is not None:
        return failure
    template = args.get("template")
    if template is None:
        experiment = project.experiment(args["name"])
        return ok(
            {
                "experiment_id": experiment.id,
                "name": experiment.name,
                "workflow_template": None,
                "has_workflow": experiment.workflow is not None,
            }
        )
    if template not in TEMPLATES:
        return err(
            f"Unknown workflow template '{template}'. Available: {sorted(TEMPLATES)}"
        )
    fn, description, expected_params = TEMPLATES[template]
    experiment = project.experiment(args["name"])
    if experiment.workflow is None:
        experiment.set_workflow(fn)
    return ok(
        {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "workflow_template": template,
            "description": description,
            "expected_parameters": list(expected_params),
        }
    )


@native_tool(ToolSpec(
    name="native:list_task_types",
    description="Return every task-type slug that can appear in a workflow IR.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    category="workflow",
    mutates=False,
))
async def list_task_types(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from molexp.workflow.registry import default_registry

    return ok(
        [
            {"slug": slug, "description": description}
            for slug, description in default_registry.items()
        ]
    )


@native_tool(ToolSpec(
    name="native:set_workflow_from_ir",
    description="Bind a JSON workflow IR to an experiment and persist it to disk.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "workflow_json": {"type": "object"},
        },
        "required": ["project_id", "experiment_id", "workflow_json"],
    },
    category="workflow",
    mutates=True,
))
async def set_workflow_from_ir(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment, failure = get_experiment(ctx, args["project_id"], args["experiment_id"])
    if failure is not None:
        return failure
    if experiment.workflow is not None:
        return err(
            f"Experiment '{args['experiment_id']}' already has a workflow bound. "
            "Delete the experiment to rebind."
        )
    workflow_json = args["workflow_json"]
    try:
        experiment.set_workflow(workflow_json)
    except (KeyError, ValueError) as exc:
        return err(f"Invalid workflow IR: {exc}")
    spec = experiment.workflow
    return ok(
        {
            "experiment_id": experiment.id,
            "workflow_id": spec.workflow_id if spec is not None else None,
            "task_count": len(workflow_json.get("task_configs", [])),
            "persisted_to": str(experiment.workflow_ir_path),
        }
    )


@native_tool(ToolSpec(
    name="native:execute_run",
    description="Run the workflow attached to an experiment against an existing run.",
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "run_id": {"type": "string"},
        },
        "required": ["project_id", "experiment_id", "run_id"],
    },
    category="workflow",
    mutates=True,
))
async def execute_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment, failure = get_experiment(ctx, args["project_id"], args["experiment_id"])
    if failure is not None:
        return failure
    run = experiment.get_run(args["run_id"])
    if run is None:
        return err(f"Run '{args['run_id']}' not found")
    workflow = experiment.workflow
    if workflow is None:
        return err(
            f"Experiment '{args['experiment_id']}' has no workflow attached "
            "(server may have restarted). Re-call create_experiment with the "
            "same name and template to rebind."
        )
    try:
        await workflow.execute(run=run)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the agent
        return err(f"Execution failed: {exc!r}", run_id=run.id)
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
            "results": _read_run_results(run),
        }
    )
