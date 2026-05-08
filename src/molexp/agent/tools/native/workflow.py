"""Native workflow tools: run lifecycle + workflow-IR binding.

Mutating tools (``submit_run``, ``retry_run``, ``set_workflow_from_ir``,
``execute_run``, ``create_experiment``) default to ``mutates=True`` so
:class:`ToolPolicy.needs_approval` gates them in plan mode.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, cast

from molexp._typing import JSONMapping, JSONValue
from molexp.agent.tools.native._helpers import (
    err,
    get_experiment,
    get_project,
    get_run,
    ok,
    require_str_arg,
)
from molexp.agent.tools.native._templates import TEMPLATES, list_templates
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec

if TYPE_CHECKING:
    from molexp.workflow.protocols import RunContextLike
    from molexp.workspace.run import Run

_TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled", "error"}


def _coerce_float(value: JSONValue, *, default: float) -> float:
    """Read a numeric tool-arg cell as ``float``, falling back to *default*.

    Tool ``args`` is a JSON-shaped mapping; numeric cells arrive as
    ``int`` or ``float``. ``None`` and non-numeric cells fall back to
    the caller-supplied default.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _read_run_results(run: Run) -> dict[str, JSONValue]:
    """Pull ``context.results`` off the on-disk ``run.json`` for a freshly-loaded Run."""

    payload = json.loads((run.run_dir / "run.json").read_text())
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        return {}
    results = context.get("results")
    return dict(results) if isinstance(results, dict) else {}


def _status_payload(run: Run) -> dict[str, JSONValue]:
    meta = run.metadata
    error_payload: dict[str, JSONValue] | None = None
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


@native_tool(
    ToolSpec(
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
    )
)
async def submit_run(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    parameters_raw = args.get("parameters") or {}
    parameters: dict[str, JSONValue] = (
        dict(parameters_raw) if isinstance(parameters_raw, dict) else {}
    )
    run = experiment.run(parameters=parameters)
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": parameters,
        }
    )


@native_tool(
    ToolSpec(
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
    )
)
async def get_run_status(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    run_id, fail = require_str_arg(args, "run_id")
    if fail is not None or run_id is None:
        return fail or err("missing run_id")
    run, failure = get_run(ctx, project_id, experiment_id, run_id)
    if failure is not None or run is None:
        return failure or err("run lookup failed")
    return ok(_status_payload(run))


@native_tool(
    ToolSpec(
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
    )
)
async def wait_for_run(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    timeout_seconds = _coerce_float(args.get("timeout_seconds"), default=300.0)
    poll_interval = max(0.1, _coerce_float(args.get("poll_interval"), default=2.0))
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    run_id, fail = require_str_arg(args, "run_id")
    if fail is not None or run_id is None:
        return fail or err("missing run_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    while True:
        run = experiment.get_run(run_id)
        if run is None:
            return err(f"Run '{run_id}' not found")
        payload = _status_payload(run)
        status_raw = payload["status"]
        status = status_raw.lower() if isinstance(status_raw, str) else ""
        if status in _TERMINAL_STATUSES:
            payload["timed_out"] = False
            return ok(payload)
        if time.monotonic() >= deadline:
            payload["timed_out"] = True
            return ok(payload)
        await asyncio.sleep(poll_interval)


@native_tool(
    ToolSpec(
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
    )
)
async def retry_run(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    run_id, fail = require_str_arg(args, "run_id")
    if fail is not None or run_id is None:
        return fail or err("missing run_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    run = experiment.get_run(run_id)
    if run is None:
        return err(f"Run '{run_id}' not found")
    new_run = experiment.run(parameters=dict(run.parameters or {}))
    return ok(
        {
            "source_run_id": run.id,
            "new_run_id": new_run.id,
            "status": str(new_run.status),
        }
    )


@native_tool(
    ToolSpec(
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
    )
)
async def get_run_results(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    run_id, fail = require_str_arg(args, "run_id")
    if fail is not None or run_id is None:
        return fail or err("missing run_id")
    run, failure = get_run(ctx, project_id, experiment_id, run_id)
    if failure is not None or run is None:
        return failure or err("run lookup failed")
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
            "results": _read_run_results(run),
        }
    )


# ── Workflow / experiment binding ────────────────────────────────────────────


@native_tool(
    ToolSpec(
        name="native:list_workflow_templates",
        description="Return the catalog of built-in workflow templates the agent can attach.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        category="workflow",
        mutates=False,
    )
)
async def list_workflow_templates(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    return ok(list_templates())


@native_tool(
    ToolSpec(
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
    )
)
async def create_experiment(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    name, fail = require_str_arg(args, "name")
    if fail is not None or name is None:
        return fail or err("missing name")
    project, failure = get_project(ctx, project_id)
    if failure is not None or project is None:
        return failure or err("project lookup failed")
    template_raw = args.get("template")
    if template_raw is None:
        experiment = project.experiment(name)
        return ok(
            {
                "experiment_id": experiment.id,
                "name": experiment.name,
                "workflow_template": None,
                "has_workflow": experiment.workflow is not None,
            }
        )
    if not isinstance(template_raw, str):
        return err(f"template must be a string; got {type(template_raw).__name__}")
    template = template_raw
    if template not in TEMPLATES:
        return err(f"Unknown workflow template '{template}'. Available: {sorted(TEMPLATES)}")
    fn, description, expected_params = TEMPLATES[template]
    experiment = project.experiment(name)
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


@native_tool(
    ToolSpec(
        name="native:list_task_types",
        description="Return every task-type slug that can appear in a workflow IR.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        category="workflow",
        mutates=False,
    )
)
async def list_task_types(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    from molexp.workflow.registry import default_registry

    return ok(
        [
            {"slug": slug, "description": description}
            for slug, description in default_registry.items()
        ]
    )


@native_tool(
    ToolSpec(
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
    )
)
async def set_workflow_from_ir(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    if experiment.workflow is not None:
        return err(
            f"Experiment '{experiment_id}' already has a workflow bound. "
            "Delete the experiment to rebind."
        )
    workflow_json_raw = args.get("workflow_json")
    if not isinstance(workflow_json_raw, dict):
        return err(f"workflow_json must be a JSON object; got {type(workflow_json_raw).__name__}")
    workflow_json = workflow_json_raw
    try:
        experiment.set_workflow(workflow_json)
    except (KeyError, ValueError) as exc:
        return err(f"Invalid workflow IR: {exc}")
    spec = experiment.workflow
    task_configs = workflow_json.get("task_configs")
    task_count = len(task_configs) if isinstance(task_configs, list) else 0
    return ok(
        {
            "experiment_id": experiment.id,
            "workflow_id": spec.workflow_id if spec is not None else None,
            "task_count": task_count,
            "persisted_to": str(experiment.workflow_path),
        }
    )


@native_tool(
    ToolSpec(
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
    )
)
async def execute_run(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    run_id, fail = require_str_arg(args, "run_id")
    if fail is not None or run_id is None:
        return fail or err("missing run_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    run = experiment.get_run(run_id)
    if run is None:
        return err(f"Run '{run_id}' not found")
    workflow = experiment.workflow
    if workflow is None:
        return err(
            f"Experiment '{experiment_id}' has no workflow attached "
            "(server may have restarted). Re-call create_experiment with the "
            "same name and template to rebind."
        )
    try:
        with run.start() as run_ctx:
            # ty does not always promote a stored attribute (``_context: Context``)
            # to the structurally-equivalent ``_StatusContextLike`` Protocol when
            # checking ``RunContextLike`` assignment; the cast acknowledges the
            # cross-layer duck-typed contract that the test suite exercises.
            await workflow.execute(run_context=cast("RunContextLike", run_ctx))
    except Exception as exc:
        return err(f"Execution failed: {exc!r}", run_id=run.id)
    return ok(
        {
            "run_id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
            "results": _read_run_results(run),
        }
    )
