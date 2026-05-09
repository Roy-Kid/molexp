"""Native workspace-structure tools: list/create projects, experiments, runs.

All tools use the harness ``ToolContext`` shape:

    async def tool(args: dict, ctx: ToolContext) -> ToolResult: ...

and are tagged with :func:`native_tool` so :class:`AgentService` picks
them up at construction time.
"""

from __future__ import annotations

from molexp._typing import JSONMapping
from molexp.agent.tools.native._helpers import (
    err,
    get_experiment,
    get_project,
    ok,
    require_str_arg,
    workspace,
)
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec
from molexp.workflow import Workflow


@native_tool(
    ToolSpec(
        name="native:list_projects",
        description="List every project in the current workspace.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        category="workspace",
        mutates=False,
    )
)
async def list_projects(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    ws, failure = workspace(ctx)
    if failure is not None or ws is None:
        return failure or err("workspace lookup failed")
    rows = [
        {
            "id": project.id,
            "name": project.name,
            "description": getattr(project, "description", "") or "",
        }
        for project in ws.list_projects()
    ]
    return ok(rows)


@native_tool(
    ToolSpec(
        name="native:list_experiments",
        description="List experiments inside a project.",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        category="workspace",
        mutates=False,
    )
)
async def list_experiments(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    project, failure = get_project(ctx, project_id)
    if failure is not None or project is None:
        return failure or err("project lookup failed")
    rows = [
        {
            "id": exp.id,
            "name": exp.name,
            "params": dict(getattr(exp, "params", {}) or {}),
            "has_workflow": Workflow.for_experiment(exp) is not None,
        }
        for exp in project.list_experiments()
    ]
    return ok(rows)


@native_tool(
    ToolSpec(
        name="native:list_runs",
        description="List runs inside a project/experiment.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "experiment_id": {"type": "string"},
            },
            "required": ["project_id", "experiment_id"],
        },
        category="workspace",
        mutates=False,
    )
)
async def list_runs(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    project_id, fail = require_str_arg(args, "project_id")
    if fail is not None or project_id is None:
        return fail or err("missing project_id")
    experiment_id, fail = require_str_arg(args, "experiment_id")
    if fail is not None or experiment_id is None:
        return fail or err("missing experiment_id")
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return failure or err("experiment lookup failed")
    rows = [
        {
            "id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
        }
        for run in experiment.list_runs()
    ]
    return ok(rows)


@native_tool(
    ToolSpec(
        name="native:create_project",
        description="Create (or get) a project by name. Idempotent.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
        category="workspace",
        mutates=True,
    )
)
async def create_project(args: JSONMapping, ctx: ToolContext) -> ToolResult:
    name, fail = require_str_arg(args, "name")
    if fail is not None or name is None:
        return fail or err("missing name")
    ws, failure = workspace(ctx)
    if failure is not None or ws is None:
        return failure or err("workspace lookup failed")
    description_raw = args.get("description", "")
    description = description_raw if isinstance(description_raw, str) else ""
    project = ws.project(name)
    return ok(
        {
            "project_id": project.id,
            "name": project.name,
            "description": description,
        }
    )
