"""Native workspace-structure tools: list/create projects, experiments, runs.

All tools use the harness ``ToolContext`` shape:

    async def tool(args: dict, ctx: ToolContext) -> ToolResult: ...

and are tagged with :func:`native_tool` so :class:`AgentService` picks
them up at construction time.
"""

from __future__ import annotations

from typing import Any

from molexp.agent.tools.native._helpers import (
    err,
    get_experiment,
    get_project,
    ok,
    workspace,
)
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec


@native_tool(ToolSpec(
    name="native:list_projects",
    description="List every project in the current workspace.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    category="workspace",
    mutates=False,
))
async def list_projects(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    ws, failure = workspace(ctx)
    if failure is not None:
        return failure
    rows = [
        {
            "id": project.id,
            "name": project.name,
            "description": getattr(project, "description", "") or "",
        }
        for project in ws.list_projects()
    ]
    return ok(rows)


@native_tool(ToolSpec(
    name="native:list_experiments",
    description="List experiments inside a project.",
    input_schema={
        "type": "object",
        "properties": {"project_id": {"type": "string"}},
        "required": ["project_id"],
    },
    category="workspace",
    mutates=False,
))
async def list_experiments(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    project, failure = get_project(ctx, args["project_id"])
    if failure is not None:
        return failure
    rows = [
        {
            "id": exp.id,
            "name": exp.name,
            "params": dict(getattr(exp, "params", {}) or {}),
            "has_workflow": exp.workflow is not None,
        }
        for exp in project.list_experiments()
    ]
    return ok(rows)


@native_tool(ToolSpec(
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
))
async def list_runs(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment, failure = get_experiment(ctx, args["project_id"], args["experiment_id"])
    if failure is not None:
        return failure
    rows = [
        {
            "id": run.id,
            "status": str(run.status),
            "parameters": dict(run.parameters or {}),
        }
        for run in experiment.list_runs()
    ]
    return ok(rows)


@native_tool(ToolSpec(
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
))
async def create_project(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    ws, failure = workspace(ctx)
    if failure is not None:
        return failure
    project = ws.project(args["name"])
    return ok(
        {
            "project_id": project.id,
            "name": project.name,
            "description": args.get("description", ""),
        }
    )
