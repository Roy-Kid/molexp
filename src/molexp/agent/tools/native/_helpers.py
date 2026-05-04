"""Shared helpers for the native-tool package."""

from __future__ import annotations

from typing import Any

from molexp.agent.tools.spec import ToolContext, ToolResult
from molexp.agent.types import AgentFailure, FailureKind


def ok(value: Any = None, **metadata: Any) -> ToolResult:
    return ToolResult(ok=True, value=value, metadata=dict(metadata))


def err(message: str, kind: FailureKind = FailureKind.TOOL_ERROR, **detail: Any) -> ToolResult:
    return ToolResult(
        ok=False,
        error=AgentFailure(kind=kind, message=message, detail=dict(detail)),
    )


def workspace(ctx: ToolContext):
    """Return ``(workspace, None)`` or ``(None, failure)`` if unbound."""

    if ctx.workspace is None:
        return None, err("Tool requires a workspace, but none is bound to this session.")
    return ctx.workspace, None


def get_project(ctx: ToolContext, project_id: str):
    ws, failure = workspace(ctx)
    if failure is not None:
        return None, failure
    project = ws.get_project(project_id)
    if project is None:
        return None, err(f"Project '{project_id}' not found")
    return project, None


def get_experiment(ctx: ToolContext, project_id: str, experiment_id: str):
    project, failure = get_project(ctx, project_id)
    if failure is not None:
        return None, failure
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return None, err(f"Experiment '{experiment_id}' not found")
    return experiment, None


def get_run(ctx: ToolContext, project_id: str, experiment_id: str, run_id: str):
    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None:
        return None, failure
    run = experiment.get_run(run_id)
    if run is None:
        return None, err(f"Run '{run_id}' not found")
    return run, None
