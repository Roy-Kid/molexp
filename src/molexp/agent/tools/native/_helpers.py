"""Shared helpers for the native-tool package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp._typing import JSONMapping, JSONValue, TaskOutput
from molexp.agent.tools.spec import ToolContext, ToolResult
from molexp.agent.types import AgentFailure, FailureKind

if TYPE_CHECKING:
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.project import Project
    from molexp.workspace.run import Run
    from molexp.workspace.workspace import Workspace


def ok(value: TaskOutput = None, **metadata: JSONValue) -> ToolResult:
    return ToolResult(ok=True, value=value, metadata=dict(metadata))


def err(
    message: str, kind: FailureKind = FailureKind.TOOL_ERROR, **detail: JSONValue
) -> ToolResult:
    return ToolResult(
        ok=False,
        error=AgentFailure(kind=kind, message=message, detail=dict(detail)),
    )


def require_str_arg(args: JSONMapping, key: str) -> tuple[str | None, ToolResult | None]:
    """Read ``args[key]`` as a string, returning ``(value, None)`` on success.

    Returns ``(None, failure)`` when the key is absent or holds a
    non-string value, with a typed :class:`AgentFailure` payload the
    caller can return verbatim.
    """
    value = args.get(key)
    if not isinstance(value, str):
        return None, err(
            f"required string argument {key!r} is missing or not a string "
            f"(got {type(value).__name__})",
        )
    return value, None


def workspace(ctx: ToolContext) -> tuple[Workspace | None, ToolResult | None]:
    """Return ``(workspace, None)`` or ``(None, failure)`` if unbound."""
    if ctx.workspace is None:
        return None, err("Tool requires a workspace, but none is bound to this session.")
    return ctx.workspace, None


def get_project(ctx: ToolContext, project_id: str) -> tuple[Project | None, ToolResult | None]:
    from molexp.workspace import ProjectNotFoundError

    ws, failure = workspace(ctx)
    if failure is not None or ws is None:
        return None, failure
    try:
        return ws.project(project_id), None
    except ProjectNotFoundError:
        return None, err(f"Project '{project_id}' not found")


def get_experiment(
    ctx: ToolContext, project_id: str, experiment_id: str
) -> tuple[Experiment | None, ToolResult | None]:
    from molexp.workspace import ExperimentNotFoundError

    project, failure = get_project(ctx, project_id)
    if failure is not None or project is None:
        return None, failure
    try:
        return project.experiment(experiment_id), None
    except ExperimentNotFoundError:
        return None, err(f"Experiment '{experiment_id}' not found")


def get_run(
    ctx: ToolContext, project_id: str, experiment_id: str, run_id: str
) -> tuple[Run | None, ToolResult | None]:
    from molexp.workspace import RunNotFoundError

    experiment, failure = get_experiment(ctx, project_id, experiment_id)
    if failure is not None or experiment is None:
        return None, failure
    try:
        return experiment.run(run_id), None
    except RunNotFoundError:
        return None, err(f"Run '{run_id}' not found")
