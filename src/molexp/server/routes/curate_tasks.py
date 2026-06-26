"""Curate-task routes — run the shared curation flow as a background task.

``POST /projects/{p}/experiments/{e}/curate-tasks`` files a content-addressed
Run under the experiment and starts ``run_curation_flow`` on it in the
background (auto-grant approvals, no LLM blocking the request).
``GET .../{task_id}`` polls status; on completion the selected capability id +
mutation summary are exposed so the UI can reflect the curation that ran.

The route is the UI counterpart to the ``molexp curate`` CLI: same content-
addressed Run, same :func:`~molexp.server.curate_runtime.flow.run_curation_flow`,
reached over HTTP instead of a TTY (Python ≡ UI — one backend code path).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from molexp.server.curate_runtime import CurateTask
    from molexp.workspace import Workspace

__all__ = ["router"]

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/curate-tasks",
    tags=["curate-tasks"],
)

_REQUEST_PREVIEW_CHARS = 80


class CurateTaskCreateRequest(BaseModel):
    """Body for starting a curation-flow background task."""

    request: str = Field(..., description="Natural-language workspace-curation request.")
    model: str | None = Field(None, description="Model id; defaults to the configured agent.model.")


class CurateTaskResponse(BaseModel):
    """One background curate task's current state (UI polls this)."""

    taskId: str
    runId: str
    projectId: str
    experimentId: str
    status: str
    createdAt: str
    model: str
    requestPreview: str
    capabilityId: str | None = None
    mutationSummary: str | None = None
    granted: bool | None = None
    error: str | None = None


class CurateTaskListResponse(BaseModel):
    tasks: list[CurateTaskResponse]
    total: int


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from in-code ``molexp.config``, if any."""
    import molexp
    from molexp.server.operator_config import AGENT_MODEL_KEY

    model = molexp.config.get(AGENT_MODEL_KEY)
    return model if isinstance(model, str) and model else None


def _to_response(task: CurateTask, *, project_id: str, experiment_id: str) -> CurateTaskResponse:
    preview = ""
    stripped = task.request.strip()
    if stripped:
        preview = stripped.splitlines()[0][:_REQUEST_PREVIEW_CHARS]
    result = task.result
    return CurateTaskResponse(
        taskId=task.task_id,
        runId=task.run_id,
        projectId=project_id,
        experimentId=experiment_id,
        status=task.status,
        createdAt=task.created_at,
        model=task.model,
        requestPreview=preview,
        capabilityId=result.capability_id if result is not None else None,
        mutationSummary=result.mutation_summary if result is not None else None,
        granted=result.granted if result is not None else None,
        error=repr(task.error) if task.error is not None else None,
    )


@router.post("", response_model=CurateTaskResponse, status_code=status.HTTP_201_CREATED)
async def create_curate_task(
    project_id: str,
    experiment_id: str,
    request: CurateTaskCreateRequest,
    workspace: Workspace = Depends(get_workspace),
) -> CurateTaskResponse:
    """Start the curation flow on a content-addressed run under the experiment.

    Async so the spawned background ``asyncio.Task`` (the curation flow) attaches
    to the app event loop; the handler itself does no awaiting and returns the
    initial ``running`` status immediately.
    """
    from molexp._typing import JSONValue
    from molexp.ids import generate_id
    from molexp.server.curate_runtime.gateway import build_curate_gateway
    from molexp.server.deps.curate_runtime import get_curate_runtime
    from molexp.workspace.errors import RunNotFoundError
    from molexp.workspace.utils import derive_run_id

    text = request.request.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "request is empty")

    model = request.model or _configured_model()
    if not model:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No model configured. Set it with `molexp config set agent.model <id>`.",
        )

    # Workspace NotFound errors map to HTTP envelopes via the registered handlers.
    experiment = workspace.get_project(project_id).get_experiment(experiment_id)

    params: dict[str, JSONValue] = {"mode": "curate", "request": text}
    run_id = derive_run_id(params)
    try:
        run = experiment.get_run(run_id)
    except RunNotFoundError:
        run = experiment.add_run(params, id=run_id)

    gateway = build_curate_gateway(model=model, run=run)
    task = get_curate_runtime().create(
        workspace_root=str(workspace.root),
        task_id=f"curate-{generate_id()}",
        run=run,
        experiment=experiment,
        workspace=workspace,
        request=text,
        model=model,
        created_at=datetime.now(tz=UTC).isoformat(),
        gateway=gateway,
    )
    return _to_response(task, project_id=project_id, experiment_id=experiment_id)


@router.get("", response_model=CurateTaskListResponse)
def list_curate_tasks(
    project_id: str,
    experiment_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> CurateTaskListResponse:
    """List the live curate tasks in this workspace (in-memory; MVP)."""
    from molexp.server.deps.curate_runtime import get_curate_runtime

    tasks = get_curate_runtime().list_tasks(str(workspace.root))
    items = [_to_response(t, project_id=project_id, experiment_id=experiment_id) for t in tasks]
    return CurateTaskListResponse(tasks=items, total=len(items))


@router.get("/{task_id}", response_model=CurateTaskResponse)
def get_curate_task(
    project_id: str,
    experiment_id: str,
    task_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> CurateTaskResponse:
    """Return one curate task's current status."""
    from molexp.server.deps.curate_runtime import get_curate_runtime

    task = get_curate_runtime().get(str(workspace.root), task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"curate task {task_id!r} not found")
    return _to_response(task, project_id=project_id, experiment_id=experiment_id)
