"""Plan-task routes — run the harness ``PlanMode`` pipeline as a background task.

``POST /projects/{p}/experiments/{e}/plan-tasks`` files a content-addressed Run
under the experiment and starts PlanMode on it in the background (auto-grant
approvals, no LLM blocking the request). ``GET .../{task_id}`` polls status; on
completion the generated workflow is persisted onto the experiment, so the
existing workflow-graph renderer (``GET .../workflow``) shows it.

The route is the UI counterpart to the ``molexp plan`` CLI: same content-
addressed Run, same harness pipeline, reached over HTTP instead of a TTY.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from molexp.server.plan_runtime.task import PlanTask
    from molexp.workspace import Workspace

__all__ = ["router"]

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/plan-tasks",
    tags=["plan-tasks"],
)

_DRAFT_PREVIEW_CHARS = 80


class PlanTaskCreateRequest(BaseModel):
    """Body for starting a PlanMode background task."""

    draft: str = Field(..., description="Natural-language experiment draft for PlanMode.")
    model: str | None = Field(None, description="Model id; defaults to the configured agent.model.")


class PlanTaskResponse(BaseModel):
    """One background plan task's current state (UI polls this)."""

    taskId: str
    runId: str
    projectId: str
    experimentId: str
    status: str
    createdAt: str
    model: str
    draftPreview: str
    workflowPersisted: bool = False
    error: str | None = None


class PlanTaskListResponse(BaseModel):
    tasks: list[PlanTaskResponse]
    total: int


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from in-code ``molexp.config``, if any."""
    import molexp
    from molexp.server.operator_config import AGENT_MODEL_KEY

    model = molexp.config.get(AGENT_MODEL_KEY)
    return model if isinstance(model, str) and model else None


def _to_response(task: PlanTask, *, project_id: str, experiment_id: str) -> PlanTaskResponse:
    preview = ""
    stripped = task.draft.strip()
    if stripped:
        preview = stripped.splitlines()[0][:_DRAFT_PREVIEW_CHARS]
    return PlanTaskResponse(
        taskId=task.task_id,
        runId=task.run_id,
        projectId=project_id,
        experimentId=experiment_id,
        status=task.status,
        createdAt=task.created_at,
        model=task.model,
        draftPreview=preview,
        workflowPersisted=task.workflow_persisted,
        error=repr(task.error) if task.error is not None else None,
    )


@router.post("", response_model=PlanTaskResponse, status_code=status.HTTP_201_CREATED)
async def create_plan_task(
    project_id: str,
    experiment_id: str,
    request: PlanTaskCreateRequest,
    workspace: Workspace = Depends(get_workspace),
) -> PlanTaskResponse:
    """Start a PlanMode pipeline on a content-addressed run under the experiment.

    Async so the spawned background ``asyncio.Task`` (the PlanMode run) attaches
    to the app event loop; the handler itself does no awaiting and returns the
    initial ``running`` status immediately.
    """
    from molexp._typing import JSONValue
    from molexp.ids import generate_id
    from molexp.server.deps.plan_runtime import get_plan_runtime
    from molexp.server.plan_runtime.gateway import build_plan_gateway
    from molexp.workspace.errors import RunNotFoundError
    from molexp.workspace.utils import derive_run_id

    draft = request.draft.strip()
    if not draft:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "draft is empty")

    model = request.model or _configured_model()
    if not model:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No model configured. Set it with `molexp config set agent.model <id>`.",
        )

    # Workspace NotFound errors map to HTTP envelopes via the registered handlers.
    experiment = workspace.get_project(project_id).get_experiment(experiment_id)

    params: dict[str, JSONValue] = {"mode": "plan", "draft": draft}
    run_id = derive_run_id(params)
    try:
        run = experiment.get_run(run_id)
    except RunNotFoundError:
        run = experiment.add_run(params, id=run_id)

    gateway = build_plan_gateway(model=model, run=run)
    task = get_plan_runtime().create(
        workspace_root=str(workspace.root),
        task_id=f"plan-{generate_id()}",
        run=run,
        experiment=experiment,
        draft=draft,
        model=model,
        created_at=datetime.now(tz=UTC).isoformat(),
        gateway=gateway,
    )
    return _to_response(task, project_id=project_id, experiment_id=experiment_id)


@router.get("", response_model=PlanTaskListResponse)
def list_plan_tasks(
    project_id: str,
    experiment_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> PlanTaskListResponse:
    """List the live plan tasks in this workspace (in-memory; MVP)."""
    from molexp.server.deps.plan_runtime import get_plan_runtime

    tasks = get_plan_runtime().list_tasks(str(workspace.root))
    items = [_to_response(t, project_id=project_id, experiment_id=experiment_id) for t in tasks]
    return PlanTaskListResponse(tasks=items, total=len(items))


@router.get("/{task_id}", response_model=PlanTaskResponse)
def get_plan_task(
    project_id: str,
    experiment_id: str,
    task_id: str,
    workspace: Workspace = Depends(get_workspace),
) -> PlanTaskResponse:
    """Return one plan task's current status."""
    from molexp.server.deps.plan_runtime import get_plan_runtime

    task = get_plan_runtime().get(str(workspace.root), task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"plan task {task_id!r} not found")
    return _to_response(task, project_id=project_id, experiment_id=experiment_id)
