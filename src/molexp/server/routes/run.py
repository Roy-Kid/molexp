"""Run routes for MolExp API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends

from molexp.workspace import RunStatus

from ..dependencies import get_workspace
from ..exceptions import InvalidStatusError, RunNotFoundError
from ..schemas import (
    RunCreateRequest,
    RunExecutionResponse,
    RunLogsResponse,
    RunResponse,
    RunStatusResponse,
    WorkflowStepInfo,
)

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/runs",
    tags=["runs"],
)


def _get_experiment(workspace, project_id: str, experiment_id: str):
    project = workspace.get_project(project_id)
    if not project:
        return None
    return project.get_experiment(experiment_id)


@router.get("", response_model=list[RunResponse])
def list_runs(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> list[RunResponse]:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, "")
    return [RunResponse.from_model(r) for r in experiment.list_runs()]


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    return RunResponse.from_model(run)


@router.post("", response_model=RunResponse, status_code=201)
def create_run(
    project_id: str,
    experiment_id: str,
    run_req: RunCreateRequest,
    workspace=Depends(get_workspace),
) -> RunResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, "")
    run = experiment.run(parameters=run_req.parameters)
    return RunResponse.from_model(run)


@router.get("/{run_id}/logs", response_model=RunLogsResponse)
def get_run_logs(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunLogsResponse:
    """Return stdout (job.out) and stderr (job.err) for a run."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run_dir: Path = run.run_dir
    stdout: str | None = None
    stderr: str | None = None

    job_out = run_dir / "job.out"
    if job_out.exists():
        stdout = job_out.read_text(errors="replace")

    job_err = run_dir / "job.err"
    if job_err.exists():
        stderr = job_err.read_text(errors="replace")

    return RunLogsResponse(stdout=stdout, stderr=stderr)


@router.get("/{run_id}/execution", response_model=RunExecutionResponse)
def get_run_execution(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunExecutionResponse:
    """Return workflow execution state from workflow.json."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    exec_root: Path = run.run_dir / "execution"
    if not exec_root.exists():
        return RunExecutionResponse()

    exec_dirs = sorted(p for p in exec_root.iterdir() if p.is_dir())
    if not exec_dirs:
        return RunExecutionResponse()

    # Latest execution is the last one alphabetically (exec-id, exec-id-2, exec-id-3…)
    latest = exec_dirs[-1]
    wf_file = latest / "workflow.json"
    if not wf_file.exists():
        return RunExecutionResponse(execution_id=latest.name)

    data = json.loads(wf_file.read_text())
    steps = [
        WorkflowStepInfo(
            index=s["index"],
            status=s.get("status", "pending"),
            step_outputs=s.get("step_outputs", {}),
        )
        for s in data.get("steps", [])
    ]
    return RunExecutionResponse(
        execution_id=data.get("execution_id", latest.name),
        status=data.get("status", "running"),
        steps=steps,
        end=data.get("end"),
    )


@router.patch("/{run_id}/status", response_model=RunStatusResponse)
def update_run_status(
    project_id: str,
    experiment_id: str,
    run_id: str,
    status: dict[str, str],
    workspace=Depends(get_workspace),
) -> RunStatusResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    new_status_str = status.get("status", run.status)
    try:
        new_status = RunStatus(new_status_str)
    except ValueError:
        raise InvalidStatusError(run.status, new_status_str)

    updates: dict = {"status": new_status.value}
    if new_status_str in ("succeeded", "failed", "cancelled"):
        updates["finished_at"] = datetime.now()

    run._update_metadata(**updates)

    return RunStatusResponse(
        id=run.id,
        status=run.status,
        finished=run.metadata.finished_at.isoformat() if run.metadata.finished_at else None,
    )
