"""Experiment routes for MolExp API."""

from __future__ import annotations

from shutil import rmtree

from fastapi import APIRouter, Depends

from molexp.workspace.metrics import read_run_metrics

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError, ProjectNotFoundError
from ..schemas import (
    ComparisonRunRow,
    ExperimentComparisonResponse,
    ExperimentCreateRequest,
    ExperimentResponse,
    MessageResponse,
)

router = APIRouter(prefix="/projects/{project_id}/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    project_id: str,
    workspace=Depends(get_workspace),
) -> list[ExperimentResponse]:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    return [ExperimentResponse.from_model(e) for e in project.list_experiments()]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)
    return ExperimentResponse.from_model(experiment, runs=experiment.list_runs())


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    project_id: str,
    req: ExperimentCreateRequest,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    if req.default_target is not None and not _target_exists(workspace, req.default_target):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=f"compute target {req.default_target!r} is not registered on this workspace",
        )
    exp = project.experiment(
        name=req.name,
        workflow_source=req.workflow_source,
        params=req.parameter_space,
        default_target=req.default_target,
    )
    return ExperimentResponse.from_model(exp)


def _target_exists(workspace, name: str) -> bool:
    return any(t.name == name for t in workspace.metadata.targets)


@router.get("/{experiment_id}/comparison", response_model=ExperimentComparisonResponse)
def get_experiment_comparison(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> ExperimentComparisonResponse:
    """Comparison matrix: parameter columns x run rows + final metric values per run."""
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)

    runs = experiment.list_runs()
    rows: list[ComparisonRunRow] = []
    param_keys: set[str] = set()
    metric_keys: set[str] = set()

    for run in runs:
        param_keys.update(run.parameters.keys())
        metrics_summary: dict[str, object] = {}
        try:
            result = read_run_metrics(run.run_dir, limit=50000)
            for series in result.series:
                key_raw = series.get("key")
                latest = series.get("latestValue")
                if isinstance(key_raw, str) and key_raw and latest is not None:
                    metrics_summary[key_raw] = latest
                    metric_keys.add(key_raw)
        except (FileNotFoundError, OSError, ValueError):
            pass

        duration: float | None = None
        if run.metadata.finished_at and run.metadata.created_at:
            duration = (run.metadata.finished_at - run.metadata.created_at).total_seconds()

        error_dict: dict[str, str] | None = None
        if run.metadata.error:
            error_dict = {
                "type": run.metadata.error.type,
                "message": run.metadata.error.message,
            }

        rows.append(
            ComparisonRunRow(
                runId=run.id,
                status=run.status,
                parameters=dict(run.parameters),
                metrics=metrics_summary,
                durationSec=duration,
                created=run.metadata.created_at.isoformat(),
                finished=run.metadata.finished_at.isoformat() if run.metadata.finished_at else None,
                error=error_dict,
            )
        )

    return ExperimentComparisonResponse(
        experimentId=experiment_id,
        projectId=project_id,
        paramKeys=sorted(param_keys),
        metricKeys=sorted(metric_keys),
        runs=rows,
    )


@router.delete("/{experiment_id}", response_model=MessageResponse)
def delete_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)
    rmtree(experiment.experiment_dir)
    return MessageResponse(message="Experiment deleted")
