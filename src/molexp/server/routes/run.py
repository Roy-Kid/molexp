"""Run routes for MolExp API."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from molexp.plugins.metrics import read_run_metrics
from molexp.workspace import RunStatus

from ..dependencies import get_workspace
from ..exceptions import InvalidStatusError, RunNotFoundError
from ..schemas import (
    RunActionResponse,
    RunCreateRequest,
    RunExecutionResponse,
    RunFileNode,
    RunFilesResponse,
    RunLogsResponse,
    RunMetricsResponse,
    RunRerunResponse,
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


@router.get("/{run_id}/metrics", response_model=RunMetricsResponse)
def get_run_metrics(
    project_id: str,
    experiment_id: str,
    run_id: str,
    metric_type: str | None = Query(default=None, alias="type"),
    key: str | None = None,
    since_line: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
    workspace=Depends(get_workspace),
) -> RunMetricsResponse:
    """Return run-local metrics from ``metrics/metrics.jsonl``."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    result = read_run_metrics(
        run.run_dir,
        metric_type=metric_type,
        key=key,
        since_line=since_line,
        limit=limit,
    )
    return RunMetricsResponse(
        nextLine=result.next_line,
        records=result.records,
        series=result.series,
        parseErrors=result.parse_errors,
    )


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

    history = run.metadata.execution_history
    if not history:
        return RunExecutionResponse()

    latest_id = history[-1].execution_id
    wf_file: Path = run.run_dir / "execution" / latest_id / "workflow.json"
    if not wf_file.exists():
        return RunExecutionResponse(execution_id=latest_id)

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
        execution_id=data.get("execution_id", latest_id),
        status=data.get("status", "running"),
        steps=steps,
        end=data.get("end"),
    )


@router.get("/{run_id}/files", response_model=RunFilesResponse)
def get_run_files(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunFilesResponse:
    """Return the on-disk file tree for a run, enriched with catalog metadata.

    Files registered in the asset catalog (artifacts, logs, checkpoints,
    error traces) carry ``assetId``, ``assetKind``, and ``taskId`` so the
    UI can render lineage chips inline.
    """
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run_dir = run.run_dir
    from molexp.workspace.assets import AssetScope

    run_scope = AssetScope(kind="run", ids=(project_id, experiment_id, run_id))
    catalog_assets = workspace.catalog.query_assets(scope=run_scope)
    asset_index: dict[str, tuple[str, str, str | None]] = {}
    for a in catalog_assets:
        rel = str(a.path)
        asset_index[rel] = (
            a.asset_id,
            a.kind,  # type: ignore[attr-defined]
            a.producer.task_id if a.producer else None,
        )

    def build(node_path: Path) -> RunFileNode:
        rel = node_path.relative_to(run_dir).as_posix() if node_path != run_dir else ""
        is_file = node_path.is_file()
        info = asset_index.get(rel)
        node = RunFileNode(
            name=node_path.name or run_dir.name,
            relPath=rel,
            type="file" if is_file else "folder",
            size=node_path.stat().st_size if is_file else None,
            modified=node_path.stat().st_mtime,
            assetId=info[0] if info else None,
            assetKind=info[1] if info else None,
            taskId=info[2] if info else None,
        )
        if not is_file and node_path.exists():
            children: list[RunFileNode] = []
            for child in sorted(node_path.iterdir(), key=lambda p: (p.is_file(), p.name)):
                children.append(build(child))
            node.children = children
        return node

    nodes: list[RunFileNode] = []
    if run_dir.exists():
        for child in sorted(run_dir.iterdir(), key=lambda p: (p.is_file(), p.name)):
            nodes.append(build(child))

    return RunFilesResponse(
        runId=run_id,
        runDir=str(run_dir.relative_to(workspace.root)),
        nodes=nodes,
    )


@router.post("/{run_id}/rerun", response_model=RunRerunResponse, status_code=201)
def rerun_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunRerunResponse:
    """Clone an existing run's parameters into a fresh run within the same experiment."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    new_run = experiment.run(parameters=dict(run.parameters))
    return RunRerunResponse(
        sourceRunId=run.id,
        newRunId=new_run.id,
        projectId=project_id,
        experimentId=experiment_id,
        status=new_run.status,
    )


@router.post("/{run_id}/kill", response_model=RunActionResponse)
def kill_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunActionResponse:
    """Best-effort kill: mark the run as cancelled in workspace metadata.

    Note: this does not yet signal an external scheduler. It updates run
    status and clears ownership labels; live process termination is the
    scheduler's responsibility once such hooks are available.
    """
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run.cancel()
    return RunActionResponse(
        runId=run.id,
        status=run.status,
        message="Run marked as cancelled",
    )


@router.get("/{run_id}/export")
def export_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> StreamingResponse:
    """Stream a zip archive of the run directory (artifacts, logs, metadata)."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run_dir: Path = run.run_dir
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if run_dir.exists():
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(run_dir).as_posix())
    buffer.seek(0)

    filename = f"run-{run.id}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
