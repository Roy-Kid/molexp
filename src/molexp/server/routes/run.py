"""Run routes for MolExp API."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from molexp._run_cancel import try_cancel
from molexp.plugins.submit_molq.submit import SubmitHandler
from molexp.workflow import (
    Workflow,
    WorkflowSnapshotRef,
    resolve_spec_entrypoint,
)
from molexp.workspace import (
    Experiment,
    RunStatus,
)
from molexp.workspace import (
    ExperimentNotFoundError as WorkspaceExperimentNotFoundError,
)
from molexp.workspace import (
    ProjectNotFoundError as WorkspaceProjectNotFoundError,
)
from molexp.workspace import (
    RunNotFoundError as WorkspaceRunNotFoundError,
)
from molexp.workspace.metrics import read_run_metrics
from molexp.workspace.targets import get_target

from ..dependencies import get_workspace
from ..exceptions import InvalidStatusError, RunNotFoundError
from ..schemas import (
    LammpsLogResponse,
    LammpsThermoStage,
    MetricSeriesResponse,
    RunActionResponse,
    RunCreateRequest,
    RunExecutionResponse,
    RunFileNode,
    RunFilesResponse,
    RunFileTextResponse,
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


def _get_experiment(workspace, project_id: str, experiment_id: str):  # noqa: ANN001, ANN202
    """Strict-getter chain — returns ``Experiment`` or ``None``.

    Translates the workspace-layer ``*NotFoundError`` exceptions to a
    boolean miss so the existing ``if not experiment:`` callers continue
    to map missing entities onto their HTTP 404 responses.
    """
    try:
        project = workspace.get_project(project_id)
    except WorkspaceProjectNotFoundError:
        return None
    try:
        return project.get_experiment(experiment_id)
    except WorkspaceExperimentNotFoundError:
        return None


def _get_run_or_none(experiment, run_id: str):  # noqa: ANN001, ANN202
    """Wrap ``experiment.get_run(run_id)`` to map ``RunNotFoundError`` → ``None``."""
    try:
        return experiment.get_run(run_id)
    except WorkspaceRunNotFoundError:
        return None


def _synthesize_snapshot(experiment: Experiment) -> dict | None:
    """Build the opaque snapshot dict the run record should carry.

    The route does not require an explicit snapshot from the caller —
    if the experiment already has a workflow bound in the workflow-
    layer registry, we resolve its entrypoint here so molq workers
    can re-import it without re-running the user script. When no
    binding exists, returns ``None`` (the run still materializes;
    submit_handler dispatch will refuse it later if a target is
    requested).
    """
    spec = Workflow.for_experiment(experiment)
    if spec is None:
        return None
    try:
        entrypoint = resolve_spec_entrypoint(spec)
    except ValueError:
        # Spec isn't bound to a module-level name — most often a
        # promote_callable result on a fixture. Fall back to source-
        # only snapshot.
        entrypoint = None
    snap = WorkflowSnapshotRef(
        source=experiment.metadata.workflow_source or "",
        entrypoint=entrypoint,
        git_commit=experiment.metadata.git_commit,
    )
    return snap.model_dump(mode="json")


def _dispatch_to_molq(target, run) -> None:  # noqa: ANN001
    """Submit *run* through molq onto *target*.

    Resources and scheduling come from the target's defaults — the API
    has no per-run CLI overrides like ``molexp run --cpus``.
    """
    snapshot = run.metadata.workflow_snapshot
    entrypoint = snapshot.get("entrypoint") if isinstance(snapshot, dict) else None
    if not entrypoint:
        raise HTTPException(
            status_code=422,
            detail=(
                f"experiment {run.experiment.id!r} has no workflow entrypoint; "
                "bind a Python Workflow or callable on the experiment "
                "before submitting via the API"
            ),
        )

    # Worker chdirs to submit_cwd before importing user code so cwd-relative
    # paths resolve the same as at submit time.
    if not run.metadata.submit_cwd:
        run._update_metadata(submit_cwd=str(Path.cwd().resolve()))

    handler = SubmitHandler(
        scheduler=target.scheduler,
        cluster=None,
        resources=target.default_resources,
        scheduling=target.default_scheduling,
        target=target,
    )
    handler(None, run, run.experiment, run.experiment.project)


@router.get("", response_model=list[RunResponse])
def list_runs(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    return RunResponse.from_model(run)


@router.post("", response_model=RunResponse, status_code=201)
def create_run(
    project_id: str,
    experiment_id: str,
    run_req: RunCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, "")

    target = None
    if run_req.target is not None:
        try:
            target = get_target(workspace, run_req.target)
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"compute target {run_req.target!r} is not registered on this workspace",
            ) from exc

    run = experiment.add_run(
        parameters=run_req.parameters,
        target=run_req.target,
        workflow_snapshot=_synthesize_snapshot(experiment),
    )
    if target is not None:
        _dispatch_to_molq(target, run)
    return RunResponse.from_model(run)


def _read_execution_logs(run, execution_id: str) -> RunLogsResponse:  # noqa: ANN001
    exec_dir: Path = run.run_dir / "executions" / execution_id
    stdout: str | None = None
    stderr: str | None = None
    out_file = exec_dir / "stdout.log"
    err_file = exec_dir / "stderr.log"
    if out_file.exists():
        stdout = out_file.read_text(errors="replace")
    if err_file.exists():
        stderr = err_file.read_text(errors="replace")
    return RunLogsResponse(execution_id=execution_id, stdout=stdout, stderr=stderr)


@router.get("/{run_id}/logs", response_model=RunLogsResponse)
def get_run_logs(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunLogsResponse:
    """Return stdout/stderr for the most recent execution of a run."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    history = run.metadata.execution_history
    if not history:
        return RunLogsResponse()
    return _read_execution_logs(run, history[-1].execution_id)


@router.get(
    "/{run_id}/executions/{execution_id}/logs",
    response_model=RunLogsResponse,
)
def get_run_execution_logs(
    project_id: str,
    experiment_id: str,
    run_id: str,
    execution_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunLogsResponse:
    """Return stdout/stderr for a specific execution attempt."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    return _read_execution_logs(run, execution_id)


@router.get("/{run_id}/metrics", response_model=RunMetricsResponse)
def get_run_metrics(
    project_id: str,
    experiment_id: str,
    run_id: str,
    metric_type: str | None = Query(default=None, alias="type"),
    key: str | None = None,
    since_line: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunMetricsResponse:
    """Return run-local metrics from ``metrics/metrics.jsonl``."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
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
        # ``entry`` is ``dict[str, JSONValue]``; ``model_validate`` runs
        # pydantic's per-field coercion / validation rather than the
        # static-typed positional constructor.
        series=[MetricSeriesResponse.model_validate(entry) for entry in result.series],
        parseErrors=result.parse_errors,
    )


@router.get("/{run_id}/file/text", response_model=RunFileTextResponse)
def get_run_file_text(
    project_id: str,
    experiment_id: str,
    run_id: str,
    path: str = Query(..., description="Relative path under run_dir"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunFileTextResponse:
    """Return the raw text content of a file under the run directory."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run_dir: Path = run.run_dir
    target = (run_dir / path).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes run directory") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="file is not text-decodable as UTF-8") from exc
    return RunFileTextResponse(path=path, content=content, size=target.stat().st_size)


@router.get("/{run_id}/lammps-log", response_model=LammpsLogResponse)
def get_run_lammps_log(
    project_id: str,
    experiment_id: str,
    run_id: str,
    path: str = Query(..., description="Relative path of the log file under run_dir"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> LammpsLogResponse:
    """Parse a LAMMPS log file and return thermo stages.

    Inlined parser — ``molpy.io`` does not export a multi-stage log
    reader, so the route owns this lightweight regex-based parse to
    avoid coupling the API surface to a transient molpy refactor.
    """
    import re

    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    run_dir: Path = run.run_dir
    target = (run_dir / path).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path escapes run directory") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"log file not found: {path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    version = text.split("\n", 1)[0].strip() if text else None

    stages: list[LammpsThermoStage] = []
    for block in re.findall(
        r"Per MPI rank memory allocation .*?\n(.*?)Loop time of",
        text,
        flags=re.DOTALL,
    ):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        columns = lines[0].split()
        rows: list[list[float]] = []
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) != len(columns):
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
        stages.append(LammpsThermoStage(columns=columns, rows=rows))

    return LammpsLogResponse(
        path=path,
        version=version,
        nStages=len(stages),
        stages=stages,
    )


@router.get("/{run_id}/execution", response_model=RunExecutionResponse)
def get_run_execution(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunExecutionResponse:
    """Return workflow execution state from workflow.json."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    history = run.metadata.execution_history
    if not history:
        return RunExecutionResponse()

    latest_id = history[-1].execution_id
    wf_file: Path = run.run_dir / "executions" / latest_id / "workflow.json"
    if not wf_file.exists():
        return RunExecutionResponse(execution_id=latest_id)

    data = json.loads(wf_file.read_text())
    steps = [
        WorkflowStepInfo(
            index=s["index"],
            status=s.get("status", "pending"),
            outputs=s.get("outputs", {}),
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunFilesResponse:
    """Return the on-disk file tree for a run, enriched with catalog metadata.

    Files registered in the asset catalog (artifacts, logs, checkpoints,
    error traces) carry ``assetId``, ``assetKind``, and ``taskId`` so the
    UI can render lineage chips inline.
    """
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunRerunResponse:
    """Clone an existing run's parameters into a fresh run within the same experiment.

    The new run inherits the source run's compute target.  When a target is
    set, the new run is also submitted through molq so a re-run from the UI
    actually re-executes (no manual ``molexp run`` step required).
    """
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    inherited_target = run.metadata.target
    target = None
    if inherited_target is not None:
        try:
            target = get_target(workspace, inherited_target)
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"compute target {inherited_target!r} is not registered on this workspace",
            ) from exc

    new_run = experiment.add_run(
        parameters=dict(run.parameters),
        target=inherited_target,
        workflow_snapshot=_synthesize_snapshot(experiment),
    )
    if target is not None:
        _dispatch_to_molq(target, new_run)
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunActionResponse:
    """Cancel a run.

    Routes through :func:`molexp._run_cancel.try_cancel`, which signals
    molq via :class:`molq.Submitor` for cluster-submitted runs and
    sends ``SIGTERM`` for runs still owned by a local pid.  When neither
    path applies (run never submitted, terminal, or executor info
    missing) we fall back to flipping the metadata status so the UI
    still reflects user intent.
    """
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    warning = try_cancel(run)
    if warning is None:
        return RunActionResponse(
            runId=run.id,
            status=run.status,
            message="Run cancelled",
        )
    run.cancel()
    return RunActionResponse(
        runId=run.id,
        status=run.status,
        message=warning,
    )


@router.get("/{run_id}/export")
def export_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> StreamingResponse:
    """Stream a zip archive of the run directory (artifacts, logs, metadata)."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunStatusResponse:
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
    run = _get_run_or_none(experiment, run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    new_status_str = status.get("status", run.status)
    try:
        new_status = RunStatus(new_status_str)
    except ValueError:
        raise InvalidStatusError(run.status, new_status_str)  # noqa: B904

    updates: dict = {"status": new_status.value}
    if new_status_str in ("succeeded", "failed", "cancelled"):
        updates["finished_at"] = datetime.now()

    run._update_metadata(**updates)

    return RunStatusResponse(
        id=run.id,
        status=run.status,
        finished=run.metadata.finished_at.isoformat() if run.metadata.finished_at else None,
    )
