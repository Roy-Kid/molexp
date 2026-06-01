"""TensorBoard scalar route — optional, behind ``molexp[tensorboard]``.

The parser is lazily imported from :mod:`molexp.plugins.tensorboard.parser`
so the route registration itself works on installs without the
optional dep; only an actual request to ``GET .../tensorboard/scalars``
will surface a 503 with a friendly install hint when the dep is
missing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from molexp.workspace import (
    ExperimentNotFoundError as WorkspaceExperimentNotFoundError,
)
from molexp.workspace import (
    ProjectNotFoundError as WorkspaceProjectNotFoundError,
)
from molexp.workspace import (
    RunNotFoundError as WorkspaceRunNotFoundError,
)

from ..dependencies import get_workspace
from ..exceptions import RunNotFoundError
from ..schemas import (
    TensorboardScalarPoint,
    TensorboardScalarSeries,
    TensorboardScalarsResponse,
)

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/tensorboard",
    tags=["tensorboard"],
)


def _resolve_run(workspace, project_id: str, experiment_id: str, run_id: str):  # noqa: ANN001, ANN202
    """Strict-getter chain — returns the workspace ``Run`` or raises 404."""
    try:
        project = workspace.get_project(project_id)
    except WorkspaceProjectNotFoundError as exc:
        raise RunNotFoundError(project_id, experiment_id, run_id) from exc
    try:
        experiment = project.get_experiment(experiment_id)
    except WorkspaceExperimentNotFoundError as exc:
        raise RunNotFoundError(project_id, experiment_id, run_id) from exc
    try:
        return experiment.get_run(run_id)
    except WorkspaceRunNotFoundError as exc:
        raise RunNotFoundError(project_id, experiment_id, run_id) from exc


@router.get("/scalars", response_model=TensorboardScalarsResponse)
def get_run_tensorboard_scalars(
    project_id: str,
    experiment_id: str,
    run_id: str,
    tag: list[str] | None = Query(default=None, description="Repeatable scalar-tag filter"),
    logdir: str | None = Query(
        default=None,
        description="Relative path under run_dir; default = discover every tfevents dir",
    ),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> TensorboardScalarsResponse:
    """Parse all (or filtered) scalar tags from a run's tfevents files.

    Returns 503 with an install hint when the optional ``tensorboard``
    extra is missing, 404 when the run is unknown, and 200 with an
    empty ``series`` list when the run has no tfevents on disk.
    """
    # Local import keeps the route module import-safe when tensorboard
    # isn't installed; only an actual request pays the parser import.
    from molexp.plugins.tensorboard import (
        discover_logdirs,
        read_scalars,
        require_tensorboard,
    )

    run = _resolve_run(workspace, project_id, experiment_id, run_id)
    run_dir = Path(run.run_dir)

    try:
        require_tensorboard()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Empty-string `?logdir=` is treated as "discover" so query-builders
    # that always serialise the key don't silently bypass the recursive
    # walk and read only run_dir's top level.
    if logdir:
        target = (run_dir / logdir).resolve()
        try:
            target.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="logdir escapes run directory") from exc
        if not target.is_dir():
            raise HTTPException(status_code=404, detail=f"logdir not found: {logdir}")
        logdirs = [target]
    else:
        logdirs = discover_logdirs(run_dir)

    series_out: list[TensorboardScalarSeries] = []
    rel_logdirs: list[str] = []
    # Drop empty-string entries that arise from `?tag=` with no value;
    # otherwise the filter narrows the result set to tags equal to ""
    # (none) and the response silently comes back empty.
    cleaned_tags = tuple(t for t in (tag or ()) if t)
    tag_filter = cleaned_tags or None
    for ldir in logdirs:
        rel = ldir.relative_to(run_dir).as_posix()
        rel_logdirs.append(rel)
        parsed = read_scalars(ldir, tags=tag_filter, relative_to=run_dir)
        for series in parsed:
            series_out.append(
                TensorboardScalarSeries(
                    tag=series.tag,
                    logdir=series.logdir,
                    points=[
                        TensorboardScalarPoint(step=p.step, wallTime=p.wall_time, value=p.value)
                        for p in series.points
                    ],
                )
            )

    return TensorboardScalarsResponse(
        runId=run.id,
        runDir=str(run_dir.relative_to(Path(workspace.root))),
        logdirs=rel_logdirs,
        series=series_out,
    )
