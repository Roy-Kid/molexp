"""``molexp migrate-layout`` — port pre-cutover workspaces to per-execution layout.

The redesign moves these run-level paths under their owning execution::

    run_dir/stdout.log         →  run_dir/executions/<exec_id>/stdout.log
    run_dir/stderr.log         →  run_dir/executions/<exec_id>/stderr.log
    run_dir/logs/*             →  run_dir/executions/<exec_id>/logs/*
    run_dir/jobs/*             →  run_dir/executions/<exec_id>/jobs/*
    run_dir/execution/<id>/*   →  run_dir/executions/<id>/*

It also writes ``projects.json`` / ``experiments.json`` / ``runs.json`` /
``executions.json`` index files at every container level.  The command is
idempotent: running it again on a migrated workspace is a no-op.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from molexp.workspace.base import _rebuild_container_index

from . import app
from ._common import rprint


def _latest_execution_id(run_dir: Path) -> str | None:
    """Return the newest exec_id from run.json's execution_history."""
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    try:
        data = json.loads(run_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    history = data.get("execution_history") or []
    if not history:
        return None
    last = history[-1]
    return last.get("execution_id")


def _move_into_exec(
    run_dir: Path,
    exec_id: str,
    *,
    files: tuple[str, ...],
    dirs: tuple[str, ...],
) -> list[str]:
    """Move legacy run-level *files*/*dirs* under ``executions/<exec_id>/``.

    Returns a list of one-line summaries describing what was moved (used
    for the per-run console line).  Pre-existing entries at the
    destination are left in place — caller has likely run migrate twice.
    """
    moves: list[str] = []
    exec_dir = run_dir / "executions" / exec_id
    exec_dir.mkdir(parents=True, exist_ok=True)

    for fname in files:
        src = run_dir / fname
        if not src.exists() or src.is_dir():
            continue
        dst = exec_dir / fname
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        moves.append(fname)

    for dname in dirs:
        src = run_dir / dname
        if not src.exists() or not src.is_dir():
            continue
        dst = exec_dir / dname
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        moves.append(dname + "/")

    return moves


def _backfill_execution_metadata(run_dir: Path) -> int:
    """Write ``execution.json`` for every attempt missing one.

    Synthesizes per-attempt metadata from the parent ``run.json``'s
    ``execution_history`` so legacy execution directories show up in the
    new ``executions.json`` index.  Returns the number of files written.
    """
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return 0
    try:
        data = json.loads(run_json.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    history = data.get("execution_history") or []
    run_id = data.get("id")
    written = 0
    for record in history:
        exec_id = record.get("execution_id")
        if not exec_id:
            continue
        exec_dir = run_dir / "executions" / exec_id
        target = exec_dir / "execution.json"
        if target.exists() or not exec_dir.exists():
            continue
        synthesized = {
            "execution_id": exec_id,
            "run_id": run_id,
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "status": record.get("status", "unknown"),
            "scheduler_job_id": record.get("scheduler_job_id"),
            "executor_info": {},
            "error": None,
        }
        target.write_text(json.dumps(synthesized, indent=2, default=str))
        written += 1
    return written


def _rename_singular_execution_dir(run_dir: Path) -> bool:
    """Rename pre-cutover ``run_dir/execution/`` to ``run_dir/executions/``.

    Returns ``True`` if the rename happened.  When both exist, merges
    children of the old singular directory into the plural one and then
    removes the old directory (idempotent on partial migrations).
    """
    old = run_dir / "execution"
    new = run_dir / "executions"
    if not old.exists():
        return False
    if not new.exists():
        old.rename(new)
        return True
    # Merge: move every child from old/ into new/ when not present.
    for child in old.iterdir():
        target = new / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))
    # Best-effort cleanup; leave it if non-empty due to collisions.
    try:
        old.rmdir()
    except OSError:
        pass
    return True


def _refresh_indices(workspace_root: Path) -> int:
    """Rebuild every container index under *workspace_root*.

    Returns the number of index files written.
    """
    written = 0
    projects_dir = workspace_root / "projects"
    if projects_dir.exists():
        _rebuild_container_index(
            container_dir=projects_dir,
            index_filename="projects.json",
            metadata_filename="project.json",
            fields=["id", "name", "description", "created_at"],
        )
        written += 1
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            experiments_dir = proj_dir / "experiments"
            if experiments_dir.exists():
                _rebuild_container_index(
                    container_dir=experiments_dir,
                    index_filename="experiments.json",
                    metadata_filename="experiment.json",
                    fields=[
                        "id",
                        "name",
                        "description",
                        "tags",
                        "n_replicas",
                        "created_at",
                    ],
                )
                written += 1
                for exp_dir in experiments_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    runs_dir = exp_dir / "runs"
                    if runs_dir.exists():
                        _rebuild_container_index(
                            container_dir=runs_dir,
                            index_filename="runs.json",
                            metadata_filename="run.json",
                            fields=[
                                "id",
                                "status",
                                "parameters",
                                "profile",
                                "created_at",
                                "finished_at",
                            ],
                        )
                        written += 1
                        for run_dir in runs_dir.iterdir():
                            if not run_dir.is_dir():
                                continue
                            executions_dir = run_dir / "executions"
                            if executions_dir.exists():
                                _rebuild_container_index(
                                    container_dir=executions_dir,
                                    index_filename="executions.json",
                                    metadata_filename="execution.json",
                                    fields=[
                                        "execution_id",
                                        "run_id",
                                        "status",
                                        "started_at",
                                        "finished_at",
                                        "scheduler_job_id",
                                    ],
                                )
                                written += 1
    return written


@app.command(name="migrate-layout")
def migrate_layout(
    path: Annotated[
        Path,
        typer.Argument(
            help="Workspace root to migrate.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Plan the migration without modifying anything.",
        ),
    ] = False,
) -> None:
    """Migrate a pre-cutover workspace to the per-execution layout.

    Idempotent: running twice on a workspace does nothing the second time.
    """
    workspace_json = path / "workspace.json"
    if not workspace_json.exists():
        rprint(f"[red]Error:[/red] {path} does not look like a workspace (no workspace.json)")
        raise typer.Exit(1)

    started = datetime.now()
    runs_seen = 0
    runs_modified = 0

    projects_dir = path / "projects"
    if projects_dir.exists():
        for proj_dir in sorted(projects_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            experiments_dir = proj_dir / "experiments"
            if not experiments_dir.exists():
                continue
            for exp_dir in sorted(experiments_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                runs_dir = exp_dir / "runs"
                if not runs_dir.exists():
                    continue
                for run_dir in sorted(runs_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    runs_seen += 1
                    moved: list[str] = []

                    if dry_run:
                        # Planning mode: report what *would* move.
                        if (run_dir / "execution").exists():
                            moved.append("execution/ → executions/")
                        latest = _latest_execution_id(run_dir)
                        for f in ("stdout.log", "stderr.log"):
                            if (run_dir / f).exists() and not (run_dir / f).is_dir():
                                if latest:
                                    moved.append(f)
                        for d in ("logs", "jobs"):
                            if (run_dir / d).exists() and (run_dir / d).is_dir() and latest:
                                moved.append(d + "/")
                    else:
                        if _rename_singular_execution_dir(run_dir):
                            moved.append("execution/→executions/")
                        latest = _latest_execution_id(run_dir)
                        if latest:
                            moved.extend(
                                _move_into_exec(
                                    run_dir,
                                    latest,
                                    files=("stdout.log", "stderr.log"),
                                    dirs=("logs", "jobs"),
                                )
                            )
                        backfilled = _backfill_execution_metadata(run_dir)
                        if backfilled:
                            moved.append(f"+{backfilled} execution.json")
                        elif any(
                            (run_dir / p).exists()
                            for p in ("stdout.log", "stderr.log", "logs", "jobs")
                        ):
                            rprint(
                                f"  [yellow]warn[/yellow] {run_dir.relative_to(path)}: "
                                "legacy run-level logs/jobs found but execution_history "
                                "is empty — skipped (cannot infer owning execution)."
                            )

                    if moved:
                        runs_modified += 1
                        rprint(
                            f"  [green]OK[/green] {run_dir.relative_to(path)}: {', '.join(moved)}"
                        )

    if dry_run:
        rprint(
            f"\n[cyan]Dry run:[/cyan] {runs_seen} run(s) scanned, "
            f"{runs_modified} would be modified."
        )
        return

    indices_written = _refresh_indices(path)

    rprint(
        f"\n[green]Done.[/green] {runs_seen} run(s) scanned, "
        f"{runs_modified} migrated, {indices_written} index file(s) refreshed "
        f"in {(datetime.now() - started).total_seconds():.2f}s."
    )
