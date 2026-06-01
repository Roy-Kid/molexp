"""``molexp runs prune`` — interactive hierarchical cleanup.

Walks project → experiment → run → execution step-by-step, letting the user
manually pick what to delete at each layer.  Removes the chosen
``execution/{exec_id}/`` directories and rewrites
``run.metadata.execution_history``.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from molexp.workspace.models import ExecutionRecord
from molexp.workspace.run import Run

from ._common import console, rprint, status_color


def _select_one(
    title: str,
    rows: Sequence[tuple[str, ...]],
    headers: tuple[str, ...],
) -> int | None:
    """Render a numbered table and prompt for a single choice.

    Returns the zero-based index of the chosen row, or ``None`` if the
    user aborted with empty input.
    """
    if not rows:
        rprint(f"[yellow]{title}: nothing to select.[/yellow]")
        return None

    table = Table(title=title)
    table.add_column("#", style="cyan", justify="right")
    for h in headers:
        table.add_column(h)
    for idx, row in enumerate(rows, start=1):
        table.add_row(str(idx), *row)
    console.print(table)

    raw = typer.prompt("Enter # (empty to abort)", default="", show_default=False)
    raw = raw.strip()
    if not raw:
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= len(rows)):
        rprint(f"[red]Invalid choice:[/red] {raw!r}")
        return None
    return int(raw) - 1


def _select_many(
    title: str,
    rows: Sequence[tuple[str, ...]],
    headers: tuple[str, ...],
    *,
    status_values: Sequence[str] | None = None,
) -> list[int]:
    """Multi-select prompt.

    Accepts comma-separated numbers, ranges ``a-b``, ``all``, or
    status keywords (``failed``/``cancelled``/...) matched against
    ``status_values`` — plain, un-coloured status strings the caller
    supplies alongside the rendered rows.  Returns zero-based indices.
    """
    if not rows:
        rprint(f"[yellow]{title}: nothing to select.[/yellow]")
        return []

    table = Table(title=title)
    table.add_column("#", style="cyan", justify="right")
    for h in headers:
        table.add_column(h)
    for idx, row in enumerate(rows, start=1):
        table.add_row(str(idx), *row)
    console.print(table)

    rprint("[dim]Formats: 1,3,5  |  2-4  |  all  |  failed,cancelled[/dim]")
    raw = typer.prompt(
        "Select records to delete (empty to abort)",
        default="",
        show_default=False,
    )
    raw = raw.strip().lower()
    if not raw:
        return []

    if raw == "all":
        return list(range(len(rows)))

    keyword_map = {"failed", "cancelled", "running", "succeeded", "pending"}
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if status_values and tokens and all(t in keyword_map for t in tokens):
        return [i for i, sv in enumerate(status_values) if sv.lower() in tokens]

    chosen: set[int] = set()
    for tok in tokens:
        if "-" in tok:
            a, _, b = tok.partition("-")
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                chosen.update(range(lo - 1, hi))
                continue
        if tok.isdigit() and 1 <= int(tok) <= len(rows):
            chosen.add(int(tok) - 1)
            continue
        rprint(f"[red]Ignoring invalid token:[/red] {tok!r}")

    return sorted(i for i in chosen if 0 <= i < len(rows))


def _execution_rows(run: Run) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for rec in run.metadata.execution_history:
        finished = rec.finished_at.strftime("%Y-%m-%d %H:%M") if rec.finished_at else "—"
        started = rec.started_at.strftime("%Y-%m-%d %H:%M")
        status = rec.status or "running"
        rows.append(
            (
                rec.execution_id,
                status,
                started,
                finished,
            )
        )
    return rows


def prune_runs(
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Workspace path (default: cwd)."),
    ] = None,
) -> None:
    """Interactively delete per-execution records of a run.

    Descends project → experiment → run → executions, letting the user
    pick at each step.  Only the ``execution/{exec_id}/`` directories and
    matching ``execution_history`` entries are touched; run status /
    parameters / artifacts are left alone.
    """
    from . import _common

    ws = _common.get_workspace(path)

    # Layer 1: project
    projects = ws.list_projects()
    if not projects:
        rprint("[yellow]Workspace has no projects.[/yellow]")
        raise typer.Exit(0)
    proj_rows = [(p.id, p.metadata.name, f"{len(p.list_experiments())} exp") for p in projects]
    i = _select_one("Projects", proj_rows, ("ID", "Name", "Exp count"))
    if i is None:
        rprint("[dim]Aborted.[/dim]")
        raise typer.Exit(0)
    project = projects[i]

    # Layer 2: experiment
    experiments = project.list_experiments()
    if not experiments:
        rprint(f"[yellow]Project {project.id!r} has no experiments.[/yellow]")
        raise typer.Exit(0)
    exp_rows = [(e.id, e.metadata.name, f"{len(e.list_runs())} runs") for e in experiments]
    i = _select_one(
        f"Experiments in {project.id!r}",
        exp_rows,
        ("ID", "Name", "Run count"),
    )
    if i is None:
        rprint("[dim]Aborted.[/dim]")
        raise typer.Exit(0)
    experiment = experiments[i]

    # Layer 3: run
    runs = experiment.list_runs()
    if not runs:
        rprint(f"[yellow]Experiment {experiment.id!r} has no runs.[/yellow]")
        raise typer.Exit(0)

    run_rows: list[tuple[str, ...]] = []
    for r in runs:
        status = str(r.status).lower()
        n_exec = len(r.metadata.execution_history)
        run_rows.append(
            (
                r.id,
                status,
                str(n_exec),
                r.metadata.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        )
    i = _select_one(
        f"Runs in {project.id}/{experiment.id}",
        run_rows,
        ("Run ID", "Status", "# exec", "Created"),
    )
    if i is None:
        rprint("[dim]Aborted.[/dim]")
        raise typer.Exit(0)
    run = runs[i]

    # Layer 4: execution records
    rows = _execution_rows(run)
    if not rows:
        rprint(f"[yellow]Run {run.id!r} has no execution history.[/yellow]")
        raise typer.Exit(0)

    colored = [
        (
            rec_id,
            f"[{status_color(status)}]{status}[/{status_color(status)}]",
            started,
            finished,
        )
        for rec_id, status, started, finished in rows
    ]
    indices = _select_many(
        f"Executions of {run.id}",
        colored,
        ("Execution ID", "Status", "Started", "Finished"),
        status_values=[status for _, status, _, _ in rows],
    )
    if not indices:
        rprint("[dim]Nothing selected — aborted.[/dim]")
        raise typer.Exit(0)

    targets: list[ExecutionRecord] = [run.metadata.execution_history[i] for i in indices]

    # Refuse to delete a still-running entry unless the run itself is
    # already terminal (zombie cleanup).
    live = [t for t in targets if t.status == "running" and str(run.status).lower() == "running"]
    if live:
        rprint(
            f"[red]Refusing:[/red] {len(live)} selected record(s) look live "
            "(status=running on an active run).  Cancel the run first, or "
            "wait for it to finish."
        )
        raise typer.Exit(1)

    confirm = typer.prompt(
        f"Delete {len(targets)} execution record(s) from run {run.id!r}? [y/N]",
        default="N",
        show_default=False,
    )
    if confirm.strip().lower() not in ("y", "yes"):
        rprint("[dim]Aborted.[/dim]")
        raise typer.Exit(0)

    removed_ids = {t.execution_id for t in targets}
    exec_root = Path(run.run_dir / "executions")
    deleted_dirs = 0
    for exec_id in removed_ids:
        exec_dir = exec_root / exec_id
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
            deleted_dirs += 1
            rprint(f"  [green]OK[/green] removed {exec_dir.relative_to(Path(run.run_dir))}")
        else:
            rprint(f"  [dim]skip[/dim]  {exec_id} (no directory)")

    new_history = [
        rec for rec in run.metadata.execution_history if rec.execution_id not in removed_ids
    ]
    run._update_metadata(execution_history=new_history)
    rprint(
        f"[green]Done.[/green] Removed {deleted_dirs} dir(s), "
        f"pruned {len(targets)} history entry/entries.  "
        f"{len(new_history)} record(s) remain."
    )


def register(run_app: typer.Typer) -> None:
    run_app.command("prune")(prune_runs)
