"""``molexp info`` — workspace statistics.

(``init`` lives in :mod:`molexp.cli.init_cmd` as the single top-level command.)
"""

from __future__ import annotations

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint, status_color
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import LocalTarget


@app.command()
def info(target_spec: TargetOption = ".") -> None:
    """Show workspace statistics."""
    from molexp.workspace import Workspace

    target, _transport, fs = resolve_workspace_target(target_spec)

    try:
        ws = Workspace(str(target), fs=fs)
    except Exception as exc:
        rprint(f"[red]Failed to open workspace:[/red] {exc}")
        raise typer.Exit(1) from exc

    rprint(f"[bold]Workspace:[/bold] {target}")
    rprint(f"  Name: {ws.metadata.name}")
    rprint(f"  ID: {ws.metadata.id}")

    if isinstance(target, LocalTarget):
        projects = ws.list_projects()
        total_experiments = 0
        total_runs = 0
        run_status_counts: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        profile_counts: dict[str, int] = {}

        for project in projects:
            experiments = project.list_experiments()
            total_experiments += len(experiments)
            for experiment in experiments:
                runs = experiment.list_runs()
                total_runs += len(runs)
                for r in runs:
                    status = str(r.status).lower()
                    if status in run_status_counts:
                        run_status_counts[status] += 1
                    pname = r.metadata.profile
                    if pname:
                        profile_counts[pname] = profile_counts.get(pname, 0) + 1

        rprint(f"  Root: {ws.root}")
        rprint("\n[bold]Statistics:[/bold]")
        rprint(f"  Projects: {len(projects)}")
        rprint(f"  Experiments: {total_experiments}")
        rprint(f"  Runs: {total_runs}")

        if total_runs > 0:
            rprint("\n[bold]Run Status:[/bold]")
            for st, count in run_status_counts.items():
                if count > 0:
                    rprint(f"  [{status_color(st)}]{st.capitalize()}[/{status_color(st)}]: {count}")

        if profile_counts:
            rprint("\n[bold]Profiles:[/bold]")
            for pname, count in sorted(profile_counts.items()):
                rprint(f"  [cyan]{pname}[/cyan]: {count}")
    else:
        rprint(f"  Path: {target.path}")
