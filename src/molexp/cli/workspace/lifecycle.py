"""``molexp info`` — workspace statistics.

(``init`` lives in :mod:`molexp.cli.init_cmd` as the single top-level command.)
"""

from __future__ import annotations

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint, status_color
from molexp.cli._target import TargetOption, open_workspace


@app.command()
def info(target_spec: TargetOption = ".") -> None:
    """Show workspace statistics."""
    try:
        target, _transport, _fs, ws = open_workspace(target_spec)
    except FileNotFoundError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        rprint(f"[red]Failed to open workspace:[/red] {exc}")
        raise typer.Exit(1) from exc

    rprint(f"[bold]Workspace:[/bold] {target}")
    rprint(f"  Name: {ws.metadata.name}")
    rprint(f"  ID: {ws.metadata.id}")
    rprint(f"  Root: {ws.root}")

    # Project/run stats work for any FileSystem (local or remote) once path
    # join preserves absolute remote roots.
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


@app.command()
def validate(
    target_spec: TargetOption = ".",
    strict: bool = False,
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Print the full ValidationReport as JSON (MCP/agent wire shape).",
    ),
) -> None:
    """Check the workspace against the layout + OKF laws.

    Read-only. Exits non-zero when the tree violates the layout law, so it
    drops straight into a pre-commit hook or CI step. ``--strict`` also fails
    on warnings (lazily-created state that is absent but legal).

    ``--json`` emits the same report dict MCP ``validate_workspace`` returns —
    ``violations[]`` with stable ``rule`` + ``hint``, plus ``next_actions``.
    """
    import json

    try:
        target, _transport, _fs, ws = open_workspace(target_spec)
    except FileNotFoundError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        rprint(f"[red]Failed to open workspace:[/red] {exc}")
        raise typer.Exit(1) from exc

    report = ws.validate()

    if json_out:
        # Machine report for agents / molmcp — no rich markup.
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        if report.errors or (strict and report.warnings):
            raise typer.Exit(1)
        return

    rprint(f"[bold]Workspace:[/bold] {target}")

    # Rule ids are printed bare: rich would eat a bracketed ``[run.ops]`` as
    # console markup and silently drop it.
    for v in report.errors:
        rprint(f"  [red]error[/red]   {v.rule} — {v.path}: {v.detail}")
        if v.hint:
            rprint(f"           [dim]hint:[/dim] {v.hint}")
    for v in report.warnings:
        rprint(f"  [yellow]warning[/yellow] {v.rule} — {v.path}: {v.detail}")
        if v.hint:
            rprint(f"           [dim]hint:[/dim] {v.hint}")

    if report.ok and not report.warnings:
        rprint("\n[green]conforms[/green] — no violations")
        return

    rprint(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)
