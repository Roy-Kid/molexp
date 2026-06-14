"""``molexp rebuild-catalog`` — rebuild the derived asset catalog.

The workspace catalog (``<root>/catalog/index.sqlite``) is **derived** from
entity ``*.json`` + asset manifests, which remain the single source of truth.
This command drops the derived DB and rewalks the tree — the self-heal path
for a deleted, stale, or concurrently-corrupted catalog. Local targets only.
"""

from __future__ import annotations

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import LocalTarget


@app.command(name="rebuild-catalog")
def rebuild_catalog(target_spec: TargetOption = ".") -> None:
    """Rebuild the derived asset catalog from on-disk entity state."""
    from molexp.workspace import Workspace

    target, _transport, fs = resolve_workspace_target(target_spec)
    if not isinstance(target, LocalTarget):
        rprint("[red]rebuild-catalog is only supported on local workspaces.[/red]")
        raise typer.Exit(1)

    try:
        ws = Workspace(str(target), fs=fs)
    except Exception as exc:
        rprint(f"[red]Failed to open workspace:[/red] {exc}")
        raise typer.Exit(1) from exc

    report = ws.catalog.rebuild()

    rprint(f"[bold]Rebuilt catalog:[/bold] {ws.root / 'catalog'}")
    rprint(f"  Workspaces:  {report.workspaces}")
    rprint(f"  Projects:    {report.projects}")
    rprint(f"  Experiments: {report.experiments}")
    rprint(f"  Runs:        {report.runs}")
    rprint(f"  Executions:  {report.executions}")
    rprint(f"  Assets:      {report.assets}")

    if report.errors:
        rprint(f"\n[red]{len(report.errors)} error(s) during rebuild:[/red]")
        for err in report.errors:
            rprint(f"  [red]•[/red] {err}")
        raise typer.Exit(1)
