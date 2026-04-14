"""``molexp asset ...`` — asset management sub-commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.table import Table

from . import app
from ._common import console, get_workspace, rprint

asset_app = typer.Typer(help="Asset management commands")
app.add_typer(asset_app, name="asset")


@asset_app.command("list")
def asset_list(
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
) -> None:
    """List workspace-level assets."""
    ws = get_workspace(path)
    assets = ws.assets.list_assets()[:limit]

    if not assets:
        rprint("[yellow]No assets found[/yellow]")
        return

    table = Table(title="Workspace Assets")
    table.add_column("Asset ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Created")

    for a in assets:
        table.add_row(
            a.asset_id[:12] + "...",
            a.name,
            a.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
