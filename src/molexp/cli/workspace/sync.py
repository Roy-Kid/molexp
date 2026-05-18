"""``molexp workspace {sync,upload,download}`` — remote file operations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from molexp.cli._common import rprint
from molexp.cli.workspace import _get_ctx_target, workspace_app
from molexp.workspace.target import LocalTarget


@workspace_app.command()
def sync(
    ctx: typer.Context,
    source: Annotated[
        str | None,
        typer.Argument(help="Local path to sync (default: current directory)."),
    ] = None,
    pull: Annotated[
        bool,
        typer.Option("--pull", help="Pull remote → local (default: push local → remote)."),
    ] = False,
    delete: Annotated[  # noqa: ARG001
        bool,
        typer.Option("--delete", help="Delete dest files not in source."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be transferred."),
    ] = False,
) -> None:
    """Sync files between local and remote workspace via rsync.

    Default direction is push (local → remote). Use --pull to reverse.
    """
    target = _get_ctx_target(ctx)
    transport = ctx.obj["transport"]

    if isinstance(target, LocalTarget):
        rprint("[yellow]Sync is only meaningful for remote targets.[/yellow]")
        rprint("Local workspace files are already on disk.")
        return

    local_path = Path(source).resolve() if source else Path.cwd()
    remote_path = target.path

    if dry_run:
        direction = "pull remote → local" if pull else "push local → remote"
        rprint(f"[dim]Would {direction}:[/dim] {local_path} ↔ {target}:{remote_path}")
        return

    if pull:
        rprint(f"[dim]Pulling {target}:{remote_path} → {local_path}...[/dim]")
        try:
            transport.download(remote_path, str(local_path), recursive=True)
        except Exception as exc:
            rprint(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        rprint(f"[green]Pulled[/green] {target}:{remote_path} → {local_path}")
    else:
        rprint(f"[dim]Pushing {local_path} → {target}:{remote_path}...[/dim]")
        try:
            transport.upload(str(local_path), remote_path, recursive=True)
        except Exception as exc:
            rprint(f"[red]Upload failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        rprint(f"[green]Pushed[/green] {local_path} → {target}:{remote_path}")


@workspace_app.command()
def upload(
    ctx: typer.Context,
    local: Annotated[str, typer.Argument(help="Local file or directory to upload.")],
    remote: Annotated[
        str | None, typer.Argument(help="Remote path (default: same basename in workspace root).")
    ] = None,
    recursive: Annotated[
        bool, typer.Option("-r", "--recursive", help="Upload directory recursively.")
    ] = False,
) -> None:
    """Upload a file or directory to the remote workspace."""
    target = _get_ctx_target(ctx)
    transport = ctx.obj["transport"]

    if isinstance(target, LocalTarget):
        rprint("[yellow]Upload is only meaningful for remote targets.[/yellow]")
        return

    local_path = Path(local).expanduser().resolve()
    if not local_path.exists():
        rprint(f"[red]Error:[/red] Local path not found: {local_path}")
        raise typer.Exit(1)

    remote_rel = remote or local_path.name
    remote_path = f"{target.path.rstrip('/')}/{remote_rel}"

    rprint(f"[dim]Uploading {local_path} → {target}:{remote_path}...[/dim]")
    try:
        transport.upload(str(local_path), remote_path, recursive=recursive)
    except Exception as exc:
        rprint(f"[red]Upload failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] Uploaded to {target}:{remote_path}")


@workspace_app.command()
def download(
    ctx: typer.Context,
    remote: Annotated[str, typer.Argument(help="Remote file or directory to download.")],
    local: Annotated[
        str | None, typer.Argument(help="Local path (default: current directory).")
    ] = None,
    recursive: Annotated[
        bool, typer.Option("-r", "--recursive", help="Download directory recursively.")
    ] = False,
) -> None:
    """Download a file or directory from the remote workspace."""
    target = _get_ctx_target(ctx)
    transport = ctx.obj["transport"]

    if isinstance(target, LocalTarget):
        rprint("[yellow]Download is only meaningful for remote targets.[/yellow]")
        return

    remote_path = remote if remote.startswith("/") else f"{target.path.rstrip('/')}/{remote}"
    local_path = str(Path(local).expanduser().resolve()) if local else str(Path.cwd())

    rprint(f"[dim]Downloading {target}:{remote_path} → {local_path}...[/dim]")
    try:
        transport.download(remote_path, local_path, recursive=recursive)
    except Exception as exc:
        rprint(f"[red]Download failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] Downloaded to {local_path}")
