"""``molexp session`` — manage persistent transport sessions.

Sessions are cached SSH connections to remote workspace targets.
They are created automatically by ``molexp workspace <target>`` and
persist until explicitly closed with ``molexp session close``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from molexp.cli._common import rprint

session_app = typer.Typer(
    name="session",
    help="Manage persistent workspace sessions.",
    no_args_is_help=True,
)


@session_app.command("list")
def session_list() -> None:
    """List active sessions."""
    from molexp.workspace.target import SessionManager

    sessions = SessionManager.list_sessions()
    if not sessions:
        rprint("[dim]No active sessions.[/dim]")
        return

    from rich.table import Table

    table = Table(title="Active Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Host", style="green")
    table.add_column("Path")
    table.add_column("Created")
    table.add_column("Last Used")

    from datetime import datetime

    for s in sessions:
        table.add_row(
            s.name,
            s.target.host,
            s.target.path,
            datetime.fromtimestamp(s.created_at).strftime("%H:%M:%S"),
            datetime.fromtimestamp(s.last_used).strftime("%H:%M:%S"),
        )
    from rich import print as _rprint

    _rprint(table)


@session_app.command("show")
def session_show(
    name: Annotated[str, typer.Argument(help="Session name (SCP notation).")],
) -> None:
    """Show details for a session."""
    from molexp.workspace.target import SessionManager

    session = SessionManager.get_by_name(name)
    if session is None:
        rprint(f"[yellow]Session not found:[/yellow] {name}")
        rprint("Use [bold]molexp session list[/bold] to see active sessions.")
        raise typer.Exit(1)

    from datetime import datetime

    rprint(f"[bold]Session:[/bold] {session.name}")
    rprint(f"  Host:      {session.target.host}")
    rprint(f"  User:      {session.target.user or '(default)'}")
    rprint(f"  Port:      {session.target.port or 22}")
    rprint(f"  Path:      {session.target.path}")
    rprint(f"  Created:   {datetime.fromtimestamp(session.created_at)}")
    rprint(f"  Last used: {datetime.fromtimestamp(session.last_used)}")


@session_app.command("close")
def session_close(
    name: Annotated[
        str | None, typer.Argument(help="Session name to close (omit with --all).")
    ] = None,
    all_sessions: Annotated[bool, typer.Option("--all", help="Close all sessions.")] = False,
) -> None:
    """Close one or all sessions."""
    from molexp.workspace.target import SessionManager

    if all_sessions:
        count = SessionManager.close_all()
        rprint(f"[green]OK[/green] Closed {count} session(s).")
        return

    if name is None:
        rprint("[red]Error:[/red] Provide a session name or --all.")
        raise typer.Exit(1)

    if SessionManager.close(name):
        rprint(f"[green]OK[/green] Closed session: {name}")
    else:
        rprint(f"[yellow]Session not found:[/yellow] {name}")
        raise typer.Exit(1)
