"""``molexp connect`` — interactive SSH login for 2FA / verification-code hosts.

HPC frontends (Arrhenius, Dardel, Alvis, …) often require a one-time
verification code via keyboard-interactive auth.  molq/molexp routine
ops run under ``BatchMode=yes`` and cannot answer that prompt.  This
command opens an interactive SSH session that establishes (or reuses) the
OpenSSH ControlMaster multiplex socket; subsequent BatchMode ops ride it
for the duration of ``ControlPersist``.
"""

from __future__ import annotations

import typer

from molexp.cli._app import app
from molexp.cli._common import rprint
from molexp.cli._target import TargetOption, resolve_workspace_target
from molexp.workspace.target import RemoteTarget


@app.command("connect")
def connect(
    target_spec: TargetOption = ".",
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-authenticate even when a ControlMaster is already alive.",
    ),
) -> None:
    """Interactively log in to a remote workspace host (OTP / 2FA).

    *target_spec* is the same form as other commands: ``Host:/path``,
    ``user@host:/path``, or ``@registered-target``.  Local paths are a
    no-op (nothing to connect).

    Example::

        molexp connect -ws Arrhenius:/home/jicli594/work/mace-nve
        # enter verification code when prompted
        molexp serve -ws Arrhenius:/home/jicli594/work/mace-nve --dev
    """
    target, transport, _fs = resolve_workspace_target(target_spec)

    if not isinstance(target, RemoteTarget):
        rprint("[dim]Local workspace — nothing to connect.[/dim]")
        raise typer.Exit(0)

    host = target.host or str(target)
    login = getattr(transport, "login", None)
    is_alive = getattr(transport, "is_master_alive", None)

    if not callable(login):
        rprint(
            f"[red]Error:[/red] transport for {host!r} does not support "
            f"interactive login (expected SshTransport)."
        )
        raise typer.Exit(1)

    if not force and callable(is_alive) and is_alive():
        rprint(
            f"[green]OK[/green] SSH master already alive for [bold]{host}[/bold] "
            f"— BatchMode ops will reuse it."
        )
        raise typer.Exit(0)

    rprint(f"[dim]Connecting to [bold]{host}[/bold] (enter verification code if prompted)…[/dim]")
    try:
        login(force=force)
    except Exception as exc:
        rprint(f"[red]Error:[/red] {exc}")
        rprint(
            f"[dim]Fallback: run [bold]ssh {host}[/bold] in this terminal, "
            f"then re-try molexp commands.[/dim]"
        )
        raise typer.Exit(1) from exc

    if callable(is_alive) and is_alive():
        rprint(
            f"[green]OK[/green] Logged in to [bold]{host}[/bold] — "
            f"ControlMaster is live. Keep ControlPersist long enough for "
            f"your session (see ~/.ssh/config)."
        )
    else:
        rprint(
            f"[yellow]Logged in to [bold]{host}[/bold], but no ControlMaster "
            f"socket was detected.[/yellow] Enable ControlMaster/ControlPath "
            f"in ~/.ssh/config so BatchMode ops can reuse the session."
        )
