"""``molexp workspace -i`` — drop the user into a shell on the target."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from molexp.cli._common import rprint
from molexp.workspace.target import LocalTarget, RemoteTarget


def run_interactive(target: Any, transport: Any, target_raw: str) -> None:  # noqa: ANN401, ARG001
    """Open an interactive shell on the workspace target."""
    if isinstance(target, LocalTarget):
        _local_shell(target)
    else:
        _remote_shell(target)


def _local_shell(target: LocalTarget) -> None:
    ws = target.path
    rprint(f"[bold]Local workspace:[/bold] {ws}")
    shell = os.environ.get("SHELL", "/bin/sh")
    subprocess.run([shell], cwd=str(ws))


def _remote_shell(target: RemoteTarget) -> None:
    user_host = f"{target.user}@{target.host}" if target.user else target.host
    cmd = f"cd {target.path} && exec ${{SHELL:-bash}}"
    ssh_argv = [
        "ssh",
        "-t",
        "-o",
        "RemoteCommand=none",
        "-o",
        "RequestTTY=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        user_host,
        cmd,
    ]
    if target.port:
        ssh_argv.insert(1, "-p")
        ssh_argv.insert(2, str(target.port))
    if target.identity_file:
        ssh_argv.insert(1, "-i")
        ssh_argv.insert(2, target.identity_file)

    rprint(f"[dim]Opening shell on {user_host}...[/dim]")
    subprocess.run(ssh_argv)
