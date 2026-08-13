"""Auth enablement checks for ``molexp serve`` (kept out of the hot serve path)."""

from __future__ import annotations

import ipaddress
import socket

import typer

from molexp.cli._common import rprint
from molexp.services.auth import (
    DEFAULT_ADMIN_USERNAME,
    get_auth_service,
    set_auth_enabled,
)
from molexp.services.operator_config import load_operator_config


def is_loopback_host(host: str) -> bool:
    """Return True when *host* only accepts local connections."""
    normalized = (host or "").strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # Hostnames: resolve and require every address to be loopback.
        try:
            infos = socket.getaddrinfo(normalized, None)
        except socket.gaierror:
            return False
        if not infos:
            return False
        for info in infos:
            addr = info[4][0]
            try:
                if not ipaddress.ip_address(addr).is_loopback:
                    return False
            except ValueError:
                return False
        return True


def auth_enabled_from_config() -> bool:
    cfg = load_operator_config()
    auth = cfg.get("auth")
    if isinstance(auth, dict):
        return bool(auth.get("enabled"))
    return False


def configure_serve_auth(
    *,
    host: str,
    auth_flag: bool,
    require_user: str | None,
) -> None:
    """Enable process auth (or refuse) based on flags / config / bind address.

    Raises :class:`typer.Exit` on misconfiguration.
    """
    want_auth = bool(auth_flag) or auth_enabled_from_config()
    loopback = is_loopback_host(host)

    if not want_auth and not loopback:
        rprint(
            "[red]Error:[/red] binding non-loopback "
            f"[bold]{host}[/bold] without auth is refused. "
            "Pass [cyan]--auth[/cyan] (and create a user with "
            f"[cyan]molexp auth login -u {DEFAULT_ADMIN_USERNAME}[/cyan]), "
            "or bind [cyan]--host localhost[/cyan]."
        )
        raise typer.Exit(1)

    if not want_auth:
        set_auth_enabled(False)
        return

    service = get_auth_service()
    service.ensure_layout()
    if not service.has_users():
        rprint(
            "[red]Error:[/red] auth is enabled but no users exist under "
            "~/.molexp/auth/.\n"
            f"  Bootstrap: [cyan]molexp auth login -u {DEFAULT_ADMIN_USERNAME}[/cyan]"
        )
        raise typer.Exit(1)

    if require_user:
        user = service.get_user(require_user)
        if user is None or user.disabled:
            rprint(
                f"[red]Error:[/red] --user {require_user!r} does not exist "
                f"(or is disabled). Create with "
                f"[cyan]molexp auth login -u {require_user}[/cyan]"
            )
            raise typer.Exit(1)

    set_auth_enabled(True)
    who = require_user or "(any configured user)"
    rprint(f"[bold]Auth enabled[/bold] — login required ({who})")
