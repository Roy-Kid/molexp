"""``molexp auth`` — filesystem auth, shaped like ``gh auth``.

Core verbs: login / logout / status / switch / token / refresh.
User-store admin: ``molexp auth users …``.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import sys
from typing import Annotated, cast

import typer

from molexp.cli._common import rprint
from molexp.services.auth import (
    DEFAULT_ADMIN_USERNAME,
    AuthError,
    get_auth_service,
    is_auth_enabled,
)
from molexp.services.auth.models import VALID_ROLES, AuthRole

auth_app = typer.Typer(
    name="auth",
    help="Authenticate and manage filesystem users (gh auth-shaped).",
    no_args_is_help=True,
)

users_app = typer.Typer(
    name="users",
    help="Manage the local user store (~/.molexp/auth/users.json).",
    no_args_is_help=True,
)
auth_app.add_typer(users_app, name="users")

# Local CLI session token path (for auth token / status when not talking HTTP).
_CLI_SESSION_FILE = "cli_session"


def _read_password(*, prompt: str = "Password: ") -> str:
    env = os.environ.get("MOLEXP_AUTH_PASSWORD")
    if env is not None and env != "":
        return env
    if not sys.stdin.isatty():
        # --password-stdin path sets this via caller; bare pipe:
        line = sys.stdin.readline()
        return line.rstrip("\n")
    return getpass.getpass(prompt)


def _password_from_flags(*, password_stdin: bool) -> str:
    if password_stdin:
        return sys.stdin.readline().rstrip("\n")
    return _read_password()


def _cli_session_path():
    from molexp.services.auth import get_auth_service

    return get_auth_service().root / _CLI_SESSION_FILE


def _load_cli_session_id() -> str | None:
    path = _cli_session_path()
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _save_cli_session_id(session_id: str) -> None:
    service = get_auth_service()
    service.ensure_layout()
    path = _cli_session_path()
    path.write_text(session_id + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def _clear_cli_session() -> None:
    path = _cli_session_path()
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


@auth_app.command("login")
def auth_login(
    user: Annotated[
        str,
        typer.Option("--user", "-u", help="Account to log in as."),
    ] = DEFAULT_ADMIN_USERNAME,
    password_stdin: Annotated[
        bool,
        typer.Option(
            "--password-stdin",
            help="Read password from stdin (like gh auth login --with-token).",
        ),
    ] = False,
    create: Annotated[
        bool,
        typer.Option(
            "--create",
            help="Create the user if missing (bootstrap: first user becomes admin).",
        ),
    ] = False,
    role: Annotated[
        str,
        typer.Option("--role", help="Role when creating a user."),
    ] = "admin",
) -> None:
    """Log in to a local filesystem account (default: admin).

    When the user store is empty, creates *user* as admin (bootstrap).
    Otherwise verifies the password and stores a CLI session token under
    ``~/.molexp/auth/cli_session``.
    """
    service = get_auth_service()
    service.ensure_layout()
    password = _password_from_flags(password_stdin=password_stdin)
    if not password:
        rprint("[red]Error:[/red] empty password")
        raise typer.Exit(1)

    existing = service.get_user(user)
    if existing is None:
        if service.has_users() and not create:
            rprint(
                f"[red]Error:[/red] user {user!r} does not exist. "
                "Pass [cyan]--create[/cyan] or "
                f"[cyan]molexp auth users create -u {user}[/cyan]."
            )
            raise typer.Exit(1)
        # Bootstrap or explicit create.
        if role not in VALID_ROLES:
            rprint(f"[red]Error:[/red] invalid role {role!r}")
            raise typer.Exit(1)
        try:
            if not service.has_users():
                created = service.bootstrap_admin(password, username=user)
            else:
                created = service.create_user(user, password, role=cast(AuthRole, role))
        except AuthError as exc:
            rprint(f"[red]Error:[/red] {exc.message}")
            raise typer.Exit(1) from exc
        rprint(
            f"[green]OK[/green] created user [cyan]{created.username}[/cyan] role={created.role}"
        )
        # Fall through to login so CLI session is minted.
    try:
        authed, session = service.login(user, password)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    _save_cli_session_id(session.session_id)
    rprint(f"[green]OK[/green] logged in as [cyan]{authed.username}[/cyan] ({authed.role})")


@auth_app.command("logout")
def auth_logout(
    user: Annotated[
        str | None,
        typer.Option("--user", "-u", help="If set, revoke all sessions for this user."),
    ] = None,
) -> None:
    """Log out of the active CLI session (or revoke a user's sessions)."""
    service = get_auth_service()
    if user:
        n = service.sessions.revoke_user(user)
        rprint(f"[green]OK[/green] revoked {n} session(s) for {user}")
        if _load_cli_session_id():
            current = service.resolve_session(_load_cli_session_id())
            if current is not None and current.username == user:
                _clear_cli_session()
        return
    sid = _load_cli_session_id()
    service.logout(sid)
    _clear_cli_session()
    rprint("[green]OK[/green] logged out")


@auth_app.command("status")
def auth_status() -> None:
    """Show auth enable flag and the active CLI account."""
    service = get_auth_service()
    sid = _load_cli_session_id()
    state = service.status(sid, enabled=is_auth_enabled())
    rprint(f"auth.enabled (process): {is_auth_enabled()}")
    rprint(f"users on disk: {service.users.count()}")
    if state.authenticated and state.user is not None:
        u = state.user
        rprint(f"logged in: [cyan]{u.username}[/cyan] role={u.role}")
    else:
        # status() forces unauthenticated when process auth is off; still show CLI session.
        user = service.resolve_session(sid)
        if user is not None:
            rprint(
                f"CLI session: [cyan]{user.username}[/cyan] role={user.role} "
                f"[dim](process auth {'on' if is_auth_enabled() else 'off'})[/dim]"
            )
        else:
            rprint("logged in: [dim]no[/dim]")


@auth_app.command("switch")
def auth_switch(
    user: Annotated[str, typer.Option("--user", "-u", help="Account to switch to.")],
    password_stdin: Annotated[
        bool,
        typer.Option("--password-stdin", help="Read password from stdin."),
    ] = False,
) -> None:
    """Switch the active CLI account (gh auth switch)."""
    service = get_auth_service()
    password = _password_from_flags(password_stdin=password_stdin)
    sid = _load_cli_session_id()
    try:
        authed, session = service.switch(sid, user, password)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    _save_cli_session_id(session.session_id)
    rprint(f"[green]OK[/green] switched to [cyan]{authed.username}[/cyan]")


@auth_app.command("token")
def auth_token() -> None:
    """Print the active CLI session token (for Authorization: Bearer)."""
    sid = _load_cli_session_id()
    token = get_auth_service().token_for(sid)
    if not token:
        rprint("[red]Error:[/red] not logged in")
        raise typer.Exit(1)
    # stdout only the token — scripts pipe this (gh auth token style).
    sys.stdout.write(token + "\n")


@auth_app.command("refresh")
def auth_refresh() -> None:
    """Extend the active CLI session TTL."""
    sid = _load_cli_session_id()
    record = get_auth_service().refresh(sid)
    if record is None:
        rprint("[red]Error:[/red] not logged in")
        raise typer.Exit(1)
    _save_cli_session_id(record.session_id)
    rprint(f"[green]OK[/green] session refreshed (expires {record.expires_at})")


# ── users subcommands ────────────────────────────────────────────────────────


@users_app.command("list")
def users_list() -> None:
    """List local users."""
    users = get_auth_service().list_users()
    if not users:
        rprint("[dim]No users.[/dim]")
        return
    for u in users:
        flag = " disabled" if u.disabled else ""
        rprint(
            f"[cyan]{u.username}[/cyan]  role={u.role}  workspaces={','.join(u.workspaces)}{flag}"
        )


@users_app.command("create")
def users_create(
    user: Annotated[str, typer.Option("--user", "-u", help="Username to create.")],
    role: Annotated[str, typer.Option("--role")] = "operator",
    workspaces: Annotated[
        str | None,
        typer.Option("--workspaces", help="Comma-separated keys or *."),
    ] = "*",
    password_stdin: Annotated[bool, typer.Option("--password-stdin")] = False,
) -> None:
    """Create a user in the local store."""
    if role not in VALID_ROLES:
        rprint(f"[red]Error:[/red] invalid role {role!r}")
        raise typer.Exit(1)
    password = _password_from_flags(password_stdin=password_stdin)
    ws = [w.strip() for w in (workspaces or "*").split(",") if w.strip()]
    try:
        created = get_auth_service().create_user(
            user, password, role=cast(AuthRole, role), workspaces=ws
        )
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] created {created.username} role={created.role}")


@users_app.command("delete")
def users_delete(
    user: Annotated[str, typer.Option("--user", "-u")],
) -> None:
    try:
        get_auth_service().delete_user(user)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] deleted {user}")


@users_app.command("set-role")
def users_set_role(
    user: Annotated[str, typer.Option("--user", "-u")],
    role: Annotated[str, typer.Argument(help="admin | operator | viewer")],
) -> None:
    if role not in VALID_ROLES:
        rprint(f"[red]Error:[/red] invalid role {role!r}")
        raise typer.Exit(1)
    try:
        updated = get_auth_service().set_role(user, cast(AuthRole, role))
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] {updated.username} role={updated.role}")


@users_app.command("set-workspaces")
def users_set_workspaces(
    user: Annotated[str, typer.Option("--user", "-u")],
    keys: Annotated[list[str], typer.Argument(help="Workspace keys or *")],
) -> None:
    try:
        updated = get_auth_service().set_workspaces(user, keys)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] {updated.username} workspaces={','.join(updated.workspaces)}")


@users_app.command("passwd")
def users_passwd(
    user: Annotated[str, typer.Option("--user", "-u")],
    password_stdin: Annotated[bool, typer.Option("--password-stdin")] = False,
) -> None:
    password = _password_from_flags(password_stdin=password_stdin)
    try:
        get_auth_service().set_password(user, password)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] password updated for {user}")


@users_app.command("disable")
def users_disable(
    user: Annotated[str, typer.Option("--user", "-u")],
) -> None:
    try:
        get_auth_service().set_disabled(user, True)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] disabled {user}")


@users_app.command("enable")
def users_enable(
    user: Annotated[str, typer.Option("--user", "-u")],
) -> None:
    try:
        get_auth_service().set_disabled(user, False)
    except AuthError as exc:
        rprint(f"[red]Error:[/red] {exc.message}")
        raise typer.Exit(1) from exc
    rprint(f"[green]OK[/green] enabled {user}")
