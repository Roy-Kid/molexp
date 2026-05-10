"""``molexp mcp ...`` — manage the user's MCP-server registry.

Mirrors the surface of ``claude mcp``: ``add``, ``add-json``, ``get``,
``list``, ``remove`` plus ``import`` / ``export`` for moving entries
between any two JSON files.

Storage is the same set of files Claude Code uses, so a single registry
serves both clients. **Schema and paths are identical to claude.**

* ``--scope user`` (default) → ``~/.claude.json`` top-level
  ``mcpServers``. Other top-level keys (claude's own settings) are
  preserved on every write.
* ``--scope project`` → ``<cwd>/.mcp.json``. Same envelope.
* ``--config PATH`` → explicit override. Wins over ``--scope``.

Server entry shape (matches Claude Code exactly)::

    # stdio
    {"type": "stdio", "command": "<cmd>", "args": [...], "env": {...}}

    # http / sse
    {"type": "http", "url": "<url>", "headers": {...}}

All JSON IO goes through :class:`molcfg.Config` so unknown top-level
keys (claude's other settings) and unknown per-entry keys (e.g. a
user's ``"usage_instructions"``) are preserved on every round-trip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from molcfg import Config, ConfigError
from rich.table import Table

from . import app
from ._common import console, rprint

mcp_app = typer.Typer(
    name="mcp",
    help="Configure and manage MCP servers (mirrors `claude mcp`).",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


# Scope → file path. Same files Claude Code uses, so a single registry
# can serve both clients.
_USER_SCOPE_PATH = Path.home() / ".claude.json"
_PROJECT_SCOPE_FILENAME = ".mcp.json"
_VALID_TRANSPORTS = ("stdio", "http", "sse")
_VALID_SCOPES = ("user", "project")


# ---------------------------------------------------------------------------
# Path resolution + molcfg-backed IO that preserves unknown keys
# ---------------------------------------------------------------------------


def _resolve_path(scope: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if scope == "user":
        return _USER_SCOPE_PATH
    if scope == "project":
        return Path.cwd() / _PROJECT_SCOPE_FILENAME
    raise typer.BadParameter(
        f"Unknown scope {scope!r}. Valid: {', '.join(_VALID_SCOPES)}."
    )


def _load_cfg(path: Path) -> Config:
    """Load *path* as a :class:`Config`, returning an empty envelope if missing.

    Always guarantees a ``mcpServers`` object exists at the top level so
    callers can index it without defensive checks.
    """
    if not path.exists():
        return Config({"mcpServers": {}})
    try:
        cfg = Config.load_json(path)
    except (json.JSONDecodeError, ConfigError) as exc:
        raise typer.BadParameter(
            f"Config file at {path} is not a valid JSON object: {exc}"
        ) from exc
    if "mcpServers" not in cfg:
        cfg["mcpServers"] = {}
    elif not isinstance(cfg["mcpServers"], Config):
        # mcpServers exists but isn't an object — refuse to clobber.
        raise typer.BadParameter(
            f"Config file at {path} has a non-object 'mcpServers' field."
        )
    return cfg


def _save_cfg(cfg: Config, path: Path) -> None:
    """Atomic-write *cfg* to *path* with 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cfg.save_json(tmp, indent=2)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _server_entry(cfg: Config, name: str) -> Config | None:
    """Return the server entry as a :class:`Config`, or ``None`` if absent."""
    return cfg.get(f"mcpServers.{name}")


def _server_names(cfg: Config) -> list[str]:
    servers = cfg["mcpServers"]
    return list(servers.keys()) if isinstance(servers, Config) else []


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_kv_pair(raw: str, *, sep: str, kind: str) -> tuple[str, str]:
    if sep not in raw:
        raise typer.BadParameter(
            f"Bad {kind} {raw!r}: expected '<key>{sep}<value>'."
        )
    key, value = raw.split(sep, 1)
    key = key.strip()
    if kind == "--header":
        value = value.strip()
    if not key:
        raise typer.BadParameter(f"Bad {kind} {raw!r}: empty key.")
    return key, value


def _build_server_entry(
    transport: str,
    command_or_url: str,
    args: list[str],
    env_pairs: list[str],
    header_pairs: list[str],
) -> dict[str, Any]:
    if transport not in _VALID_TRANSPORTS:
        raise typer.BadParameter(
            f"Unknown transport {transport!r}. "
            f"Valid: {', '.join(_VALID_TRANSPORTS)}."
        )
    if transport == "stdio":
        if header_pairs:
            raise typer.BadParameter("--header is only valid for http/sse transports.")
        env_dict = dict(_parse_kv_pair(e, sep="=", kind="--env") for e in env_pairs)
        return {
            "type": "stdio",
            "command": command_or_url,
            "args": list(args),
            "env": env_dict,
        }
    # http / sse
    if env_pairs:
        raise typer.BadParameter("--env is only valid for stdio transports.")
    if args:
        raise typer.BadParameter(
            "Positional args are only valid for stdio transports; "
            "use --header for http/sse."
        )
    headers = dict(_parse_kv_pair(h, sep=":", kind="--header") for h in header_pairs)
    return {
        "type": transport,
        "url": command_or_url,
        "headers": headers,
    }


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------


_ScopeOpt = Annotated[
    str,
    typer.Option(
        "--scope",
        "-s",
        help="Where to read/write: user (~/.claude.json) or project "
        "(<cwd>/.mcp.json).",
    ),
]
_ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        help="Override the path resolved by --scope. Wins over --scope.",
    ),
]


# ---------------------------------------------------------------------------
# Commands — matching `claude mcp` surface
# ---------------------------------------------------------------------------


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name (registry key).")],
    command_or_url: Annotated[
        str,
        typer.Argument(
            help="Command (stdio) or URL (http/sse).",
            metavar="COMMAND_OR_URL",
        ),
    ],
    args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Trailing args passed to the stdio subprocess "
            "(after a `--` separator).",
        ),
    ] = None,
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="stdio | http | sse"),
    ] = "stdio",
    env: Annotated[
        list[str] | None,
        typer.Option(
            "--env", "-e",
            help="Env var (stdio only). Repeatable. Format: KEY=VALUE.",
        ),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option(
            "--header", "-H",
            help="HTTP header (http/sse only). Repeatable. Format: NAME:VALUE.",
        ),
    ] = None,
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite if entry exists."),
    ] = False,
) -> None:
    """Add an MCP server to the registry."""
    cfg_path = _resolve_path(scope, config)
    cfg = _load_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in "
            f"{cfg_path}. Pass --force to overwrite."
        )
        raise typer.Exit(1)
    entry = _build_server_entry(
        transport=transport,
        command_or_url=command_or_url,
        args=list(args or []),
        env_pairs=list(env or []),
        header_pairs=list(header or []),
    )
    cfg[f"mcpServers.{name}"] = entry
    _save_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] ({transport}) to {cfg_path}")


@mcp_app.command("add-json")
def mcp_add_json(
    name: Annotated[str, typer.Argument(help="Server name (registry key).")],
    json_str: Annotated[
        str,
        typer.Argument(help="JSON object describing the server entry.", metavar="JSON"),
    ],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite if entry exists."),
    ] = False,
) -> None:
    """Add an MCP server from a raw JSON string (full schema control)."""
    try:
        entry = json.loads(json_str)
    except json.JSONDecodeError as exc:
        rprint(f"[red]Error:[/red] Invalid JSON: {exc}")
        raise typer.Exit(1) from None
    if not isinstance(entry, dict):
        rprint(f"[red]Error:[/red] JSON must be an object, got {type(entry).__name__}.")
        raise typer.Exit(1)

    cfg_path = _resolve_path(scope, config)
    cfg = _load_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in "
            f"{cfg_path}. Pass --force to overwrite."
        )
        raise typer.Exit(1)
    cfg[f"mcpServers.{name}"] = entry
    _save_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] from JSON to {cfg_path}")


@mcp_app.command("get")
def mcp_get(
    name: Annotated[str, typer.Argument(help="Server name.")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Print the JSON entry for a single server."""
    cfg_path = _resolve_path(scope, config)
    cfg = _load_cfg(cfg_path)
    entry = _server_entry(cfg, name)
    if entry is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    rprint(f"[bold]{name}[/bold]  ([dim]{cfg_path}[/dim])")
    payload = entry.to_dict() if isinstance(entry, Config) else entry
    console.print_json(json.dumps(payload))


@mcp_app.command("list")
def mcp_list(
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """List configured MCP servers."""
    cfg_path = _resolve_path(scope, config)
    cfg = _load_cfg(cfg_path)
    names = sorted(_server_names(cfg))
    if not names:
        rprint(f"[yellow]No MCP servers configured[/yellow] ({cfg_path})")
        return

    table = Table(title=f"MCP servers ({cfg_path})")
    table.add_column("Name", style="cyan")
    table.add_column("Transport", style="green")
    table.add_column("Command / URL")
    table.add_column("Args / Headers", overflow="fold")
    for name in names:
        entry = _server_entry(cfg, name)
        if isinstance(entry, Config):
            entry = entry.to_dict()
        if not isinstance(entry, dict):
            entry = {}
        transport = entry.get("type") or ("http" if "url" in entry else "stdio")
        if "url" in entry:
            target = entry["url"]
            extra = ", ".join(
                f"{k}={v!r}" for k, v in (entry.get("headers") or {}).items()
            )
        else:
            target = entry.get("command", "")
            extra = " ".join(entry.get("args") or [])
        table.add_row(name, transport, target, extra)
    console.print(table)


@mcp_app.command("remove")
def mcp_remove(
    name: Annotated[str, typer.Argument(help="Server name.")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Remove an MCP server from the registry."""
    cfg_path = _resolve_path(scope, config)
    cfg = _load_cfg(cfg_path)
    if _server_entry(cfg, name) is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    del cfg[f"mcpServers.{name}"]
    _save_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Removed [bold]{name}[/bold] from {cfg_path}")


# ---------------------------------------------------------------------------
# import / export — bridge between any two compatible JSON files
# ---------------------------------------------------------------------------


def _copy_servers(
    src_cfg: Config, dst_cfg: Config, *, force: bool
) -> tuple[list[str], list[str], list[str]]:
    """Copy ``mcpServers`` from *src_cfg* into *dst_cfg* in place.

    Returns ``(added, overwritten, skipped)``.
    """
    added: list[str] = []
    skipped: list[str] = []
    overwritten: list[str] = []
    for name in _server_names(src_cfg):
        entry = src_cfg[f"mcpServers.{name}"]
        if isinstance(entry, Config):
            entry = entry.to_dict()
        target_exists = _server_entry(dst_cfg, name) is not None
        if target_exists and not force:
            skipped.append(name)
            continue
        dst_cfg[f"mcpServers.{name}"] = entry
        (overwritten if target_exists else added).append(name)
    return added, overwritten, skipped


def _print_copy_summary(
    *,
    verb: str,
    src: Path,
    dst: Path,
    added: list[str],
    overwritten: list[str],
    skipped: list[str],
) -> None:
    rprint(
        f"[green]OK[/green] {verb} {len(added)} new, "
        f"{len(overwritten)} overwritten, {len(skipped)} skipped "
        f"({src} → {dst})"
    )
    if added:
        rprint(f"  added:       {', '.join(added)}")
    if overwritten:
        rprint(f"  overwritten: {', '.join(overwritten)}")
    if skipped:
        rprint(
            f"  skipped:     {', '.join(skipped)}  "
            "[dim](pass --force to overwrite)[/dim]"
        )


@mcp_app.command("import")
def mcp_import(
    from_path: Annotated[
        Path,
        typer.Argument(
            help="Source JSON file with `mcpServers` envelope "
            "(e.g. ~/.molexp/mcp.json or another claude config).",
        ),
    ],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite entries that already exist."),
    ] = False,
) -> None:
    """Copy `mcpServers` from FROM_PATH into the active registry.

    Conflict resolution:
      - Without --force: existing entries in the target are kept; only
        new names from the source are added.
      - With --force: every entry in the source overwrites its
        same-named target counterpart.

    All other top-level keys in both files are preserved untouched.
    """
    src_path = from_path.expanduser().resolve()
    if not src_path.exists():
        rprint(f"[red]Error:[/red] Source file not found: {src_path}")
        raise typer.Exit(1)
    src_cfg = _load_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Source has no `mcpServers` to import:[/yellow] {src_path}")
        return

    dst_path = _resolve_path(scope, config)
    dst_cfg = _load_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_cfg(dst_cfg, dst_path)
    _print_copy_summary(
        verb="Imported",
        src=src_path,
        dst=dst_path,
        added=added,
        overwritten=overwritten,
        skipped=skipped,
    )


@mcp_app.command("export")
def mcp_export(
    to_path: Annotated[
        Path,
        typer.Argument(
            help="Destination JSON file. Created if missing; existing top-level "
            "keys outside `mcpServers` are preserved.",
        ),
    ],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite entries that already exist."),
    ] = False,
) -> None:
    """Copy the active `mcpServers` registry into TO_PATH.

    Useful for: sharing the user-scope registry to a project's
    ``.mcp.json``, or pushing molexp config into a peer claude install.
    """
    src_path = _resolve_path(scope, config)
    src_cfg = _load_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Active registry is empty:[/yellow] {src_path}")
        return

    dst_path = to_path.expanduser().resolve()
    dst_cfg = _load_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_cfg(dst_cfg, dst_path)
    _print_copy_summary(
        verb="Exported",
        src=src_path,
        dst=dst_path,
        added=added,
        overwritten=overwritten,
        skipped=skipped,
    )


__all__ = ["mcp_app"]
