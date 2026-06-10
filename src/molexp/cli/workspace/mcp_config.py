"""``molexp mcp`` — MCP server registry configuration (mirrors ``claude mcp``).

Extracted from ``resources.py``: this group edits MCP-config JSON files
(``~/.claude.json`` or project ``.mcp.json``), not workspace resources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from molcfg import Config, ConfigError
from rich.console import Console
from rich.table import Table

from molexp.cli._common import rprint

_console = Console()

mcp_app = typer.Typer(
    name="mcp", help="Configure MCP servers (mirrors `claude mcp`).", no_args_is_help=True
)

_USER_SCOPE_PATH = Path.home() / ".claude.json"
_PROJECT_SCOPE_FILENAME = ".mcp.json"
_VALID_TRANSPORTS = ("stdio", "http", "sse")
_VALID_SCOPES = ("user", "project")

_ScopeOpt = Annotated[str, typer.Option("--scope", "-s", help="user or project")]
_ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Override path from --scope")]


def _resolve_mcp_path(scope: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if scope == "user":
        return _USER_SCOPE_PATH
    if scope == "project":
        return Path.cwd() / _PROJECT_SCOPE_FILENAME
    raise typer.BadParameter(f"Unknown scope {scope!r}. Valid: {', '.join(_VALID_SCOPES)}.")


def _load_mcp_cfg(path: Path) -> Config:
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
        raise typer.BadParameter(f"Config file at {path} has a non-object 'mcpServers' field.")
    return cfg


def _save_mcp_cfg(cfg: Config, path: Path) -> None:
    import os as _os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cfg.save_json(tmp, indent=2)
    _os.chmod(tmp, 0o600)  # noqa: PTH101
    tmp.replace(path)


def _server_entry(cfg: Config, name: str) -> Config | None:
    return cfg.get(f"mcpServers.{name}")


def _server_names(cfg: Config) -> list[str]:
    servers = cfg["mcpServers"]
    return list(servers.keys()) if isinstance(servers, Config) else []


def _parse_kv_pair(raw: str, *, sep: str, kind: str) -> tuple[str, str]:
    if sep not in raw:
        raise typer.BadParameter(f"Bad {kind} {raw!r}: expected '<key>{sep}<value>'.")
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
            f"Unknown transport {transport!r}. Valid: {', '.join(_VALID_TRANSPORTS)}."
        )
    if transport == "stdio":
        if header_pairs:
            raise typer.BadParameter("--header is only valid for http/sse transports.")
        env_dict = dict(_parse_kv_pair(e, sep="=", kind="--env") for e in env_pairs)
        return {"type": "stdio", "command": command_or_url, "args": list(args), "env": env_dict}
    if env_pairs:
        raise typer.BadParameter("--env is only valid for stdio transports.")
    if args:
        raise typer.BadParameter("Positional args are only valid for stdio transports.")
    headers = dict(_parse_kv_pair(h, sep=":", kind="--header") for h in header_pairs)
    return {"type": transport, "url": command_or_url, "headers": headers}


def _copy_servers(
    src_cfg: Config, dst_cfg: Config, *, force: bool
) -> tuple[list[str], list[str], list[str]]:
    added, overwritten, skipped = [], [], []
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


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name")],
    command_or_url: Annotated[
        str, typer.Argument(help="Command (stdio) or URL (http/sse)", metavar="COMMAND_OR_URL")
    ],
    args: Annotated[
        list[str] | None, typer.Argument(help="Trailing args (after -- separator)")
    ] = None,
    transport: Annotated[
        str, typer.Option("--transport", "-t", help="stdio | http | sse")
    ] = "stdio",
    env: Annotated[
        list[str] | None,
        typer.Option("--env", "-e", help="Env var (stdio only, repeatable, KEY=VALUE)"),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header (http/sse only, repeatable, NAME:VALUE)"),
    ] = None,
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if entry exists")] = False,
) -> None:
    """Add an MCP server to the registry."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in {cfg_path}. Pass --force to overwrite."
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
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] ({transport}) to {cfg_path}")


@mcp_app.command("add-json")
def mcp_add_json(
    name: Annotated[str, typer.Argument(help="Server name")],
    json_str: Annotated[
        str, typer.Argument(help="JSON object describing the server entry", metavar="JSON")
    ],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if entry exists")] = False,
) -> None:
    """Add an MCP server from a raw JSON string."""
    try:
        entry = json.loads(json_str)
    except json.JSONDecodeError as exc:
        rprint(f"[red]Error:[/red] Invalid JSON: {exc}")
        raise typer.Exit(1) from None
    if not isinstance(entry, dict):
        rprint(f"[red]Error:[/red] JSON must be an object, got {type(entry).__name__}.")
        raise typer.Exit(1)
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is not None and not force:
        rprint(
            f"[red]Error:[/red] Server [bold]{name}[/bold] already exists in {cfg_path}. Pass --force to overwrite."
        )
        raise typer.Exit(1)
    cfg[f"mcpServers.{name}"] = entry
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Added [bold]{name}[/bold] from JSON to {cfg_path}")


@mcp_app.command("get")
def mcp_get(
    name: Annotated[str, typer.Argument(help="Server name")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Print the JSON entry for a single server."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    entry = _server_entry(cfg, name)
    if entry is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    rprint(f"[bold]{name}[/bold]  ([dim]{cfg_path}[/dim])")
    payload = entry.to_dict() if isinstance(entry, Config) else entry
    _console.print_json(json.dumps(payload))


@mcp_app.command("list")
def mcp_list(
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """List configured MCP servers."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
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
            extra = ", ".join(f"{k}={v!r}" for k, v in (entry.get("headers") or {}).items())
        else:
            target = entry.get("command", "")
            extra = " ".join(entry.get("args") or [])
        table.add_row(name, transport, target, extra)
    _console.print(table)


@mcp_app.command("remove")
def mcp_remove(
    name: Annotated[str, typer.Argument(help="Server name")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
) -> None:
    """Remove an MCP server from the registry."""
    cfg_path = _resolve_mcp_path(scope, config)
    cfg = _load_mcp_cfg(cfg_path)
    if _server_entry(cfg, name) is None:
        rprint(f"[red]Error:[/red] Server [bold]{name}[/bold] not found in {cfg_path}.")
        raise typer.Exit(1)
    del cfg[f"mcpServers.{name}"]
    _save_mcp_cfg(cfg, cfg_path)
    rprint(f"[green]OK[/green] Removed [bold]{name}[/bold] from {cfg_path}")


@mcp_app.command("import")
def mcp_import(
    from_path: Annotated[Path, typer.Argument(help="Source JSON file with `mcpServers` envelope")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite entries that already exist")
    ] = False,
) -> None:
    """Copy `mcpServers` from FROM_PATH into the active registry."""
    src_path = from_path.expanduser().resolve()
    if not src_path.exists():
        rprint(f"[red]Error:[/red] Source file not found: {src_path}")
        raise typer.Exit(1)
    src_cfg = _load_mcp_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Source has no `mcpServers` to import:[/yellow] {src_path}")
        return
    dst_path = _resolve_mcp_path(scope, config)
    dst_cfg = _load_mcp_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_mcp_cfg(dst_cfg, dst_path)
    rprint(
        f"[green]OK[/green] Imported {len(added)} new, {len(overwritten)} overwritten, {len(skipped)} skipped ({src_path} → {dst_path})"
    )
    if added:
        rprint(f"  added:       {', '.join(added)}")
    if overwritten:
        rprint(f"  overwritten: {', '.join(overwritten)}")
    if skipped:
        rprint(f"  skipped:     {', '.join(skipped)}  [dim](pass --force to overwrite)[/dim]")


@mcp_app.command("export")
def mcp_export(
    to_path: Annotated[Path, typer.Argument(help="Destination JSON file")],
    scope: _ScopeOpt = "user",
    config: _ConfigOpt = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite entries that already exist")
    ] = False,
) -> None:
    """Copy the active `mcpServers` registry into TO_PATH."""
    src_path = _resolve_mcp_path(scope, config)
    src_cfg = _load_mcp_cfg(src_path)
    if not _server_names(src_cfg):
        rprint(f"[yellow]Active registry is empty:[/yellow] {src_path}")
        return
    dst_path = to_path.expanduser().resolve()
    dst_cfg = _load_mcp_cfg(dst_path)
    added, overwritten, skipped = _copy_servers(src_cfg, dst_cfg, force=force)
    _save_mcp_cfg(dst_cfg, dst_path)
    rprint(
        f"[green]OK[/green] Exported {len(added)} new, {len(overwritten)} overwritten, {len(skipped)} skipped ({src_path} → {dst_path})"
    )
    if added:
        rprint(f"  added:       {', '.join(added)}")
    if overwritten:
        rprint(f"  overwritten: {', '.join(overwritten)}")
    if skipped:
        rprint(f"  skipped:     {', '.join(skipped)}  [dim](pass --force to overwrite)[/dim]")
