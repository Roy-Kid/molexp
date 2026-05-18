"""``molexp config`` — global molexp configuration."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Annotated

import typer

from molexp.cli._common import rprint

config_app = typer.Typer(
    name="config",
    help="Manage global molexp configuration.",
    no_args_is_help=True,
)

_CONFIG_PATH = Path.home() / ".molexp" / "config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import os

    tmp = _CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.chmod(tmp, 0o600)  # noqa: PTH101
    tmp.replace(_CONFIG_PATH)


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = _load_config()
    if not cfg:
        rprint(f"[dim]No configuration set.[/dim] ({_CONFIG_PATH} - config path)")
        return

    rprint(f"[bold]Config:[/bold] {_CONFIG_PATH}")
    from rich import print as _rprint

    _rprint(json.dumps(cfg, indent=2))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (dot-notation supported).")],
    value: Annotated[str, typer.Argument(help="Value to set.")],
) -> None:
    """Set a configuration value.

    Example: molexp config set defaults.shell bash
    """
    cfg = _load_config()
    # Coerce value
    coerced: str | int | float | bool = value
    if value.lower() == "true":
        coerced = True
    elif value.lower() == "false":
        coerced = False
    else:
        try:
            coerced = int(value)
        except ValueError:
            with contextlib.suppress(ValueError):
                coerced = float(value)

    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = coerced
    _save_config(cfg)
    rprint(f"[green]OK[/green] Set {key} = {coerced!r}")


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="Config key to remove (dot-notation).")],
) -> None:
    """Remove a configuration value."""
    cfg = _load_config()
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            rprint(f"[yellow]Key not found:[/yellow] {key}")
            return
        node = node[part]
    if parts[-1] not in node:
        rprint(f"[yellow]Key not found:[/yellow] {key}")
        return
    del node[parts[-1]]
    _save_config(cfg)
    rprint(f"[green]OK[/green] Removed {key}")
