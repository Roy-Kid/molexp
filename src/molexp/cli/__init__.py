"""Command-line interface for molexp.

The ``app`` Typer instance is created here; individual command modules
are imported below for their registration side effects. After built-in
commands are wired up, third-party plugins discovered through the
``molexp.cli_plugins`` entry-point group also get a chance to attach
subcommands via :func:`molexp.plugins.discover_cli_plugins`.
"""

from __future__ import annotations

import typer
from mollog import get_logger

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)

# Order matters only for --help display. Import for side-effect registration.
from molexp.cli import (
    asset,
    experiment,
    explore_cmd,
    mcp_cmd,
    project,
    run_cmd,
    runs,
    serve_cmd,
    target_cmd,
    watch_cmd,
    workspace_cmd,
)

_logger = get_logger(__name__)


def _register_third_party_cli_plugins(app: typer.Typer) -> None:
    """Let every discovered third-party plugin attach its CLI commands.

    Failure isolation: a single plugin raising during ``register``
    is logged and skipped — the rest of the CLI must still boot.
    """
    from molexp.plugins import discover_cli_plugins

    for plugin in discover_cli_plugins():
        try:
            plugin.register(app)
        except Exception as exc:
            _logger.warning(f"plugin '{plugin.id}' register raised; skipping: {exc}")


_register_third_party_cli_plugins(app)


if __name__ == "__main__":
    app()
