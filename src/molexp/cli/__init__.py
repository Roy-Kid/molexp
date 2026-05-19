"""Command-line interface for molexp.

The CLI is organised around ``molexp workspace <TARGET>`` as the primary
entry point.  TARGET unifies local and remote workspaces into one concept::

    molexp workspace . info              # local (cwd)
    molexp workspace user@host:/path info  # remote (SCP-style)
    molexp workspace @target-name info     # registered compute target

Session and config management are top-level::

    molexp session list
    molexp config set key value
"""

from __future__ import annotations

import typer
from mollog import get_logger

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)

# ── Top-level ergonomic shortcuts ─────────────────────────────────────────────
from molexp.cli.init_cmd import init as _init_cmd  # noqa: E402

app.command(name="init")(_init_cmd)

# ── Primary entry point: workspace ────────────────────────────────────────────
from molexp.cli.workspace import workspace_app  # noqa: E402

app.add_typer(workspace_app, name="workspace")

# ── Session management ────────────────────────────────────────────────────────
from molexp.cli.session_cmd import session_app  # noqa: E402

app.add_typer(session_app, name="session")

# ── Global config ─────────────────────────────────────────────────────────────
from molexp.cli.config_cmd import config_app  # noqa: E402

app.add_typer(config_app, name="config")

# ── Runs subcommands (currently: prune) ───────────────────────────────────────
from molexp.cli.prune import register as _register_prune  # noqa: E402

runs_app = typer.Typer(name="runs", help="Run-level operations (prune, ...).")
_register_prune(runs_app)
app.add_typer(runs_app, name="runs")

# ── Compute targets ───────────────────────────────────────────────────────────
from molexp.cli.target_cmd import target_app  # noqa: E402

app.add_typer(target_app, name="target")

# ── Third-party CLI plugin discovery ──────────────────────────────────────────

_logger = get_logger(__name__)


def _register_third_party_cli_plugins(app: typer.Typer) -> None:
    from molexp.plugins import discover_cli_plugins

    for plugin in discover_cli_plugins():
        try:
            plugin.register(app)
        except Exception as exc:
            _logger.warning(f"plugin '{plugin.id}' register raised; skipping: {exc}")


_register_third_party_cli_plugins(app)


if __name__ == "__main__":
    app()
