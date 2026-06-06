"""molexp CLI — flat command tree.

Top-level verbs (``run`` / ``serve`` / ``monitor`` / ``explore`` / ``info`` /
``exec`` / ``shell`` / ``sync`` / ``push`` / ``pull`` / ``init`` / ``agent``)
and noun groups (``project`` / ``experiment`` / ``runs`` / ``asset`` /
``target`` / ``session`` / ``config`` / ``mcp``) register directly on the app.
Each workspace-bound command takes a ``-t/--target`` option (default: cwd) via
:mod:`molexp.cli._target`. There is no ``workspace`` god-group.
"""

from __future__ import annotations

import typer
from mollog import get_logger

from molexp.cli._app import app

# ── init + agent (top-level command functions) ───────────────────────────────
from molexp.cli.init_cmd import init as _init_cmd

# ── Verbs — self-register on `app` via @app.command when imported ─────────────
from molexp.cli.workspace import catalog as _catalog
from molexp.cli.workspace import explore as _explore
from molexp.cli.workspace import lifecycle as _lifecycle
from molexp.cli.workspace import monitor as _monitor
from molexp.cli.workspace import run as _run
from molexp.cli.workspace import serve as _serve
from molexp.cli.workspace import sync as _sync

app.command(name="init")(_init_cmd)

from molexp.cli.agent_cmd import agent as _agent_cmd  # noqa: E402

app.command(name="agent")(_agent_cmd)

# ── Noun groups (resource CRUD) — flat at top level ──────────────────────────
from molexp.cli.prune import register as _register_prune  # noqa: E402
from molexp.cli.target_cmd import target_app  # noqa: E402
from molexp.cli.workspace.resources import (  # noqa: E402
    asset_app,
    experiment_app,
    mcp_app,
    project_app,
    run_app,
)

_register_prune(run_app)  # add `prune` to the runs group (list / info / prune / …)
app.add_typer(project_app, name="project")
app.add_typer(experiment_app, name="experiment")
app.add_typer(run_app, name="runs")
app.add_typer(asset_app, name="asset")
app.add_typer(target_app, name="target")
app.add_typer(mcp_app, name="mcp")

# ── session + config groups ──────────────────────────────────────────────────
from molexp.cli.session_cmd import session_app  # noqa: E402

app.add_typer(session_app, name="session")

from molexp.cli.config_cmd import config_app  # noqa: E402

app.add_typer(config_app, name="config")

# ── Third-party CLI plugin discovery ─────────────────────────────────────────
_logger = get_logger(__name__)


def _register_third_party_cli_plugins(app: typer.Typer) -> None:
    from molexp.plugins import discover_cli_plugins

    for plugin in discover_cli_plugins():
        try:
            plugin.register(app)
        except Exception as exc:
            _logger.warning(f"plugin '{plugin.id}' register raised; skipping: {exc}")


_register_third_party_cli_plugins(app)
