"""Example third-party molexp plugin used by the test suite.

The shape mirrors what a real downstream package would publish through
the **two independent** entry-point channels introduced in spec 07:

* ``cli_plugin: CliPlugin`` — a Typer command appended to the molexp CLI
  (entry-point group ``molexp.cli_plugins``).
* ``bundle_dir: () -> Path`` — a directory of pre-built ESM assets
  served at ``/api/plugins/example/`` (entry-point group
  ``molexp.ui_plugins``). The directory contains ``manifest.json`` +
  ``index.js``; UI semantics live entirely in the manifest.

A real third-party package may contribute either or both. This fixture
contributes both to validate the "independent but same-package" form.

Tests inject this module via a fake ``importlib.metadata.entry_points``
return value rather than relying on ``pip install``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from molexp.plugins.cli import CliPlugin

# Unique markers so DOM / stdout assertions can grep for the fixture.
HELLO_MARKER = "molexp-example-plugin-hello"
RENDERER_MARKER = "molexp-example-plugin-renderer"


def _hello(name: str = typer.Option("world", help="Who to greet.")) -> None:
    """Print a marker line so CliRunner can assert the command ran."""
    typer.echo(f"{HELLO_MARKER}:{name}")


def _register_cli(app: typer.Typer) -> None:
    """Attach the plugin's ``hello`` command to the molexp Typer app."""
    app.command(name="hello", help="Example third-party plugin command.")(_hello)


cli_plugin = CliPlugin(
    id="example",
    name="Example Plugin",
    version="0.0.1",
    register=_register_cli,
)


def bundle_dir() -> Path:
    """Return the directory containing the plugin's UI bundle.

    The directory ships ``manifest.json`` (UI semantics) and
    ``index.js`` (the ESM entry that the browser dynamic-imports).
    """
    return Path(__file__).parent / "ui_dist"


__all__ = [
    "HELLO_MARKER",
    "RENDERER_MARKER",
    "bundle_dir",
    "cli_plugin",
]
