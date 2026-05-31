"""Shared top-level Typer app instance.

Defined in its own module so command modules can register on it via
``@app.command(...)`` without importing :mod:`molexp.cli` (which imports them
back — a cycle). :mod:`molexp.cli` assembles the final flat command tree.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)
