"""Tests covering third-party CLI command registration via the new
``CliPlugin`` contract.

Asserts that ``molexp.cli`` queries ``discover_cli_plugins()`` at boot,
calls each plugin's ``register(app)``, and isolates failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

# Make the example_plugin fixture importable as a top-level module so
# the entry-point resolution path matches what a real installed package
# would look like.
_FIXTURE_ROOT = Path(__file__).parent / "_fixtures"
if str(_FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIXTURE_ROOT))

from molexp.plugins.cli import CliPlugin, _discover_cli_uncached


class _FakeEntryPoint:
    def __init__(self, name: str, plugin: CliPlugin) -> None:
        self.name = name
        self.group = "molexp.cli_plugins"
        self._plugin = plugin

    def load(self) -> CliPlugin:
        return self._plugin


def _install_fake_eps(monkeypatch: pytest.MonkeyPatch, plugins: list[CliPlugin]) -> None:
    """Wire up a fake ``entry_points()`` returning the given CliPlugins."""
    fakes = [_FakeEntryPoint(f"fake-{idx}", plug) for idx, plug in enumerate(plugins)]

    class _FakeEntryPoints:
        def select(self, *, group: str):
            return tuple(ep for ep in fakes if ep.group == group)

    monkeypatch.setattr(
        "molexp.plugins.cli.importlib_metadata.entry_points",
        lambda: _FakeEntryPoints(),
    )
    _discover_cli_uncached.cache_clear()


def _reload_cli() -> object:
    """Force a fresh import of ``molexp.cli`` so registration re-runs."""
    for mod_name in list(sys.modules):
        if mod_name == "molexp.cli" or mod_name.startswith("molexp.cli."):
            sys.modules.pop(mod_name)
    import molexp.cli as cli_module

    return cli_module


def test_third_party_command_runs(monkeypatch: pytest.MonkeyPatch):
    """The example fixture's ``hello`` command appears under ``molexp``."""
    from example_plugin import HELLO_MARKER, cli_plugin  # type: ignore[attr-defined]

    _install_fake_eps(monkeypatch, [cli_plugin])

    cli_module = _reload_cli()

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["hello", "--name", "alice"])
    assert result.exit_code == 0, result.output
    assert HELLO_MARKER in result.output
    assert "alice" in result.output

    _discover_cli_uncached.cache_clear()


def test_failing_register_does_not_break_help(monkeypatch: pytest.MonkeyPatch):
    """A plugin whose ``register`` raises is skipped without aborting
    the rest of CLI startup; ``molexp --help`` still exits 0.
    """

    def boom(app: typer.Typer) -> None:
        raise RuntimeError("plugin sabotage")

    bad = CliPlugin(id="bad", name="Bad", version="0.0.1", register=boom)
    _install_fake_eps(monkeypatch, [bad])

    cli_module = _reload_cli()

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0, result.output

    _discover_cli_uncached.cache_clear()


def test_other_plugins_still_register_when_one_fails(monkeypatch: pytest.MonkeyPatch):
    """If plugin A's register raises, plugin B's register still runs and its
    command still routes."""

    def boom(app: typer.Typer) -> None:
        raise RuntimeError("plugin sabotage")

    def reg_good(app: typer.Typer) -> None:
        @app.command("survive")
        def _survive() -> None:
            typer.echo("still-alive")

    bad = CliPlugin(id="bad", name="Bad", version="0.0.1", register=boom)
    good = CliPlugin(id="good", name="Good", version="0.0.1", register=reg_good)
    _install_fake_eps(monkeypatch, [bad, good])

    cli_module = _reload_cli()

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["survive"])
    assert result.exit_code == 0, result.output
    assert "still-alive" in result.output

    _discover_cli_uncached.cache_clear()
