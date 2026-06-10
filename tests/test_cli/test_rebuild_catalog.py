"""Regression tests for ``molexp rebuild-catalog`` (ac-006 / P1-1).

The derived catalog (``<root>/catalog/index.sqlite``) is rebuilt from the
entity ``*.json`` + manifests, which are the single source of truth. Before
this command, ``AssetCatalog.rebuild()`` had zero production callers, so the
self-heal path the derived catalog depends on was unreachable.
"""

from __future__ import annotations

import shutil

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed(root) -> Workspace:
    ws = Workspace(root=root, name="lab")
    proj = ws.add_project("demo")
    exp = proj.add_experiment("baseline")
    for i in range(2):
        r = exp.add_run(params={"seed": i})
        with r.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1 * i})
    return ws


@pytest.mark.integration
def test_rebuild_catalog_command_is_registered(runner: CliRunner) -> None:
    result = runner.invoke(app, ["rebuild-catalog", "--help"])
    assert result.exit_code == 0, result.stdout
    assert "catalog" in result.stdout.lower()


@pytest.mark.integration
def test_rebuild_catalog_restores_deleted_catalog(runner: CliRunner, tmp_path) -> None:
    """Deleting the derived catalog then running the command restores a
    query-equivalent catalog (exit 0)."""
    ws = _seed(tmp_path / "lab")
    before_runs = {r["run_id"] for r in ws.catalog.query_runs()}
    before_assets = {a.asset_id for a in ws.catalog.query_assets()}
    assert len(before_runs) == 2

    shutil.rmtree(tmp_path / "lab" / "catalog")

    result = runner.invoke(app, ["rebuild-catalog", "--workspace", str(tmp_path / "lab")])
    assert result.exit_code == 0, result.stdout
    assert "Rebuilt catalog" in result.stdout

    fresh = Workspace(tmp_path / "lab")
    assert {r["run_id"] for r in fresh.catalog.query_runs()} == before_runs
    assert {a.asset_id for a in fresh.catalog.query_assets()} == before_assets
