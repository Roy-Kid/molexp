"""Tests for ``molexp.workspace.curation.inventory`` tree scanning.

Pins ``scan_workspace``: counts are totals across the whole ``Workspace ->
Project -> Experiment -> Run`` tree; ``asset_count`` is sourced from scanning
the authoritative manifests (``scan.scan_assets``); the nested ``*Inventory``
models carry each run's id and lifecycle status.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.curation import scan_workspace


def _seed_two_run_tree(root: Path) -> Workspace:
    """One project / one experiment / two runs (one succeeded, one pending).

    The succeeded run saves a run-scoped artifact, so the workspace owns at
    least one manifest-persisted asset that ``scan.scan_assets`` counts.
    """
    ws = Workspace(root=root, name="Curation Lab")
    proj = ws.add_project("alpha")
    exp = proj.add_experiment("baseline", params={"lr": 1e-3})
    succeeded = exp.add_run(params={"seed": 0})
    with succeeded.start() as ctx:
        ctx.register_artifact({"loss": 0.1}, name="metrics.json")
    exp.add_run(params={"seed": 1})  # left pending — never started
    return ws


class TestScanWorkspace:
    def test_counts_are_tree_totals(self, tmp_path: Path) -> None:
        inv = scan_workspace(_seed_two_run_tree(tmp_path / "lab"))
        assert inv.project_count == 1
        assert inv.experiment_count == 1
        assert inv.run_count == 2

    def test_asset_count_matches_manifest_scan(self, tmp_path: Path) -> None:
        from molexp.workspace.assets import scan

        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        assert inv.asset_count == len(scan.scan_assets(ws.root))
        assert inv.asset_count >= 1  # the succeeded run's artifact is persisted

    def test_nested_inventory_carries_run_id_and_status(self, tmp_path: Path) -> None:
        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        runs = inv.projects[0].experiments[0].runs

        on_disk = {r.id for r in ws.list_projects()[0].list_experiments()[0].list_runs()}
        assert {r.id for r in runs} == on_disk
        assert sorted(r.status for r in runs) == ["pending", "succeeded"]

    def test_empty_workspace_yields_zeroed_inventory(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "empty", name="Empty Lab")
        ws.materialize()
        inv = scan_workspace(ws)
        assert inv.name == "Empty Lab"
        assert inv.projects == ()
        assert inv.project_count == 0
        assert inv.experiment_count == 0
        assert inv.run_count == 0
        assert inv.asset_count == 0
