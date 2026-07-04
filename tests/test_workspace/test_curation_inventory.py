"""RED tests for ``molexp.workspace.curation`` inventory scanning.

Pins ``scan_workspace`` and the frozen ``*Inventory`` pydantic models.

Until the ``molexp.workspace.curation`` package exists these tests fail at
collection with ``ModuleNotFoundError`` — the intended RED state. Once the
subpackage ships they pin the inventory contract:

* counts are totals across the whole ``Workspace → Project → Experiment →
  Run`` tree;
* ``asset_count`` is sourced from scanning the authoritative manifests
  (``scan.scan_assets`` — run-scoped artifacts/logs/checkpoints);
* every returned model is frozen.
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
        ctx.artifact.save("metrics.json", {"loss": 0.1})
    exp.add_run(params={"seed": 1})  # left pending — never started
    return ws


# ── Basics: tree-wide counts ─────────────────────────────────────────────────


class TestScanWorkspaceBasics:
    def test_counts_are_tree_totals(self, tmp_path: Path) -> None:
        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        assert inv.project_count == 1
        assert inv.experiment_count == 1
        assert inv.run_count == 2

    def test_asset_count_matches_scan(self, tmp_path: Path) -> None:
        """``asset_count`` equals the manifest scan total (authoritative source)."""
        from molexp.workspace.assets import scan

        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        assert inv.asset_count == len(scan.scan_assets(ws.root))
        assert inv.asset_count >= 1  # the succeeded run's artifact is persisted


# ── Basics: nested model shape ───────────────────────────────────────────────


class TestInventoryShape:
    def test_run_ids_match_on_disk_runs(self, tmp_path: Path) -> None:
        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        exp_inv = inv.projects[0].experiments[0]

        actual = {r.id for r in ws.list_projects()[0].list_experiments()[0].list_runs()}
        assert {r.id for r in exp_inv.runs} == actual

    def test_per_run_status_recorded(self, tmp_path: Path) -> None:
        ws = _seed_two_run_tree(tmp_path / "lab")
        inv = scan_workspace(ws)
        statuses = sorted(r.status for r in inv.projects[0].experiments[0].runs)
        assert statuses == ["pending", "succeeded"]


# ── Edge case: empty workspace ───────────────────────────────────────────────


class TestScanWorkspaceEdgeCases:
    def test_empty_workspace_yields_empty_inventory(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "empty", name="Empty Lab")
        ws.materialize()
        inv = scan_workspace(ws)
        assert inv.name == "Empty Lab"
        assert inv.projects == ()
        assert inv.project_count == 0
        assert inv.experiment_count == 0
        assert inv.run_count == 0
        assert inv.asset_count == 0
