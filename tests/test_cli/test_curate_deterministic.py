"""curate-unify-03 — the deterministic (LLM-free) `molexp curate` subcommands.

`move-run` / `delete-folder` / `rehome-asset` build a §8 ChangeProposal directly
and gate + execute it through the shared `run_curation_proposal` backend. Under
CliRunner (non-TTY) the InteractiveApprover auto-approves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path)
    ws.materialize()
    proj = ws.add_project("p")
    proj.add_experiment("source", workflow_source="t.py")
    proj.add_experiment("target", workflow_source="t.py")
    proj.get_experiment("source").add_run(id="subject")
    return ws


def test_move_run_relocates(runner: CliRunner, tmp_path: Path) -> None:
    """ac-004 — curate move-run relocates the run and reports executed."""
    _seed(tmp_path)
    result = runner.invoke(
        app,
        ["curate", "move-run", "--run", "subject", "--to", "target", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "executed" in result.output
    fresh = Workspace(tmp_path).get_project("p")
    assert not fresh.get_experiment("source").has_run("subject")
    assert fresh.get_experiment("target").has_run("subject")


def test_delete_folder_removes(runner: CliRunner, tmp_path: Path) -> None:
    """ac-005 — curate delete-folder deletes the target."""
    _seed(tmp_path)
    result = runner.invoke(
        app,
        ["curate", "delete-folder", "--folder", "subject", "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "executed" in result.output
    assert not Workspace(tmp_path).get_project("p").get_experiment("source").has_run("subject")


def test_rehome_asset_reimports(runner: CliRunner, tmp_path: Path) -> None:
    """ac-006 — curate rehome-asset re-homes the asset into the target scope."""
    from molexp.workspace.assets.scan import scan_assets

    ws = _seed(tmp_path)
    payload = tmp_path / "a.txt"
    payload.write_text("payload-bytes")
    asset = ws.get_project("p").get_experiment("source").data_assets.import_asset("d", payload)

    result = runner.invoke(
        app,
        [
            "curate",
            "rehome-asset",
            "--asset",
            asset.asset_id,
            "--from",
            "source",
            "--to",
            "target",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "executed" in result.output
    target_dir = Workspace(tmp_path).get_project("p").get_experiment("target").experiment_dir
    assert scan_assets(target_dir), "the re-homed asset should be present under target"


def test_missing_flag_fails(runner: CliRunner, tmp_path: Path) -> None:
    """A required flag missing for the op exits non-zero."""
    _seed(tmp_path)
    result = runner.invoke(
        app, ["curate", "move-run", "--run", "subject", "--workspace", str(tmp_path)]
    )
    assert result.exit_code != 0
