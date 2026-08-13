"""``molexp validate`` — the CLI face of the workspace conformance check.

The exit code is the contract: a violating tree must fail the process so the
command drops into a pre-commit hook or a CI step unchanged.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path, name="lab")
    ws.materialize()
    ws.add_project("alpha").add_experiment("sweep").add_run(params={"t": 1})
    return ws


def test_validate_reports_conforming_tree(tmp_path: Path) -> None:
    _workspace(tmp_path)
    result = CliRunner().invoke(app, ["validate", "-ws", str(tmp_path)])
    # A never-executed run warns (no _ops sidecar) but the tree still conforms.
    assert result.exit_code == 0, result.output
    assert "0 error(s)" in result.output


def test_validate_exits_nonzero_on_a_violation(tmp_path: Path) -> None:
    _workspace(tmp_path)
    (tmp_path / "leftover-output").mkdir()

    result = CliRunner().invoke(app, ["validate", "-ws", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "layout.stray" in result.output
    assert "leftover-output" in result.output


def test_strict_promotes_warnings_to_failure(tmp_path: Path) -> None:
    _workspace(tmp_path)
    result = CliRunner().invoke(app, ["validate", "-ws", str(tmp_path), "--strict"])
    assert result.exit_code == 1, result.output
    assert "run.ops" in result.output


def test_validate_json_emits_full_report(tmp_path: Path) -> None:
    import json

    _workspace(tmp_path)
    (tmp_path / "stray-dir").mkdir()
    result = CliRunner().invoke(app, ["validate", "-ws", str(tmp_path), "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error_count"] >= 1
    assert any(v["rule"] == "layout.stray" for v in payload["violations"])
    assert all("hint" in v and v["hint"] for v in payload["violations"])
    assert payload["next_actions"]
