"""Run-recovery CLI surfaces (bugs 2 + 4 aside, asset list, events section).

* Bug 2 (display side) — ``molexp runs info`` shows the Error + "Retry with"
  block only for runs whose *status* is actually failed/cancelled; a run that
  has since succeeded never advertises a stale error.
* Bug 4 — ``--fresh`` is only meaningful with ``--rerun``; alone it errors.
* Asset list — the default view scans every scope (workspace / project /
  experiment / run) with a Scope column, matching ``molexp context``'s count;
  ``--scope`` filters by scope kind.
* Event spine — ``runs info`` appends the run's recent workspace events when
  the workspace opted into the timeline.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace
from molexp.workspace.models import ErrorInfo, RunStatus

runner = CliRunner()


@pytest.fixture
def ws(tmp_path):
    return Workspace(root=tmp_path / "lab", name="lab")


@pytest.fixture
def run(ws):
    exp = ws.add_project("proj").add_experiment("exp")
    return exp.add_run(params={"seed": 0})


def _runs_info(ws, run):
    return runner.invoke(
        app,
        ["runs", "info", "proj", "exp", run.id, "-t", str(ws.root)],
    )


# ── Bug 2 display side: status-gated error block ─────────────────────────────


class TestRunsInfoErrorGating:
    def test_failed_run_shows_error_and_retry_hint(self, ws, run) -> None:
        run._update_metadata(
            error=ErrorInfo(type="RuntimeError", message="boom", timestamp=datetime.now())
        )
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        result = _runs_info(ws, run)
        assert result.exit_code == 0, result.output
        assert "RuntimeError" in result.output
        assert "boom" in result.output
        assert "--resume" in result.output

    def test_succeeded_run_never_shows_stale_error(self, ws, run) -> None:
        # A record written before the lifecycle learned to clear errors:
        # error present but the run has since succeeded.
        run._update_metadata(
            error=ErrorInfo(type="RuntimeError", message="boom", timestamp=datetime.now())
        )
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        result = _runs_info(ws, run)
        assert result.exit_code == 0, result.output
        assert "boom" not in result.output
        assert "Retry with" not in result.output


# ── Event spine: runs info shows recent events ───────────────────────────────


class TestRunsInfoEvents:
    def test_events_section_lists_appended_events(self, ws, run) -> None:
        from molexp.workspace.events import WorkspaceEventLog

        log = WorkspaceEventLog(ws.root)
        log.append("run.created", "test", refs=[run.id])
        log.append("run.failed", "test", refs=[run.id])

        result = _runs_info(ws, run)
        assert result.exit_code == 0, result.output
        assert "Recent events:" in result.output
        assert "run.failed" in result.output


# ── Bug 4: --fresh gating ────────────────────────────────────────────────────


class TestFreshFlagGating:
    def test_fresh_without_rerun_errors(self, tmp_path) -> None:
        script = tmp_path / "train.py"
        script.write_text("print('hi')\n")
        result = runner.invoke(app, ["run", str(script), "--local", "--fresh"])
        assert result.exit_code == 1
        assert "--fresh" in result.output
        assert "--rerun" in result.output


# ── Asset list: recursive across scopes ──────────────────────────────────────


class TestAssetListAllScopes:
    def test_lists_run_scoped_assets_by_default(self, ws, run) -> None:
        with run.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1})
        result = runner.invoke(app, ["asset", "list", "-t", str(ws.root)])
        assert result.exit_code == 0, result.output
        assert "metrics.json" in result.output
        assert "run:" in result.output  # scope column locates the asset

    def test_scope_filter_narrows(self, ws, run) -> None:
        with run.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1})
        result = runner.invoke(app, ["asset", "list", "--scope", "run", "-t", str(ws.root)])
        assert result.exit_code == 0, result.output
        assert "metrics.json" in result.output

    def test_unknown_scope_value_errors(self, ws) -> None:
        result = runner.invoke(app, ["asset", "list", "--scope", "bogus", "-t", str(ws.root)])
        assert result.exit_code == 1
        # rich may wrap the line; check the salient pieces.
        assert "unknown scope" in result.output
        assert "experiment" in result.output
