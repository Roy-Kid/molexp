"""CLI tests for `molexp runs cancel`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace
from molexp.workspace.run import RunStatus

runner = CliRunner()


def _make_workspace(tmp_path, status="pending", molq_job_id=None, slurm_job_id=None):
    """Create a minimal workspace/project/experiment/run for testing."""
    ws_path = tmp_path / "workspace"
    ws = Workspace(root=ws_path, name="test-ws")
    project = ws.project("proj1")
    exp = project.experiment("exp1")
    run = exp.run(parameters={"lr": 0.001})
    if status != "pending":
        run._set_status(RunStatus(status))
    if molq_job_id or slurm_job_id:
        executor_info: dict[str, str] = {}
        if molq_job_id:
            executor_info["job_id"] = molq_job_id
        if slurm_job_id:
            executor_info["scheduler_job_id"] = slurm_job_id
        run._update_metadata(executor_info=executor_info)
    return ws_path, project, exp, run


class TestRunsCancelExperimentScope:
    def test_cancel_all_pending_runs(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--all", "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_by_status_filter(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path, status="running")

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--status", "running", "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_skips_terminal_runs(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path, status="succeeded")

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--all", "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        # In experiment-scope --all mode, terminal runs are excluded before selection,
        # so the message is "nothing to cancel" rather than "already terminal".
        assert "nothing to cancel" in result.output or "terminal" in result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "succeeded"

    def test_cancel_no_matching_runs(self, tmp_path):
        ws_path, project, exp, _ = _make_workspace(tmp_path, status="succeeded")

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--status", "running",
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        assert "No runs matched" in result.output

    def test_cancel_requires_all_or_status(self, tmp_path):
        ws_path, project, exp, _ = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 1
        assert "--all or --status" in result.output

    def test_cancel_requires_both_project_and_experiment(self, tmp_path):
        ws_path, project, _, _ = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--all", "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 1
        assert "--project and --experiment" in result.output

    def test_cancel_aborted_by_user(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--all",
            "--path", str(ws_path),
        ], input="N\n")

        assert result.exit_code == 0
        assert "Aborted" in result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "pending"

    def test_cancel_shows_confirmation_table(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel",
            "--project", project.id,
            "--experiment", exp.id,
            "--all", "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        assert "Runs to cancel" in result.output


class TestRunsCancelByRunId:
    def test_cancel_by_explicit_run_id(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_by_run_id_skips_terminal(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(tmp_path, status="succeeded")

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        assert "terminal" in result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "succeeded"

    def test_cancel_unknown_run_id_warns(self, tmp_path):
        ws_path, _, _, _ = _make_workspace(tmp_path)

        result = runner.invoke(app, [
            "runs", "cancel", "nonexistent-run-id",
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        assert "not found" in result.output.lower()


class TestRunsCancelMolqIntegration:
    def test_cancel_calls_molq_when_job_id_present(self, tmp_path, mocker):
        ws_path, project, exp, run = _make_workspace(
            tmp_path, molq_job_id="molq-uuid-1234"
        )

        mock_submitor = mocker.MagicMock()
        mock_molq = mocker.MagicMock()
        mock_molq.Submitor = mocker.MagicMock(return_value=mock_submitor)
        mocker.patch.dict("sys.modules", {"molq": mock_molq})

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes", "--scheduler", "slurm",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0, result.output
        mock_molq.Submitor.assert_called_once_with(cluster_name="default", scheduler="slurm")
        mock_submitor.cancel.assert_called_once_with("molq-uuid-1234")

    def test_cancel_updates_workspace_when_molq_fails(self, tmp_path, mocker):
        ws_path, project, exp, run = _make_workspace(
            tmp_path, molq_job_id="molq-uuid-5678"
        )

        mock_submitor = mocker.MagicMock()
        mock_submitor.cancel.side_effect = RuntimeError("scancel failed")
        mock_molq = mocker.MagicMock()
        mock_molq.Submitor = mocker.MagicMock(return_value=mock_submitor)
        mocker.patch.dict("sys.modules", {"molq": mock_molq})

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        assert "Warning" in result.output
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_warns_when_no_molq_job_id(self, tmp_path, mocker):
        ws_path, project, exp, run = _make_workspace(tmp_path)

        mock_submitor = mocker.MagicMock()
        mock_molq = mocker.MagicMock()
        mock_molq.Submitor = mocker.MagicMock(return_value=mock_submitor)
        mocker.patch.dict("sys.modules", {"molq": mock_molq})

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        assert "no molq job metadata" in result.output
        mock_submitor.cancel.assert_not_called()
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_local_scheduler_skips_molq(self, tmp_path):
        ws_path, project, exp, run = _make_workspace(
            tmp_path, molq_job_id="molq-uuid-local"
        )

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes", "--scheduler", "local",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        reloaded = Workspace.load(ws_path).get_project(project.id).get_experiment(exp.id).get_run(run.id)
        assert reloaded.status == "cancelled"

    def test_cancel_uses_custom_cluster_name(self, tmp_path, mocker):
        ws_path, project, exp, run = _make_workspace(
            tmp_path, molq_job_id="molq-uuid-custom"
        )

        mock_submitor = mocker.MagicMock()
        mock_molq = mocker.MagicMock()
        mock_molq.Submitor = mocker.MagicMock(return_value=mock_submitor)
        mocker.patch.dict("sys.modules", {"molq": mock_molq})

        result = runner.invoke(app, [
            "runs", "cancel", run.id,
            "--yes", "--scheduler", "slurm", "--cluster", "alvis",
            "--path", str(ws_path),
        ])

        assert result.exit_code == 0
        mock_molq.Submitor.assert_called_once_with(cluster_name="alvis", scheduler="slurm")
