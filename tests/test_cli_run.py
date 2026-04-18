"""CLI tests for `molexp run` --config / --profile behavior."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()


def _write_script(
    path,
    workspace_root,
    body="ctx.set_result('epochs', ctx.config.get('epochs', 'default'))",
):
    path.write_text(
        "\n".join(
            [
                "import molexp as me",
                "",
                f"ws = me.Workspace({str(workspace_root)!r})",
                "project = ws.project('demo')",
                "exp = project.experiment('train')",
                "",
                "def train(ctx: me.RunContext) -> None:",
                f"    {body}",
                "",
                "exp.set_workflow(train)",
                "me.entry(ws)",
                "",
            ]
        )
    )


def _write_molcfg(path):
    path.write_text(
        "defaults:\n"
        "  epochs: 100\n"
        "  dataset: md17\n"
        "profiles:\n"
        "  dry-run:\n"
        "    epochs: 1\n"
        "    skip_heavy: true\n"
        "  smoke:\n"
        "    epochs: 5\n"
    )


class TestRunCommand:
    def test_profile_executes_workflow_and_persists_metadata(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        result = runner.invoke(
            app,
            ["run", str(script), "--config", str(molcfg), "--profile", "dry-run"],
        )
        assert result.exit_code == 0, result.output

        ws = Workspace.load(workspace_root)
        runs = ws.get_project("demo").get_experiment("train").list_runs()
        assert len(runs) == 1

        run = runs[0]
        # profile name normalized
        assert run.metadata.profile == "dry_run"
        # defaults were merged into config
        assert run.metadata.config["epochs"] == 1
        assert run.metadata.config["dataset"] == "md17"
        # content hash present
        assert run.metadata.config_hash is not None
        # run succeeded (profile is orthogonal to status)
        assert run.status == "succeeded"

        run_json = json.loads((run.run_dir / "run.json").read_text())
        assert run_json["context"]["results"]["epochs"] == 1

    def test_resume_replays_non_succeeded_runs_of_same_profile(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(
            script,
            workspace_root,
            body=(
                "import pathlib, os\n"
                "    marker = pathlib.Path(ctx.run.run_dir) / 'fail_once'\n"
                "    if not marker.exists():\n"
                "        marker.touch()\n"
                "        raise RuntimeError('boom')\n"
                "    ctx.set_result('epochs', ctx.config['epochs'])"
            ),
        )

        # First run: fails
        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "smoke"]
        )
        ws = Workspace.load(workspace_root)
        runs = ws.get_project("demo").get_experiment("train").list_runs()
        assert len(runs) == 1
        assert runs[0].status == "failed"
        assert runs[0].metadata.profile == "smoke"

        # Resume: re-executes the failed run because profile matches
        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "--resume",
            ],
        )
        assert result.exit_code == 0, result.output

        ws = Workspace.load(workspace_root)
        runs = ws.get_project("demo").get_experiment("train").list_runs()
        assert len(runs) == 1
        assert runs[0].status == "succeeded"

    def test_succeeded_runs_are_skipped_by_default(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root, body="ctx.set_result('mode', 'wet')")

        # First run succeeds
        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "smoke"]
        )
        assert result.exit_code == 0, result.output

        # Second run: same profile — skipped because already succeeded
        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "smoke"]
        )
        assert result.exit_code == 0, result.output
        assert "skipped" in result.output

    def test_different_profiles_produce_different_runs(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "dry-run"]
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "smoke"]
        )
        assert result.exit_code == 0, result.output

        ws = Workspace.load(workspace_root)
        runs = ws.get_project("demo").get_experiment("train").list_runs()
        # Two distinct runs, one per profile
        assert len(runs) == 2
        profiles = {r.metadata.profile for r in runs}
        assert profiles == {"dry_run", "smoke"}

    def test_unknown_profile_reports_error(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        result = runner.invoke(
            app, ["run", str(script), "--config", str(molcfg), "--profile", "missing"]
        )
        assert result.exit_code == 1
        assert "Unknown profile" in result.output

    def test_profile_without_config_aborts(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(script, workspace_root)

        # No molcfg.yaml in CWD and no --config
        result = runner.invoke(
            app, ["run", str(script), "--profile", "dry-run"]
        )
        assert result.exit_code == 1
        assert "no config file" in result.output.lower()

    def test_run_help_shows_backends(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "local" in result.output

    def test_run_help_has_grouped_options(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.output
        assert "--scheduler" in result.output
        assert "--slurm" in result.output
        assert "--profile" in result.output
        assert "--config" in result.output
        assert "HPC Options" in result.output
