"""CLI tests for `molexp run` dry-run behavior."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()


def _write_script(path, workspace_root, body="ctx.set_result('mode', 'dry' if ctx.dry_run else 'wet')"):
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


class TestRunCommand:
    def test_dry_run_executes_workflow(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(
            script,
            workspace_root,
            body="ctx.set_result('mode', 'dry' if ctx.dry_run else 'wet')",
        )

        result = runner.invoke(app, ["run", str(script), "--dry-run"])

        assert result.exit_code == 0, result.output

        workspace = Workspace.load(workspace_root)
        project = workspace.get_project("demo")
        assert project is not None

        experiment = project.get_experiment("train")
        assert experiment is not None

        runs = experiment.list_runs()
        assert len(runs) == 1

        run = runs[0]
        run_json = json.loads((run.run_dir / "run.json").read_text())

        assert run.metadata.dry_run is True
        assert run.metadata.labels["mode"] == "dry-run"
        assert run.status == "dry_run"
        assert run_json["context"]["results"]["mode"] == "dry"

    def test_resume_executes_dry_run_runs(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(
            script,
            workspace_root,
            body="ctx.set_result('mode', 'dry' if ctx.dry_run else 'wet')",
        )

        result = runner.invoke(app, ["run", str(script), "--dry-run"])
        assert result.exit_code == 0, result.output

        workspace = Workspace.load(workspace_root)
        experiment = workspace.get_project("demo").get_experiment("train")
        runs = experiment.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "dry_run"

        result = runner.invoke(app, ["run", str(script), "--resume"])
        assert result.exit_code == 0, result.output

        workspace = Workspace.load(workspace_root)
        experiment = workspace.get_project("demo").get_experiment("train")
        runs = experiment.list_runs()
        assert len(runs) == 1

        run = runs[0]
        run_json = json.loads((run.run_dir / "run.json").read_text())

        assert run.status == "succeeded"
        assert run.metadata.dry_run is False
        assert run_json["context"]["results"]["mode"] == "wet"

    def test_resume_skips_non_dry_run_runs(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(script, workspace_root, body="ctx.set_result('mode', 'wet')")

        result = runner.invoke(app, ["run", str(script)])
        assert result.exit_code == 0, result.output

        result = runner.invoke(app, ["run", str(script), "--resume"])
        assert result.exit_code == 0, result.output
        assert "skipped" in result.output

    def test_normal_run_skips_dry_run_status(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(
            script,
            workspace_root,
            body="ctx.set_result('mode', 'dry' if ctx.dry_run else 'wet')",
        )

        result = runner.invoke(app, ["run", str(script), "--dry-run"])
        assert result.exit_code == 0, result.output

        result = runner.invoke(app, ["run", str(script)])
        assert result.exit_code == 0, result.output
        assert "dry_run, skipped" in result.output

    def test_resume_and_dry_run_are_mutually_exclusive(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("x = 1\n")

        result = runner.invoke(app, ["run", str(script), "--resume", "--dry-run"])

        assert result.exit_code == 1
        assert "--resume and --dry-run are mutually exclusive" in result.output

    def test_run_help_shows_backends(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "local" in result.output

    def test_run_help_has_grouped_options(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.output
        assert "--slurm" in result.output
        assert "--pbs" in result.output
        assert "--lsf" in result.output
        assert "HPC Options" in result.output
        assert "--partition" in result.output
        assert "--gpus" in result.output
        assert "--mem" in result.output
