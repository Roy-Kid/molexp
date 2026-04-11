"""CLI tests for `molexp run` dry-run behavior."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()


class TestRunCommand:
    def test_dry_run_executes_workflow(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        script.write_text(
            "\n".join(
                [
                    "import molexp as me",
                    "",
                    f"project = me.Project('demo', config={{'workspace_root': {str(workspace_root)!r}}})",
                    "exp = project.experiment('train')",
                    "",
                    "def train(ctx: me.RunContext) -> None:",
                    "    if ctx.dry_run:",
                    "        ctx.set_result('mode', 'dry')",
                    "        return",
                    "    ctx.set_result('mode', 'wet')",
                    "",
                    "exp.set_workflow(train)",
                    "me.entry(project)",
                    "",
                ]
            )
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
        script.write_text(
            "\n".join(
                [
                    "import molexp as me",
                    "",
                    f"project = me.Project('demo', config={{'workspace_root': {str(workspace_root)!r}}})",
                    "exp = project.experiment('train')",
                    "",
                    "def train(ctx: me.RunContext) -> None:",
                    "    if ctx.dry_run:",
                    "        ctx.set_result('mode', 'dry')",
                    "        return",
                    "    ctx.set_result('mode', 'wet')",
                    "",
                    "exp.set_workflow(train)",
                    "me.entry(project)",
                    "",
                ]
            )
        )

        # Step 1: dry-run
        result = runner.invoke(app, ["run", str(script), "--dry-run"])
        assert result.exit_code == 0, result.output

        workspace = Workspace.load(workspace_root)
        project = workspace.get_project("demo")
        experiment = project.get_experiment("train")
        runs = experiment.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "dry_run"

        # Step 2: resume
        result = runner.invoke(app, ["run", str(script), "--resume"])
        assert result.exit_code == 0, result.output

        workspace = Workspace.load(workspace_root)
        project = workspace.get_project("demo")
        experiment = project.get_experiment("train")
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
        script.write_text(
            "\n".join(
                [
                    "import molexp as me",
                    "",
                    f"project = me.Project('demo', config={{'workspace_root': {str(workspace_root)!r}}})",
                    "exp = project.experiment('train')",
                    "",
                    "def train(ctx: me.RunContext) -> None:",
                    "    ctx.set_result('mode', 'wet')",
                    "",
                    "exp.set_workflow(train)",
                    "me.entry(project)",
                    "",
                ]
            )
        )

        # Normal run first
        result = runner.invoke(app, ["run", str(script)])
        assert result.exit_code == 0, result.output

        # Resume should skip succeeded runs
        result = runner.invoke(app, ["run", str(script), "--resume"])
        assert result.exit_code == 0, result.output
        assert "skipped" in result.output

    def test_normal_run_skips_dry_run_status(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        script.write_text(
            "\n".join(
                [
                    "import molexp as me",
                    "",
                    f"project = me.Project('demo', config={{'workspace_root': {str(workspace_root)!r}}})",
                    "exp = project.experiment('train')",
                    "",
                    "def train(ctx: me.RunContext) -> None:",
                    "    ctx.set_result('mode', 'dry' if ctx.dry_run else 'wet')",
                    "",
                    "exp.set_workflow(train)",
                    "me.entry(project)",
                    "",
                ]
            )
        )

        # Dry-run first
        result = runner.invoke(app, ["run", str(script), "--dry-run"])
        assert result.exit_code == 0, result.output

        # Normal run should skip dry_run runs
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
        # Remote backends appear when molq is installed
        # (may or may not be present in test env)

    def test_run_local_help_has_no_resource_options(self):
        result = runner.invoke(app, ["run", "local", "--help"])
        assert result.exit_code == 0
        assert "--partition" not in result.output
        assert "--gpus" not in result.output
        assert "--mem" not in result.output
