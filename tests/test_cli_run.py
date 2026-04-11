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
        assert run.status == "succeeded"
        assert run_json["context"]["results"]["mode"] == "dry"

    def test_dry_run_and_slurm_are_mutually_exclusive(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("x = 1\n")

        result = runner.invoke(app, ["run", str(script), "--dry-run", "--slurm"])

        assert result.exit_code == 1
        assert "--dry-run and --slurm are mutually exclusive" in result.output
