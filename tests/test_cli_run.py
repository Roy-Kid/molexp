"""CLI tests for `molexp run` --config / --profile behavior."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    """Strip ANSI escape codes — Rich may insert style codes around flag
    tokens, breaking literal substring matches like `'--local'`."""
    return _ANSI_RE.sub("", s)


def _write_script(
    path,
    workspace_root,
    body="ctx.set_result('epochs', ctx.config.get('epochs', 'default'))",
):
    path.write_text(
        "\n".join(
            [
                "import molexp as me",
                "from molexp.workflow import default_binding_registry, promote_callable",
                "",
                f"ws = me.Workspace({str(workspace_root)!r})",
                "project = ws.add_project('demo')",
                "exp = project.add_experiment('train')",
                "",
                "def train(ctx: me.RunContext) -> None:",
                f"    {body}",
                "",
                "default_binding_registry.bind(exp, promote_callable(train, name='train'))",
                "me.entry(ws)",
                "",
            ]
        )
    )


def _write_rootless_script(
    path,
    body="ctx.set_result('epochs', ctx.config.get('epochs', 'default'))",
):
    """Variant of :func:`_write_script` that omits the workspace root.

    The script constructs ``Workspace(name=...)`` with NO root argument, so
    the framework must infer the root (CLI override > script dir > cwd).
    """
    path.write_text(
        "\n".join(
            [
                "import molexp as me",
                "from molexp.workflow import default_binding_registry, promote_callable",
                "",
                "ws = me.Workspace(name='electrolyte')",
                "project = ws.add_project('demo')",
                "exp = project.add_experiment('train')",
                "",
                "def train(ctx: me.RunContext) -> None:",
                f"    {body}",
                "",
                "default_binding_registry.bind(exp, promote_callable(train, name='train'))",
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
    def test_run_ignores_orphan_project_on_disk(self, tmp_path):
        """`molexp run` must not fail when the workspace has project/experiment
        dirs from prior scripts that the current script does not register.
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        # Seed an orphan project with an orphan experiment, both on disk only.
        orphan_exp = workspace_root / "projects" / "orphan-proj" / "experiments" / "orphan-exp"
        orphan_exp.mkdir(parents=True)
        (orphan_exp.parent.parent / "project.json").write_text(
            '{"id":"orphan-proj","name":"orphan","description":"","owner":"",'
            '"tags":[],"config":{},"created_at":"2026-04-21T12:00:00"}'
        )
        (orphan_exp / "experiment.json").write_text(
            '{"id":"orphan-exp","name":"orphan","parameter_space":{},'
            '"n_replicas":1,"seeds":[],"workflow_source":null,'
            '"workflow_type":null,"git_commit":null,"description":"",'
            '"tags":[],"created_at":"2026-04-21T12:00:00"}'
        )

        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "no workflow" not in result.output

    def test_profile_executes_workflow_and_persists_metadata(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
                "-t",
                str(workspace_root),
            ],
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

        run_json = json.loads(Path(run.run_dir / "run.json").read_text())
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
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
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
                "-t",
                str(workspace_root),
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
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output

        # Second run: same profile — skipped because already succeeded
        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
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
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
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
            app,
            [
                "run",
                str(script),
                "--config",
                str(molcfg),
                "--profile",
                "missing",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 1
        assert "Unknown profile" in result.output

    def test_profile_without_config_aborts(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_script(script, workspace_root)

        # No molcfg.yaml in CWD and no --config
        result = runner.invoke(
            app, ["run", str(script), "--profile", "dry-run", "-t", str(workspace_root)]
        )
        assert result.exit_code == 1
        assert "no config file" in result.output.lower()


class TestRootInferencePrecedence:
    """ac-006 / ac-007 / ac-008 — end-to-end root resolution precedence."""

    def test_no_flag_materializes_under_script_dir(self, tmp_path):
        # ac-006: no workspace flag -> root inferred to the SCRIPT's directory,
        # not cwd. Put the script in a dir distinct from any cwd default.
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script = script_dir / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_rootless_script(script)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        # Workspace materializes under the script's own directory.
        assert (script_dir / "workspace.json").exists()
        assert (script_dir / "projects").exists()

    def test_explicit_flag_overrides_script_dir(self, tmp_path):
        # ac-007: explicit -ws <override_dir> wins over the script directory.
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        override_dir = tmp_path / "override"
        override_dir.mkdir()
        script = script_dir / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_rootless_script(script)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
                "-ws",
                str(override_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (override_dir / "workspace.json").exists()
        # Script dir must NOT have been materialized.
        assert not (script_dir / "workspace.json").exists()

    def test_explicit_root_in_script_not_rewritten(self, tmp_path):
        # ac-008: a script hardcoding Workspace(<explicit_root>, name=...) run
        # with no flag stays at <explicit_root>, never silently rewritten.
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        explicit_root = tmp_path / "explicit"
        script = script_dir / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, explicit_root)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--config",
                str(molcfg),
                "--profile",
                "dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (explicit_root / "workspace.json").exists()
        # Neither the script dir got materialized as a workspace.
        assert not (script_dir / "workspace.json").exists()

    def test_run_help_shows_backends(self):
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "local" in plain

    def test_run_help_has_grouped_options(self):
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--local" in plain
        assert "--scheduler" in plain
        assert "--profile" in plain
        assert "--config" in plain
        assert "HPC Options" in plain
