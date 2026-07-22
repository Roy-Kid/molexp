"""CLI tests for `molexp run` --config / --profile behavior."""

from __future__ import annotations

import re

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
    body="return config.get('epochs', 'default')",
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
                "def train(inputs, config):",
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
    body="return config.get('epochs', 'default')",
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
                "def train(inputs, config):",
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

        # Pure contract: the promoted fn RETURNS its result; the engine persists
        # it as the task's workflow output (no RunContext.set_result).
        from molexp.workflow import read_node_outputs

        exec_id = run.execution_history[-1].execution_id
        outputs = read_node_outputs(run.run_dir, exec_id)
        assert outputs["train"] == 1

    def test_resume_replays_non_succeeded_runs_of_same_profile(self, tmp_path):
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(
            script,
            workspace_root,
            body=(
                "import pathlib\n"
                "    marker = pathlib.Path(inputs['workdir']) / 'fail_once'\n"
                "    if not marker.exists():\n"
                "        marker.touch()\n"
                "        raise RuntimeError('boom')\n"
                "    return config['epochs']"
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
        _write_script(script, workspace_root, body="return 'wet'")

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
