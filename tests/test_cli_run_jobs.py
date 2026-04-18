"""CLI tests for ``molexp run -j/--jobs N`` — local sweep-level parallelism.

Phase-1 surface (see ``docs/spec/unified-pydantic-graph-dispatch.md``).
"""

from __future__ import annotations

import time

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()


def _write_parallel_script(path, workspace_root, n_experiments: int, sleep_s: float) -> None:
    """Write a script with *n_experiments* each sleeping *sleep_s* seconds."""
    path.write_text(
        "\n".join(
            [
                "import time",
                "import molexp as me",
                "",
                f"ws = me.Workspace({str(workspace_root)!r})",
                "project = ws.project('demo')",
                "",
                "def sleeper(ctx):",
                f"    time.sleep({sleep_s})",
                "    ctx.set_result('done', True)",
                "",
                f"for i in range({n_experiments}):",
                "    exp = project.experiment(f'exp{i}')",
                "    exp.set_workflow(sleeper)",
                "",
                "me.entry(ws)",
                "",
            ]
        )
    )


class TestJobsFlag:
    def test_help_lists_jobs_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--jobs" in result.output or "-j" in result.output

    def test_jobs_gt_1_runs_experiments_in_parallel(self, tmp_path):
        """Three experiments × 0.3s should finish in roughly 0.3s with ``-j 3``."""
        sleep_s = 0.3
        n = 3
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_parallel_script(script, workspace_root, n_experiments=n, sleep_s=sleep_s)

        t0 = time.perf_counter()
        result = runner.invoke(app, ["run", str(script), "--local", "-j", str(n)])
        wall = time.perf_counter() - t0

        assert result.exit_code == 0, result.output
        # Parallel wall time should be < 2× single-experiment time.
        # Sequential would be n * sleep_s = 0.9s; parallel ~= sleep_s + overhead.
        assert wall < 2 * sleep_s * n * 0.7, (
            f"Expected parallel execution (~{sleep_s}s), got {wall:.2f}s — "
            "probably fell back to sequential."
        )

        ws = Workspace.load(workspace_root)
        all_runs = [
            run
            for exp in ws.get_project("demo").list_experiments()
            for run in exp.list_runs()
        ]
        assert len(all_runs) == n
        for run in all_runs:
            assert run.status == "succeeded"

    def test_jobs_1_is_sequential(self, tmp_path):
        """Three experiments × 0.3s with ``-j 1`` must take ~0.9s (strict serial)."""
        sleep_s = 0.3
        n = 3
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_parallel_script(script, workspace_root, n_experiments=n, sleep_s=sleep_s)

        t0 = time.perf_counter()
        result = runner.invoke(app, ["run", str(script), "--local", "-j", "1"])
        wall = time.perf_counter() - t0

        assert result.exit_code == 0, result.output
        # With n experiments × sleep_s each, serial wall time >= n * sleep_s * 0.9.
        assert wall >= n * sleep_s * 0.9, (
            f"Expected serial execution (~{n * sleep_s}s), got {wall:.2f}s — "
            "concurrency leaked through."
        )

    def test_jobs_default_is_sequential(self, tmp_path):
        """No ``-j`` flag → jobs=1 (backwards compatible with current behaviour)."""
        sleep_s = 0.2
        n = 3
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        _write_parallel_script(script, workspace_root, n_experiments=n, sleep_s=sleep_s)

        t0 = time.perf_counter()
        result = runner.invoke(app, ["run", str(script), "--local"])
        wall = time.perf_counter() - t0

        assert result.exit_code == 0, result.output
        assert wall >= n * sleep_s * 0.9, (
            f"Default should be sequential: expected ~{n * sleep_s}s, got {wall:.2f}s"
        )

    def test_jobs_from_profile(self, tmp_path):
        """`jobs:` key in molcfg profile takes effect when CLI ``-j`` is omitted."""
        sleep_s = 0.3
        n = 3
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_parallel_script(script, workspace_root, n_experiments=n, sleep_s=sleep_s)
        molcfg.write_text(
            "defaults: {}\n"
            "profiles:\n"
            "  fast:\n"
            f"    jobs: {n}\n"
        )

        t0 = time.perf_counter()
        result = runner.invoke(
            app,
            ["run", str(script), "--local", "--config", str(molcfg), "--profile", "fast"],
        )
        wall = time.perf_counter() - t0

        assert result.exit_code == 0, result.output
        assert wall < 2 * sleep_s * n * 0.7, (
            f"Profile jobs={n} should parallelize, expected ~{sleep_s}s, got {wall:.2f}s"
        )

    def test_cli_jobs_overrides_profile(self, tmp_path):
        """CLI ``-j 1`` beats profile ``jobs: 4``."""
        sleep_s = 0.2
        n = 3
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_parallel_script(script, workspace_root, n_experiments=n, sleep_s=sleep_s)
        molcfg.write_text(
            "defaults: {}\n"
            "profiles:\n"
            "  loud:\n"
            "    jobs: 4\n"
        )

        t0 = time.perf_counter()
        result = runner.invoke(
            app,
            ["run", str(script), "--local", "-j", "1",
             "--config", str(molcfg), "--profile", "loud"],
        )
        wall = time.perf_counter() - t0

        assert result.exit_code == 0, result.output
        assert wall >= n * sleep_s * 0.9, (
            f"CLI -j 1 should override profile jobs=4; expected ~{n*sleep_s}s, got {wall:.2f}s"
        )
