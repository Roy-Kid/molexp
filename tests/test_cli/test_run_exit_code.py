"""RED tests: ``molexp run`` (and the ``execute`` worker) must exit non-zero
when a run fails.

Before this change, ``molexp run --local`` printed "OK N runs completed" and
exited 0 even when a task raised and the run settled to ``status=failed`` — a
silent-failure trap for scripts, CI steps, and schedulers that read the exit
code. In-process execution settles each run to a terminal status synchronously,
so a run that did not reach ``succeeded`` must fail the command; the ``execute``
worker (what molq launches on a node) must likewise propagate the failure as a
non-zero exit so the scheduler marks the job failed.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _write_script(path: Path, workspace_root: Path, *, body: str) -> None:
    """Write a one-task script whose body runs verbatim inside ``train``.

    Uses the legacy ``fn(inputs, config)`` entry form that ``promote_callable``
    adapts, so the body can ``raise`` or ``return`` without any binding setup.
    """
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


def _only_run(workspace_root: Path):
    ws = Workspace.load(workspace_root)
    runs = ws.get_project("demo").get_experiment("train").list_runs()
    assert len(runs) == 1, f"expected exactly one run, got {len(runs)}"
    return runs[0]


def test_run_local_exits_nonzero_when_task_fails(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    script = tmp_path / "train.py"
    _write_script(script, workspace_root, body="raise RuntimeError('boom')")

    result = runner.invoke(app, ["run", str(script), "--local", "-t", str(workspace_root)])

    # Precondition: the run really did fail (not a setup error).
    assert _only_run(workspace_root).status == "failed", result.output
    # The command must surface that failure as a non-zero exit and an honest
    # summary — never "OK ... completed".
    assert result.exit_code != 0, result.output
    plain = _plain(result.output)
    assert "did not succeed" in plain, plain


def test_run_local_names_the_failed_run(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    script = tmp_path / "train.py"
    _write_script(script, workspace_root, body="raise RuntimeError('boom')")

    result = runner.invoke(app, ["run", str(script), "--local", "-t", str(workspace_root)])

    run = _only_run(workspace_root)
    plain = _plain(result.output)
    assert run.id in plain, plain
    assert "failed" in plain, plain


def test_run_local_exits_zero_when_task_succeeds(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    script = tmp_path / "train.py"
    _write_script(script, workspace_root, body="return {'ok': True}")

    result = runner.invoke(app, ["run", str(script), "--local", "-t", str(workspace_root)])

    assert result.exit_code == 0, result.output
    assert _only_run(workspace_root).status == "succeeded", result.output
    assert "OK" in _plain(result.output), result.output


def test_execute_worker_exits_nonzero_on_failed_run(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    script = tmp_path / "train.py"
    _write_script(script, workspace_root, body="raise RuntimeError('boom')")

    # First `run --local` records the defining script on the run and fails it.
    first = runner.invoke(app, ["run", str(script), "--local", "-t", str(workspace_root)])
    assert first.exit_code != 0, first.output
    run_dir = Path(_only_run(workspace_root).run_dir)

    # The worker entry point re-executes the run from its recorded script; the
    # task fails again, so `execute` must exit non-zero for the scheduler.
    result = runner.invoke(app, ["execute", str(run_dir)])

    assert result.exit_code != 0, result.output
    assert _only_run(workspace_root).status == "failed", result.output
