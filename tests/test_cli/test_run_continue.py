"""RED tests for spec ``continue-two-verbs-02-cli``.

Two continuation verbs replace the single run-level ``--resume``:

* ``--resume`` → NODE-GRANULAR: reopen the run's last non-succeeded execution,
  seed already-completed task outputs from disk, recompute only the rest.
* ``--rerun`` (NEW flag) → fresh new execution on the same run, no seed.
* The two flags are mutually exclusive.

These tests target the not-yet-written production surface:
  - the ``--rerun`` CLI flag on ``molexp run``
  - ``molexp.cli.workspace.run._last_resumable_execution_id``
  - the resume-seeding / rerun-fresh wiring of the local run handler

All tests are deterministic and isolated (``tmp_path`` per test).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import molexp as me
from molexp.cli import app
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord

if TYPE_CHECKING:
    from molexp.workspace.run import Run

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    """Strip ANSI escape codes so literal flag substrings match."""
    return _ANSI_RE.sub("", s)


# ── local copies of the shared test_cli_run.py helpers ─────────────────────────


def _write_script(
    path: Path,
    workspace_root: Path,
    body: str = "ctx.set_result('epochs', ctx.config.get('epochs', 'default'))",
) -> None:
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


def _write_molcfg(path: Path) -> None:
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


_FAIL_ONCE_BODY = (
    "import pathlib\n"
    "    marker = pathlib.Path(ctx.run.run_dir) / 'fail_once'\n"
    "    if not marker.exists():\n"
    "        marker.touch()\n"
    "        raise RuntimeError('boom')\n"
    "    ctx.set_result('epochs', ctx.config['epochs'])"
)


def _replica_run_id(run_params: dict[str, Any], profile_name: str | None = None) -> str:
    """Compute the deterministic run id the replica-dispatch path derives.

    Mirrors the id-seed construction in ``_dispatch_runs`` so a hand-built run
    binds to the same slot the CLI selects. When *profile_name* is given the
    profile name + config hash join the seed exactly as production does.
    """
    from molexp.cli._common import deterministic_run_id

    id_seed = dict(run_params)
    if profile_name is not None:
        from molexp.profile import ProfileConfig

        cfg = ProfileConfig({}, name=profile_name)
        id_seed["_profile"] = profile_name
        id_seed["_config_hash"] = cfg.content_hash()
    return deterministic_run_id(id_seed)


def _only_run(workspace_root: Path) -> Run:
    ws = Workspace.load(workspace_root)
    runs = ws.get_project("demo").get_experiment("train").list_runs()
    assert len(runs) == 1, f"expected exactly one run, got {len(runs)}"
    return runs[0]


def _history(workspace_root: Path) -> list[ExecutionRecord]:
    return list(_only_run(workspace_root).metadata.execution_history)


class _ExecuteRecorder:
    """Async stand-in for ``WorkflowRuntime.execute`` that records kwargs.

    The body is intentionally trivial: the surrounding ``RunContext`` lifecycle
    is responsible for appending / reopening the ``ExecutionRecord`` and marking
    the run status, so a no-op execute lets the lifecycle close cleanly while we
    inspect how the CLI seeded the call.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        compiled: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(dict(kwargs))

        class _Result:
            def __init__(self) -> None:
                self.results: dict[str, Any] = {}

        return _Result()

    @property
    def call(self) -> dict[str, Any]:
        assert len(self.calls) == 1, f"expected one execute call, got {len(self.calls)}"
        return self.calls[0]


def _patch_execute(monkeypatch: pytest.MonkeyPatch) -> _ExecuteRecorder:
    recorder = _ExecuteRecorder()
    import molexp.cli.workspace.run as run_mod

    monkeypatch.setattr(run_mod.WorkflowRuntime, "execute", recorder, raising=True)
    return recorder


def _seed_failed_execution(
    workspace_root: Path,
    *,
    node_outputs: dict[str, Any],
) -> str:
    """Create a single run with one FAILED execution on disk.

    Writes a ``workflow.json`` under that execution holding *node_outputs* as
    completed-task outputs so ``read_node_outputs`` can recover them. Returns
    the failed ``execution_id``.
    """
    ws = me.Workspace(workspace_root)
    project = ws.add_project("demo")
    exp = project.add_experiment("train")
    # Mirror the deterministic id the replica-dispatch path computes so the
    # resume/rerun selection binds to THIS run instead of creating a new one.
    seed = exp.get_seeds()[0]
    run_params = {**exp.params, "seed": seed, "replica": 0}
    run_id = _replica_run_id(run_params)
    run = exp.add_run(parameters=run_params, id=run_id)

    exec_id = "exec-seedfail"
    rec = ExecutionRecord(
        execution_id=exec_id,
        started_at=datetime(2026, 6, 1, 12, 0, 0),
        finished_at=datetime(2026, 6, 1, 12, 1, 0),
        status="failed",
    )
    run._update_metadata(status="failed", execution_history=[rec])

    exec_dir = Path(run.run_dir) / "executions" / exec_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    task_configs = [
        {"task_id": name, "status": "completed", "outputs": value}
        for name, value in node_outputs.items()
    ]
    import json

    (exec_dir / "workflow.json").write_text(json.dumps({"task_configs": task_configs}))
    return exec_id


# ── ac-001: _last_resumable_execution_id helper ────────────────────────────────


class TestLastResumableExecutionId:
    def test_returns_last_non_succeeded_record(self, tmp_path: Path) -> None:
        from molexp.cli.workspace.run import _last_resumable_execution_id

        ws = me.Workspace(tmp_path / "ws")
        run = ws.add_project("demo").add_experiment("train").add_run(parameters={"seed": 0})
        run._update_metadata(
            execution_history=[
                ExecutionRecord(
                    execution_id="exec-1", started_at=datetime(2026, 1, 1), status="failed"
                ),
                ExecutionRecord(
                    execution_id="exec-2", started_at=datetime(2026, 1, 2), status="succeeded"
                ),
                ExecutionRecord(
                    execution_id="exec-3", started_at=datetime(2026, 1, 3), status="failed"
                ),
            ]
        )
        assert _last_resumable_execution_id(run) == "exec-3"

    def test_skips_trailing_succeeded_to_find_earlier_failure(self, tmp_path: Path) -> None:
        from molexp.cli.workspace.run import _last_resumable_execution_id

        ws = me.Workspace(tmp_path / "ws")
        run = ws.add_project("demo").add_experiment("train").add_run(parameters={"seed": 0})
        run._update_metadata(
            execution_history=[
                ExecutionRecord(
                    execution_id="exec-1", started_at=datetime(2026, 1, 1), status="failed"
                ),
                ExecutionRecord(
                    execution_id="exec-2", started_at=datetime(2026, 1, 2), status="cancelled"
                ),
            ]
        )
        # Most recent non-succeeded is exec-2.
        assert _last_resumable_execution_id(run) == "exec-2"

    def test_returns_none_for_empty_history(self, tmp_path: Path) -> None:
        from molexp.cli.workspace.run import _last_resumable_execution_id

        ws = me.Workspace(tmp_path / "ws")
        run = ws.add_project("demo").add_experiment("train").add_run(parameters={"seed": 0})
        assert run.metadata.execution_history == []
        assert _last_resumable_execution_id(run) is None


# ── ac-002: --resume reopens prior execution and seeds completed outputs ────────


class TestResumeSeedsCompletedOutputs:
    def test_resume_reopens_failed_execution_with_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from molexp.workflow import read_node_outputs

        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        exec_id = _seed_failed_execution(workspace_root, node_outputs={"prep": {"value": 7}})
        recorder = _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--resume",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output

        call = recorder.call
        # Reopened the SAME failed execution, not a fresh one.
        assert call.get("execution_id") == exec_id, call
        # Seeded with the completed outputs recovered from disk.
        expected_seed = read_node_outputs(Path(_only_run(workspace_root).run_dir), exec_id)
        assert call.get("seed_outputs") == expected_seed
        assert expected_seed, "fixture should have produced non-empty completed outputs"

    def test_resume_appends_no_new_execution_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        _seed_failed_execution(workspace_root, node_outputs={"prep": {"value": 7}})
        len_before = len(_history(workspace_root))
        _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--resume",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        # Reopen reuses the existing slot — history length is unchanged.
        assert len(_history(workspace_root)) == len_before


# ── ac-003: --rerun starts a fresh execution with no seed ──────────────────────


class TestRerunFreshExecution:
    def test_rerun_executes_without_reopen_or_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        _seed_failed_execution(workspace_root, node_outputs={"prep": {"value": 7}})
        recorder = _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--rerun",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output

        call = recorder.call
        assert call.get("execution_id") is None, call
        assert not call.get("seed_outputs"), call

    def test_rerun_appends_new_execution_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        _seed_failed_execution(workspace_root, node_outputs={"prep": {"value": 7}})
        len_before = len(_history(workspace_root))
        _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--rerun",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(_history(workspace_root)) == len_before + 1


# ── ac-004: --resume and --rerun are mutually exclusive ────────────────────────


class TestMutualExclusivity:
    def test_resume_and_rerun_together_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        recorder = _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--resume",
                "--rerun",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code != 0, result.output
        plain = _plain(result.output)
        assert "--resume" in plain
        assert "--rerun" in plain
        # No run handler should have fired.
        assert recorder.calls == []


# ── --resume on a pending run (no execution) runs its first execution fresh ─────


class TestResumePendingRunRunsFresh:
    def test_resume_on_run_with_empty_history_runs_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        # A pending run created but never executed: resume selects it (status
        # != succeeded) and, with no execution to reopen, runs its FIRST
        # execution fresh — not an error, not a fallback.
        ws = me.Workspace(workspace_root)
        exp = ws.add_project("demo").add_experiment("train")
        run_params = {**exp.params, "seed": exp.get_seeds()[0], "replica": 0}
        run = exp.add_run(parameters=run_params, id=_replica_run_id(run_params))
        run._update_metadata(profile=None, execution_history=[])

        recorder = _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--resume",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        # Ran fresh: no execution_id to reopen, no seed.
        assert recorder.call["execution_id"] is None
        assert not recorder.call["seed_outputs"]


# ── ac-006: --resume reopens even when no node completed (empty seed) ───────────


class TestResumeEmptySeedReopen:
    def test_resume_with_empty_outputs_reopens_without_new_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from molexp.workflow import read_node_outputs

        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        # Failed before any node completed → read_node_outputs returns {}.
        exec_id = _seed_failed_execution(workspace_root, node_outputs={})
        assert read_node_outputs(Path(_only_run(workspace_root).run_dir), exec_id) == {}
        len_before = len(_history(workspace_root))
        recorder = _patch_execute(monkeypatch)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--resume",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output

        call = recorder.call
        assert call.get("execution_id") == exec_id, call
        assert not call.get("seed_outputs"), call
        # Reopened slot — no new record appended.
        assert len(_history(workspace_root)) == len_before


# ── ac-007: shared selection rules under both verbs ────────────────────────────


class TestSharedSelectionRules:
    @pytest.mark.parametrize("verb", ["--resume", "--rerun"])
    def test_succeeded_run_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        # First real run with the smoke profile succeeds.
        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert _only_run(workspace_root).status == "succeeded"

        recorder = _patch_execute(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                verb,
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        # Already succeeded → skipped → execute never called.
        assert recorder.calls == []
        assert "skipped" in _plain(result.output)

    @pytest.mark.parametrize("verb", ["--resume", "--rerun"])
    def test_profile_mismatch_run_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
    ) -> None:
        workspace_root = tmp_path / "workspace"
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        # A failed run carrying the dry_run profile and a resumable execution.
        ws = me.Workspace(workspace_root)
        exp = ws.add_project("demo").add_experiment("train")
        # Bind to the slot the smoke-profile dispatch will select, but stamp the
        # run with a different profile so the mismatch guard skips it.
        run_params = {**exp.params, "seed": exp.get_seeds()[0], "replica": 0}
        run = exp.add_run(
            parameters=run_params, id=_replica_run_id(run_params, profile_name="smoke")
        )
        run._update_metadata(
            status="failed",
            profile="dry_run",
            execution_history=[
                ExecutionRecord(
                    execution_id="exec-x", started_at=datetime(2026, 1, 1), status="failed"
                )
            ],
        )

        recorder = _patch_execute(monkeypatch)
        # Invoke with a DIFFERENT profile (smoke) → mismatch → skipped.
        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--local",
                verb,
                "--config",
                str(molcfg),
                "--profile",
                "smoke",
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert recorder.calls == []


# ── ac-008: background worker forwards the chosen verb ──────────────────────────


class _PopenRecorder:
    def __init__(self) -> None:
        self.argv: list[str] | None = None

    def __call__(self, cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        self.argv = list(cmd)

        class _Proc:
            pid = 4321

        return _Proc()


class TestBackgroundForwardsVerb:
    def test_bg_rerun_forwards_rerun_not_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        import molexp.cli.workspace.run as run_mod

        recorder = _PopenRecorder()
        monkeypatch.setattr(run_mod.subprocess, "Popen", recorder, raising=True)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--bg",
                "--rerun",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert recorder.argv is not None
        assert "--rerun" in recorder.argv
        assert "--resume" not in recorder.argv

    def test_bg_resume_forwards_resume_not_rerun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        script = tmp_path / "train.py"
        molcfg = tmp_path / "molcfg.yaml"
        _write_molcfg(molcfg)
        _write_script(script, workspace_root)

        import molexp.cli.workspace.run as run_mod

        recorder = _PopenRecorder()
        monkeypatch.setattr(run_mod.subprocess, "Popen", recorder, raising=True)

        result = runner.invoke(
            app,
            [
                "run",
                str(script),
                "--bg",
                "--resume",
                "--config",
                str(molcfg),
                "-t",
                str(workspace_root),
            ],
        )
        assert result.exit_code == 0, result.output
        assert recorder.argv is not None
        assert "--resume" in recorder.argv
        assert "--rerun" not in recorder.argv
