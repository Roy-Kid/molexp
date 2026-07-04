"""Two-phase execution pruning — ``molexp.workspace.prune`` (vision-loop-07 §2).

The ONE prune core the CLI (``molexp runs prune``) and the harness lifecycle
capability share: ``plan_execution_prune`` turns a selection (explicit ids /
statuses / everything) into a frozen, reviewable ``ExecutionPrunePlan``;
``apply_execution_prune`` deletes exactly what the plan lists via
``Run.delete_execution`` (rmtree + ``_ops`` history rewrite, atomic per entry).

Contract points under test (spec Testing strategy + CLI-parity with the old
inline core at ``cli/prune.py``):

* planning is read-only and refusal lives at PLAN time — a ``running`` record
  on an actively-running run raises the typed ``LivePruneRefusedError``;
* a ``running`` record on a *terminal* run is a zombie leftover and prunes
  normally (same rule the CLI enforced);
* apply removes exactly the planned dirs and rewrites the ``_ops`` execution
  history — run status / params / other attempts are untouched.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord, RunStatus
from molexp.workspace.prune import (
    ExecutionPrunePlan,
    LivePruneRefusedError,
    apply_execution_prune,
    plan_execution_prune,
)
from molexp.workspace.run import Run

# ── fixtures ─────────────────────────────────────────────────────────────────


def _seed_executions(run: Run, statuses: tuple[str, ...]) -> list[str]:
    """Seed one on-disk execution dir + history record per status; return ids."""
    history: list[ExecutionRecord] = []
    exec_ids: list[str] = []
    for i, status in enumerate(statuses, start=1):
        exec_id = f"exec-{run.id}" if i == 1 else f"exec-{run.id}-{i}"
        exec_dir = Path(str(run.run_dir)) / "executions" / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        (exec_dir / "workflow.json").write_text(f'{{"status":"{status}"}}', encoding="utf-8")
        history.append(
            ExecutionRecord(
                execution_id=exec_id,
                started_at=datetime(2026, 7, 1, 10, i),
                finished_at=datetime(2026, 7, 1, 10, i + 1),
                status=status,
            )
        )
        exec_ids.append(exec_id)
    run.update_ops(
        lambda s: s.model_copy(update={"executions": tuple(history), "status": RunStatus.SUCCEEDED})
    )
    return exec_ids


@pytest.fixture
def seeded(tmp_path: Path) -> tuple[Path, Run, list[str]]:
    """A terminal run with three attempts: one succeeded + two failed."""
    ws = Workspace(root=tmp_path, name="prune-lab")
    exp = ws.add_project("proj-a").add_experiment("exp-x", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1})
    run.materialize()
    exec_ids = _seed_executions(run, ("succeeded", "failed", "failed"))
    return tmp_path, run, exec_ids


def _reloaded_history_ids(ws_path: Path, run: Run) -> list[str]:
    """Execution-history ids read back through a FRESH workspace (disk truth)."""
    reloaded = Workspace.load(ws_path).get_project("proj-a").get_experiment("exp-x").get_run(run.id)
    return [rec.execution_id for rec in reloaded.execution_history]


# ── plan: selection semantics ────────────────────────────────────────────────


class TestPlanSelection:
    def test_statuses_filter_selects_only_matching_attempts(self, seeded) -> None:
        _, run, exec_ids = seeded
        plan = plan_execution_prune(run, statuses=["failed"])
        assert plan.run_id == run.id
        assert [entry.execution_id for entry in plan.entries] == exec_ids[1:]
        assert all(entry.status == "failed" for entry in plan.entries)
        assert all(entry.dir_exists for entry in plan.entries)

    def test_explicit_execution_ids_win(self, seeded) -> None:
        _, run, exec_ids = seeded
        plan = plan_execution_prune(run, execution_ids=[exec_ids[1]])
        assert [entry.execution_id for entry in plan.entries] == [exec_ids[1]]

    def test_unknown_execution_id_raises_key_error(self, seeded) -> None:
        _, run, _ = seeded
        with pytest.raises(KeyError):
            plan_execution_prune(run, execution_ids=["exec-nope"])

    def test_no_filters_selects_every_attempt(self, seeded) -> None:
        _, run, exec_ids = seeded
        plan = plan_execution_prune(run)
        assert [entry.execution_id for entry in plan.entries] == exec_ids

    def test_no_match_yields_an_empty_plan(self, seeded) -> None:
        _, run, _ = seeded
        plan = plan_execution_prune(run, statuses=["cancelled"])
        assert plan.entries == ()

    def test_history_entry_without_a_dir_reports_dir_exists_false(self, seeded) -> None:
        _, run, exec_ids = seeded
        ghost = f"exec-{run.id}-9"
        history = [
            *run.execution_history,
            ExecutionRecord(
                execution_id=ghost,
                started_at=datetime(2026, 7, 1, 11, 0),
                finished_at=datetime(2026, 7, 1, 11, 1),
                status="failed",
            ),
        ]
        run.update_ops(lambda s: s.model_copy(update={"executions": tuple(history)}))
        plan = plan_execution_prune(run, execution_ids=[ghost])
        assert [entry.dir_exists for entry in plan.entries] == [False]
        del exec_ids  # selection is history-driven, not disk-driven


class TestPlanIsReadOnlyAndFrozen:
    def test_planning_touches_no_disk_state(self, seeded) -> None:
        _, run, exec_ids = seeded
        plan_execution_prune(run, statuses=["failed"])
        exec_root = Path(str(run.run_dir)) / "executions"
        assert sorted(p.name for p in exec_root.iterdir()) == sorted(exec_ids)
        assert [rec.execution_id for rec in run.execution_history] == exec_ids


# ── plan: live-record refusal (typed, at PLAN time) ──────────────────────────


class TestLiveRecordRefusal:
    @staticmethod
    def _live_run(tmp_path: Path) -> Run:
        ws = Workspace(root=tmp_path, name="live-lab")
        exp = ws.add_project("proj-a").add_experiment("exp-x", workflow_source="s.py", params={})
        run = exp.add_run(params={})
        run.materialize()
        exec_id = f"exec-{run.id}"
        (Path(str(run.run_dir)) / "executions" / exec_id).mkdir(parents=True)
        run.update_ops(
            lambda s: s.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "executions": (
                        ExecutionRecord(
                            execution_id=exec_id,
                            started_at=datetime(2026, 7, 1, 10, 0),
                            status="running",
                        ),
                    ),
                }
            )
        )
        return run

    def test_running_record_on_active_run_refuses_at_plan_time(self, tmp_path: Path) -> None:
        run = self._live_run(tmp_path)
        with pytest.raises(LivePruneRefusedError):
            plan_execution_prune(run)
        assert (Path(str(run.run_dir)) / "executions" / f"exec-{run.id}").exists()
        assert len(run.execution_history) == 1

    def test_running_record_on_terminal_run_prunes_normally(self, tmp_path: Path) -> None:
        """A zombie leftover (running record, terminal run) is prunable history."""
        run = self._live_run(tmp_path)
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
        plan = plan_execution_prune(run, statuses=["running"])
        assert [entry.execution_id for entry in plan.entries] == [f"exec-{run.id}"]
        apply_execution_prune(run, plan)
        assert not (Path(str(run.run_dir)) / "executions" / f"exec-{run.id}").exists()
        assert run.execution_history == []


# ── apply: exact deletion + history rewrite ──────────────────────────────────


class TestApply:
    def test_apply_removes_exactly_the_planned_dirs(self, seeded) -> None:
        _, run, exec_ids = seeded
        plan = plan_execution_prune(run, statuses=["failed"])
        removed = apply_execution_prune(run, plan)
        assert removed == 2
        exec_root = Path(str(run.run_dir)) / "executions"
        assert sorted(p.name for p in exec_root.iterdir()) == [exec_ids[0]]

    def test_apply_rewrites_the_ops_history(self, seeded) -> None:
        ws_path, run, exec_ids = seeded
        plan = plan_execution_prune(run, statuses=["failed"])
        apply_execution_prune(run, plan)
        assert _reloaded_history_ids(ws_path, run) == [exec_ids[0]]

    def test_apply_leaves_run_status_and_params_alone(self, seeded) -> None:
        _, run, _ = seeded
        plan = plan_execution_prune(run, statuses=["failed"])
        apply_execution_prune(run, plan)
        assert run.status == "succeeded"
        assert run.parameters == {"seed": 1}

    def test_apply_removes_history_row_even_without_a_dir(self, seeded) -> None:
        ws_path, run, _exec_ids = seeded
        ghost = f"exec-{run.id}-9"
        history = [
            *run.execution_history,
            ExecutionRecord(
                execution_id=ghost,
                started_at=datetime(2026, 7, 1, 11, 0),
                finished_at=datetime(2026, 7, 1, 11, 1),
                status="failed",
            ),
        ]
        run.update_ops(lambda s: s.model_copy(update={"executions": tuple(history)}))
        plan = plan_execution_prune(run, execution_ids=[ghost])
        removed = apply_execution_prune(run, plan)
        assert removed == 0
        assert ghost not in _reloaded_history_ids(ws_path, run)

    def test_apply_refuses_a_plan_built_for_another_run(self, seeded, tmp_path: Path) -> None:
        _, run, _ = seeded
        foreign = ExecutionPrunePlan(run_id="someone-else", entries=())
        with pytest.raises(ValueError, match="someone-else"):
            apply_execution_prune(run, foreign)
