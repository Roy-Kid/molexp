"""RED tests for molexp-research-orchestration workspace extensions.

Covers:

- ``Experiment.assets.query(kind=..., recursive=True)`` cross-run aggregation
  (acceptance ac-004 — assets across runs land at experiment level).
- ``RunContext.checkpoint_step`` / ``RunContext.resumed_step`` /
  ``RunContext.suspend`` + ``ResumePolicy.chunk_at(step_budget)``
  walltime chunking under a single ``run.json``
  (acceptance ac-005).
"""

from __future__ import annotations

from molexp.workspace import Workspace
from molexp.workspace.run import RunStatus

# ── ac-004 ── cross-run aggregation ──────────────────────────────────────────


class TestExperimentAssetsRecursive:
    def test_query_recursive_returns_assets_from_all_runs(self, tmp_path) -> None:
        ws = Workspace(root=tmp_path, name="agg-test")
        proj = ws.project("p1")
        exp = proj.experiment("e1")

        # Three runs, each emits an artifact asset.
        for i in range(3):
            run = exp.run(parameters={"i": i})
            with run.start() as ctx:
                ctx.artifact.save(f"result_{i}.json", b"{}")

        # Status quo: exact-scope query on experiment scope should not see
        # run-scoped assets — confirm 0 to bound the contrast.
        exact = exp.assets.query(kind="artifact")
        assert len(exact) == 0

        # New behaviour: recursive=True walks sub-scopes underneath.
        deep = exp.assets.query(kind="artifact", recursive=True)
        assert len(deep) == 3
        names = sorted(a.name for a in deep)
        assert names == ["result_0.json", "result_1.json", "result_2.json"]

    def test_query_recursive_at_project_level_aggregates_across_experiments(self, tmp_path) -> None:
        ws = Workspace(root=tmp_path, name="agg-test-2")
        proj = ws.project("p1")

        for j in range(2):
            exp = proj.experiment(f"e{j}")
            run = exp.run(parameters={"j": j})
            with run.start() as ctx:
                ctx.artifact.save(f"out_{j}.json", b"{}")

        all_artifacts = proj.assets.query(kind="artifact", recursive=True)
        assert len(all_artifacts) == 2

    def test_query_recursive_false_default_preserves_exact_scope(self, tmp_path) -> None:
        """Default behaviour must not change — only opt-in recursion expands scope."""
        ws = Workspace(root=tmp_path, name="agg-test-3")
        proj = ws.project("p1")
        exp = proj.experiment("e1")
        run = exp.run()
        with run.start() as ctx:
            ctx.artifact.save("only.json", b"{}")

        # Default exact-scope: experiment-level returns 0, run-level returns 1.
        assert len(exp.assets.query(kind="artifact")) == 0
        assert len(run.assets.query(kind="artifact")) == 1


# ── ac-005 ── walltime chunking ──────────────────────────────────────────────


class TestWalltimeChunking:
    def test_three_executions_under_one_run_produce_one_runjson(self, tmp_path) -> None:
        from molexp.workspace.resume_policy import ResumePolicy

        ws = Workspace(root=tmp_path, name="chunk-test")
        proj = ws.project("p1")
        exp = proj.experiment("e1")
        run = exp.run(parameters={"target_steps": 10})

        target_steps = 10
        chunk_budget = 4

        def chunked_iteration() -> None:
            with run.start() as ctx:
                start = ctx.resumed_step
                acc = ctx.get_result("sum") or 0
                budget = ResumePolicy.chunk_at(step_budget=chunk_budget)
                for s in range(start, target_steps):
                    if budget.exhausted():
                        ctx.suspend(at_step=s)
                        return
                    acc += s
                    budget.tick()
                    # Persist accumulator each iteration so a suspend
                    # mid-loop preserves the accumulated state.
                    ctx.set_result("sum", acc)
                    ctx.checkpoint_step(s + 1, data={"sum": acc})

        # Three calls of budget=4 cover 0..3, 4..7, 8..9 (last one finishes).
        chunked_iteration()
        chunked_iteration()
        chunked_iteration()

        # Three ExecutionRecord entries under one run.json.
        assert len(run.metadata.execution_history) == 3
        # Run is final — last execution finished naturally.
        assert run.status == RunStatus.SUCCEEDED
        # Resumed_step on a finished run reflects final step count.
        assert run.metadata.last_step == target_steps

        # Result equals one-shot summation.
        run_one_shot = exp.run(id="one-shot")
        with run_one_shot.start() as ctx:
            ctx.set_result("sum", sum(range(target_steps)))

        assert run.get_result("sum") == run_one_shot.get_result("sum")

    def test_resumed_step_zero_on_first_execution(self, tmp_path) -> None:
        ws = Workspace(root=tmp_path, name="chunk-resume0")
        proj = ws.project("p1")
        exp = proj.experiment("e1")
        run = exp.run()
        with run.start() as ctx:
            assert ctx.resumed_step == 0

    def test_suspend_marks_run_resumable_not_succeeded(self, tmp_path) -> None:
        ws = Workspace(root=tmp_path, name="chunk-suspend")
        proj = ws.project("p1")
        exp = proj.experiment("e1")
        run = exp.run()
        with run.start() as ctx:
            ctx.checkpoint_step(2, data={})
            ctx.suspend(at_step=2)

        # Suspended runs are not SUCCEEDED — they wait for resumption.
        assert run.status != RunStatus.SUCCEEDED
        assert run.status != RunStatus.FAILED
        # Subsequent start() observes the prior step as the resume point.
        with run.start() as ctx:
            assert ctx.resumed_step == 2
