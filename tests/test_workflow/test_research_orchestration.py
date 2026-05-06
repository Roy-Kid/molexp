"""RED tests for molexp-research-orchestration workflow extensions.

Covers the four workflow-layer primitives introduced for agent-authored
research pipelines:

- ``dependent_params=`` on ``Workflow.add()`` / ``@wf.task``
  (acceptance ac-001)
- ``@wf.reduce(over=...)`` cross-replicate reducer
  (acceptance ac-002)
- ``Workflow.register_task_path()`` agent-authored Task hot-load
  (acceptance ac-003)
- ``Workflow.sanity_check(after=, predicate=, on_fail=)``
  (acceptance ac-007)
"""

from __future__ import annotations

import textwrap

import pytest

from molexp.workflow import TaskContext, Workflow

# ── ac-001 ── dependent_params ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestDependentParams:
    async def test_resolves_from_upstream_output_into_config(self) -> None:
        """Downstream sees dependent_params(prev) merged into TaskContext.config."""
        wf = Workflow(name="dep-params")

        @wf.task
        async def cooling(ctx: TaskContext) -> dict:
            return {"Tg": 0.6}

        @wf.task(
            depends_on=["cooling"],
            dependent_params=lambda prev: {"T": 0.7 * prev["cooling"].output["Tg"]},
        )
        async def mechanical(ctx: TaskContext) -> float:
            return float(ctx.config["T"])

        result = await wf.build().execute()
        assert result.status == "completed"
        assert result.outputs["mechanical"] == pytest.approx(0.42)

    async def test_no_dependent_params_keeps_config_unchanged(self) -> None:
        """Sanity: tasks without dependent_params don't get spurious config."""
        wf = Workflow(name="dep-params-absent")

        @wf.task
        async def upstream(ctx: TaskContext) -> int:
            return 1

        @wf.task(depends_on=["upstream"])
        async def downstream(ctx: TaskContext) -> bool:
            return "T" not in ctx.config

        result = await wf.build().execute()
        assert result.outputs["downstream"] is True


# ── ac-002 ── @wf.reduce ──────────────────────────────────────────────────────


class TestReduce:
    def test_reduce_decorator_registers_aggregator(self) -> None:
        """``@wf.reduce(over='replicate')`` registers a reducer on the spec."""
        wf = Workflow(name="rep-reduce")

        @wf.task
        async def cooling(ctx: TaskContext) -> dict:
            return {"Tg": 0.6}

        @wf.reduce(over="replicate")
        def aggregate(replicate_outputs: list[dict]) -> dict:
            return {"mean_Tg": sum(r["Tg"] for r in replicate_outputs) / len(replicate_outputs)}

        spec = wf.build()
        # Outputs from 3 sibling replicate runs (sweep-level fan-in).
        replicate_outputs = [{"Tg": 0.58}, {"Tg": 0.60}, {"Tg": 0.62}]
        reduced = spec.run_reducer(replicate_outputs)
        assert reduced["mean_Tg"] == pytest.approx(0.60)

    def test_reduce_dimension_recorded(self) -> None:
        wf = Workflow(name="rep-reduce-dim")

        @wf.task
        async def stub(ctx: TaskContext) -> int:
            return 1

        @wf.reduce(over="replicate")
        def mean(rs: list[float]) -> float:
            return sum(rs) / len(rs)

        spec = wf.build()
        assert spec.reducer_dimension == "replicate"

    def test_no_reducer_raises_on_run_reducer(self) -> None:
        wf = Workflow(name="no-reducer")

        @wf.task
        async def t(ctx) -> int:
            return 1

        spec = wf.build()
        with pytest.raises(LookupError):
            spec.run_reducer([1, 2, 3])


# ── ac-003 ── register_task_path ─────────────────────────────────────────────


class TestRegisterTaskPath:
    def test_loads_task_class_from_file_into_workflow_scope(self, tmp_path) -> None:
        plugin = tmp_path / "agent_task_foo.py"
        plugin.write_text(
            textwrap.dedent(
                """
                from molexp.workflow import Task

                class FooTask(Task):
                    async def execute(self, ctx) -> str:
                        return "foo"
                """
            )
        )

        wf1 = Workflow(name="wf1")
        wf1.register_task_path(plugin)
        FooTask = wf1.resolve_task_class("FooTask")
        assert FooTask is not None
        instance = FooTask()
        assert callable(getattr(instance, "execute", None))

    def test_scope_does_not_leak_to_sibling_workflow(self, tmp_path) -> None:
        plugin = tmp_path / "agent_task_bar.py"
        plugin.write_text(
            textwrap.dedent(
                """
                from molexp.workflow import Task

                class BarTask(Task):
                    async def execute(self, ctx) -> str:
                        return "bar"
                """
            )
        )

        wf1 = Workflow(name="wf-with-bar")
        wf1.register_task_path(plugin)
        assert wf1.resolve_task_class("BarTask") is not None

        wf2 = Workflow(name="wf-without")
        with pytest.raises(KeyError):
            wf2.resolve_task_class("BarTask")

    def test_invalid_path_raises(self, tmp_path) -> None:
        wf = Workflow(name="wf-bad")
        with pytest.raises((FileNotFoundError, OSError)):
            wf.register_task_path(tmp_path / "does_not_exist.py")


# ── ac-007 ── sanity_check hook ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestSanityCheck:
    async def test_halt_fails_workflow_when_predicate_false(self) -> None:
        wf = Workflow(name="sanity-halt")

        @wf.task
        async def collapsing(ctx: TaskContext) -> dict:
            return {"density": 0.95}

        wf.sanity_check(
            after="collapsing",
            predicate=lambda state: 0.80 <= state.results["collapsing"]["density"] <= 0.90,
            on_fail="halt",
        )

        result = await wf.build().execute()
        assert result.status == "failed"

    async def test_continue_on_pass_does_not_halt(self) -> None:
        wf = Workflow(name="sanity-pass")

        @wf.task
        async def good(ctx: TaskContext) -> dict:
            return {"density": 0.85}

        wf.sanity_check(
            after="good",
            predicate=lambda state: 0.80 <= state.results["good"]["density"] <= 0.90,
            on_fail="halt",
        )

        result = await wf.build().execute()
        assert result.status == "completed"
        assert result.outputs["good"] == {"density": 0.85}

    async def test_replan_emits_event_and_continues(self) -> None:
        """on_fail='replan' records a structured event but does not halt."""
        wf = Workflow(name="sanity-replan")

        @wf.task
        async def collapsing(ctx: TaskContext) -> dict:
            return {"density": 0.95}

        wf.sanity_check(
            after="collapsing",
            predicate=lambda state: state.results["collapsing"]["density"] < 0.90,
            on_fail="replan",
        )

        result = await wf.build().execute()
        # Even with a sanity miss, on_fail='replan' allows the workflow to finish.
        assert result.status == "completed"
        events = getattr(result, "sanity_events", None)
        assert events is not None and len(events) >= 1
        ev = events[0]
        assert ev["task"] == "collapsing"
        assert ev["on_fail"] == "replan"
        assert ev["passed"] is False
