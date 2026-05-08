"""Workflow-layer primitives for research pipelines.

The agent-coupled primitives (``register_task_path`` and
``sanity_check``) have been removed from the workflow layer; their
tests live with the agent layer instead. This file retains:

- ``dependent_params=`` on ``Workflow.add()`` / ``@wf.task``
- ``@wf.reduce(over=...)`` cross-replicate reducer
"""

from __future__ import annotations

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
        # Outputs from 3 sibling replicate runs (cross-replicate fan-in).
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
