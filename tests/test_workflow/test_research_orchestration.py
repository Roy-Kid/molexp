"""Workflow-layer primitives for research pipelines.

The agent-coupled primitives (``register_task_path`` and
``sanity_check``) have been removed from the workflow layer; their
tests live with the agent layer instead. This file retains:

- ``dependent_params=`` on ``Workflow.add()`` / ``@wf.task``
- ``@wf.reduce(over=...)`` cross-replicate reducer
"""

from __future__ import annotations

import pytest

from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

# ── ac-001 ── dependent_params ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestDependentParams:
    async def test_resolves_from_upstream_output_into_config(self) -> None:
        """A dependent_params(prev) overlay binds to the downstream body's param by name."""
        wf = WorkflowCompiler(name="dep-params")

        @wf.task
        async def cooling(ctx: TaskContext) -> dict:
            return {"Tg": 0.6}

        @wf.task(
            depends_on=["cooling"],
            dependent_params=lambda prev: {"T": 0.7 * prev["cooling"].output["Tg"]},
        )
        async def mechanical(T: float) -> float:  # 'T' is delivered by dependent_params
            return float(T)

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["mechanical"] == pytest.approx(0.42)


# ── ac-002 ── @wf.reduce ──────────────────────────────────────────────────────


class TestReduce:
    def test_reduce_decorator_registers_aggregator(self) -> None:
        """``@wf.reduce(over='replicate')`` registers a reducer on the spec."""
        wf = WorkflowCompiler(name="rep-reduce")

        @wf.task
        async def cooling(ctx: TaskContext) -> dict:
            return {"Tg": 0.6}

        @wf.reduce(over="replicate")
        def aggregate(replicate_outputs: list[dict]) -> dict:
            return {"mean_Tg": sum(r["Tg"] for r in replicate_outputs) / len(replicate_outputs)}

        spec = wf.compile()
        # Outputs from 3 sibling replicate runs (cross-replicate fan-in).
        replicate_outputs = [{"Tg": 0.58}, {"Tg": 0.60}, {"Tg": 0.62}]
        reduced = spec.run_reducer(replicate_outputs)
        assert reduced["mean_Tg"] == pytest.approx(0.60)

    def test_no_reducer_raises_on_run_reducer(self) -> None:
        wf = WorkflowCompiler(name="no-reducer")

        @wf.task
        async def t(ctx) -> int:
            return 1

        spec = wf.compile()
        with pytest.raises(LookupError):
            spec.run_reducer([1, 2, 3])
