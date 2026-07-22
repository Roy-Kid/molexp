"""Dataflow-by-name: a batch task body's params bind by name from inputs.

A task body's non-``ctx`` parameters are bound **by name** from {upstream
outputs} | {run params}; ``ctx`` is an optional leading parameter (workdir /
artifacts only, never the data channel). Upstream dicts feeding a multi-dep task
are merged (``task(**a_out, **b_out)``). Pins ``_engine.node._bind_call_args`` /
``_binding_source`` for the batch dispatch path (the streaming-actor dual lives
in ``test_actor_binding``).
"""

from __future__ import annotations

import asyncio

from molexp.workflow import WorkflowCompiler, WorkflowRuntime


def _run(wf: WorkflowCompiler):
    return asyncio.run(WorkflowRuntime().execute(wf.compile()))


class TestByNameBinding:
    def test_single_upstream_dict_binds_present_keys_by_name(self) -> None:
        wf = WorkflowCompiler(name="byname")

        @wf.task
        async def produce(ctx) -> dict:
            return {"x": 3, "y": 4}

        @wf.task(depends_on=["produce"])
        async def consume(ctx, x: int, y: int = 0) -> dict:
            # both keys present upstream bind by name; y=4 overrides its default.
            return {"sum": x + y}

        result = _run(wf)
        assert result.status == "succeeded"
        assert result.outputs["consume"] == {"sum": 7}

    def test_multi_upstream_dicts_merge_by_name(self) -> None:
        """``c(**a_out, **b_out)``: two upstream dicts merge into c's params; no ctx."""
        wf = WorkflowCompiler(name="merge")

        @wf.task
        async def a(ctx) -> dict:
            return {"p": 1}

        @wf.task
        async def b(ctx) -> dict:
            return {"q": 2}

        @wf.task(depends_on=["a", "b"])
        async def c(p: int, q: int) -> dict:  # pure typed function — no ctx at all
            return {"r": p + q}

        result = _run(wf)
        assert result.status == "succeeded"
        assert result.outputs["c"] == {"r": 3}

    def test_var_keyword_param_absorbs_all_inputs(self) -> None:
        wf = WorkflowCompiler(name="kw")

        @wf.task
        async def src(ctx) -> dict:
            return {"a": 1, "b": 2}

        @wf.task(depends_on=["src"])
        async def sink(**inputs: object) -> dict:
            return {"keys": sorted(inputs)}

        result = _run(wf)
        assert result.outputs["sink"] == {"keys": ["a", "b"]}

    def test_default_used_when_named_input_absent(self) -> None:
        wf = WorkflowCompiler(name="default")

        @wf.task
        async def src(ctx) -> dict:
            return {"a": 1}

        @wf.task(depends_on=["src"])
        async def sink(ctx, a: int, scale: float = 2.0) -> dict:
            return {"out": a * scale}

        result = _run(wf)
        assert result.outputs["sink"] == {"out": 2.0}

    def test_missing_required_input_fails_the_run(self) -> None:
        wf = WorkflowCompiler(name="missing")

        @wf.task
        async def src(ctx) -> dict:
            return {"a": 1}

        @wf.task(depends_on=["src"])
        async def sink(ctx, b: int) -> dict:  # 'b' was never produced upstream
            return {"b": b}

        result = _run(wf)
        assert result.status == "failed"
