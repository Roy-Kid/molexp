"""``persist._compile_source_to_ir`` attaches each task's own source to its IR node.

This is what lets the UI graph inspector show a node's code: the generated
``build_workflow()`` program is AST-split so every ``task_config`` carries the
``@wf.task`` function that defines it.
"""

from __future__ import annotations

from molexp.server.plan_runtime.persist import _compile_source_to_ir

_SOURCE = """
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def alpha(ctx: TaskContext) -> dict:
        return {"x": 1}

    @wf.task(depends_on=["alpha"])
    async def beta(ctx: TaskContext) -> dict:
        x = ctx.inputs["x"]
        return {"y": x + 1}

    return wf
"""


def test_compile_attaches_per_task_source() -> None:
    ir = _compile_source_to_ir(_SOURCE)
    assert ir is not None
    by_id = {tc["task_id"]: tc for tc in ir["task_configs"]}
    assert set(by_id) == {"alpha", "beta"}

    # Each node carries its own decorated function — decorators included, dedented.
    assert by_id["alpha"]["source"].startswith("@wf.task")
    assert "async def alpha" in by_id["alpha"]["source"]
    assert '@wf.task(depends_on=["alpha"])' in by_id["beta"]["source"]
    assert 'ctx.inputs["x"]' in by_id["beta"]["source"]

    # No bleed: alpha's slice must not include beta's body.
    assert "async def beta" not in by_id["alpha"]["source"]


def test_compile_returns_none_on_bad_source() -> None:
    # Observability sugar must never raise — a non-compiling program → None.
    assert _compile_source_to_ir("def build_workflow(:\n    pass") is None
