"""``services.plan_runtime.persist`` attaches each task's own source to its IR node.

This is what lets the UI graph inspector show a node's code: the generated
``build_workflow()`` program is AST-split so every ``task_config`` carries the
``@wf.task`` function that defines it. Distinct behavior from
``persist_plan_workflow_to_experiment`` (owned by ``test_services/test_plan_persist``).
"""

from __future__ import annotations

from molexp.services.plan_runtime.persist import (
    _compile_package_to_ir,
    _compile_source_to_ir,
)

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

_PKG_INIT = """
from molexp.workflow import WorkflowCompiler
from workflow.gamma import gamma
from workflow.delta import delta


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="pkg-demo")
    wf.task(gamma)
    wf.task(depends_on=["gamma"])(delta)
    return wf
"""

_GAMMA = """
from molexp.workflow import TaskContext


async def gamma(ctx: TaskContext, sigma: float = 1.0) -> dict:
    return {"x": sigma}
"""

_DELTA = """
from molexp.workflow import TaskContext


async def delta(ctx: TaskContext) -> dict:
    x = ctx.inputs["x"]
    return {"y": x + 1}
"""


def _package_source() -> object:
    from molexp.harness.schemas import WorkflowSource

    return WorkflowSource(
        source=_PKG_INIT,
        module_name="workflow",
        bound_workflow_id="bw-pkg",
        symbols=["WorkflowCompiler", "TaskContext"],
        files=[
            {"path": "workflow/__init__.py", "source": _PKG_INIT},
            {"path": "workflow/gamma.py", "source": _GAMMA},
            {"path": "workflow/delta.py", "source": _DELTA},
        ],
    )


class TestCompileSourceToIR:
    def test_attaches_per_task_source_without_bleed(self) -> None:
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

    def test_returns_none_on_non_compiling_source(self) -> None:
        # Observability sugar must never raise — a non-compiling program → None.
        assert _compile_source_to_ir("def build_workflow(:\n    pass") is None


class TestCompilePackageToIR:
    def test_compiles_multi_file_package_via_subprocess_and_annotates(self) -> None:
        """A multi-file WorkflowSource (per-task modules + assembly) must yield a
        display IR too — built in a SUBPROCESS (the harness never execs the
        multi-module program in-process; the display path honors the same trust
        posture), with each node's source sliced from its own module and the
        typed-default params surfaced as input_schema. Regression: production
        multi-file plans logged `No module named 'workflow'` and the UI graph
        stayed empty."""
        ir = _compile_package_to_ir(_package_source())
        assert ir is not None
        by_id = {tc["task_id"]: tc for tc in ir["task_configs"]}
        assert set(by_id) == {"gamma", "delta"}
        assert "async def gamma" in by_id["gamma"]["source"]
        assert 'ctx.inputs["x"]' in by_id["delta"]["source"]
        assert {f["name"] for f in ir["input_schema"]} == {"sigma"}

    def test_returns_none_on_broken_package(self) -> None:
        from molexp.harness.schemas import WorkflowSource

        broken = WorkflowSource(
            source="import missing_sibling_mod\n\ndef build_workflow():\n    return None\n",
            module_name="workflow",
            bound_workflow_id="bw-broken",
            symbols=[],
            files=[
                {
                    "path": "workflow/__init__.py",
                    "source": "import missing_sibling_mod\n\ndef build_workflow():\n    return None\n",
                }
            ],
        )
        assert _compile_package_to_ir(broken) is None
