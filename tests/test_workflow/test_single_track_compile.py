"""Tests for single-track compile — no dead pg Graph emission.

Spec: workflow-rectification (criterion `single-track-compile`,
`workflowstep-is-only-basenode`).

After the rectification:

- ``Task`` and ``Actor`` are NOT subclasses of ``pydantic_graph.BaseNode``.
- ``CompiledWorkflow`` has no ``graph`` / ``node_classes`` attributes.
- ``compiled.task_by_name`` values are the user-registered ``Task`` /
  ``Actor`` instances themselves, not codegen subclasses.
- ``make_task_node_class`` is gone from ``_pydantic_graph/node.py``.
"""

from __future__ import annotations

import pytest

from molexp.workflow import Actor, Task, TaskContext, WorkflowCompiler, WorkflowRuntime


def test_task_is_not_pg_basenode_subclass():
    from pydantic_graph import BaseNode as PgBaseNode

    assert not issubclass(Task, PgBaseNode), (
        "Task must not inherit from pydantic_graph.BaseNode; it is a plain "
        "molexp abstraction (the scheduler invokes its execute() directly)."
    )


def test_actor_is_not_pg_basenode_subclass():
    from pydantic_graph import BaseNode as PgBaseNode

    assert not issubclass(Actor, PgBaseNode), "Actor must not inherit from pydantic_graph.BaseNode."


def test_make_task_node_class_is_gone():
    from molexp.workflow._pydantic_graph import node as node_mod

    assert not hasattr(node_mod, "make_task_node_class"), (
        "make_task_node_class (per-task pg BaseNode codegen) must be removed; "
        "the dead-track is gone."
    )


def test_compiled_workflow_has_no_graph_attribute():
    wf = WorkflowCompiler(name="probe")

    @wf.task
    async def step(ctx: TaskContext) -> int:
        return 1

    spec = wf.compile()

    compiled = spec.graph

    assert not hasattr(compiled, "graph"), (
        "CompiledWorkflow.graph (the dead pg Graph) must be removed."
    )
    assert not hasattr(compiled, "node_classes"), (
        "CompiledWorkflow.node_classes (per-task BaseNode subclasses) must be removed."
    )


def test_compiled_registration_holds_user_instances():
    """The compiled artifact must hold the exact user-registered Task instance —
    no codegen subclass wrapping. ``is`` identity is the contract. (Asserts on
    ``compiled._tasks`` — the removed ``graph.task_by_name`` was a LoweredGraph
    internal; under genuine pg lowering the registration is the source of
    truth for the per-task Step body.)"""

    class MyTask(Task):
        async def execute(self, ctx: TaskContext) -> int:
            return 42

    user_instance = MyTask()
    wf = WorkflowCompiler(name="identity").add(user_instance, name="my")
    spec = wf.compile()

    reg = next(t for t in spec._tasks if t.name == "my")
    assert reg.fn_or_class is user_instance, (
        "the registration's fn_or_class must be the user-registered Task instance "
        "itself, not a codegen subclass wrapping it."
    )


def test_task_no_basenode_run_stub():
    """Task should not carry a ``run`` method that satisfies pg's BaseNode
    contract — the scheduler invokes ``execute(ctx)`` directly, and ``run``
    used to exist only to satisfy BaseNode. After the rectification, either
    ``Task.run`` is absent OR it is not the BaseNode-protocol stub (i.e. its
    docstring does not say 'overridden by the workflow compiler')."""
    run = getattr(Task, "run", None)
    if run is not None:
        # If ``run`` survives for backward compat reasons, it must not be
        # the dead-track stub.
        doc = run.__doc__ or ""
        assert "overridden by the workflow compiler" not in doc, (
            "The dead-track BaseNode.run stub on Task must be removed."
        )


@pytest.mark.asyncio
async def test_workflow_executes_without_per_task_codegen():
    """End-to-end: a workflow with two tasks compiles + runs to completion
    without per-task pg BaseNode codegen."""
    wf = WorkflowCompiler(name="e2e")

    @wf.task
    async def a(ctx: TaskContext) -> int:
        return 1

    @wf.task(depends_on=["a"])
    async def b(ctx: TaskContext) -> int:
        return ctx.inputs + 1

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.status == "completed"
    assert result.outputs == {"a": 1, "b": 2}
