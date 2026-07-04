"""Tests for single-track compile — no dead pg Graph emission.

Spec: workflow-rectification (criterion `single-track-compile`,
`workflowstep-is-only-basenode`).

After the rectification:

- ``Task`` and ``Actor`` are NOT subclasses of ``pydantic_graph.BaseNode``.
- ``CompiledWorkflow`` has no ``graph`` / ``node_classes`` attributes.
- ``compiled.task_by_name`` values are the user-registered ``Task`` /
  ``Actor`` instances themselves, not codegen subclasses.
- ``make_task_node_class`` is gone from ``_engine/node.py``.
"""

from __future__ import annotations

from molexp.workflow import Task, TaskContext, WorkflowCompiler


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
