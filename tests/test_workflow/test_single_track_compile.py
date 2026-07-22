"""Single-track compile invariant — no codegen subclass wrapping.

Locks the workflow-rectification red line (`single-track-compile`): ``compile()``
registers the exact user-supplied ``Task`` / ``Actor`` instance as the per-task
Step body — it does NOT wrap it in a generated ``BaseNode`` subclass (there is no
pg ``Graph``, no ``node_classes``, no ``make_task_node_class``). ``is`` identity
against the compiled registration is the contract.
"""

from __future__ import annotations

from molexp.workflow import Task, TaskContext, WorkflowCompiler


class TestCompiledWorkflow:
    def test_registration_holds_user_instance_not_codegen_subclass(self):
        class MyTask(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 42

        user_instance = MyTask()
        spec = WorkflowCompiler(name="identity").add(user_instance, name="my").compile()

        reg = next(t for t in spec._tasks if t.name == "my")
        assert reg.fn_or_class is user_instance, (
            "the registration's fn_or_class must be the user-registered Task "
            "instance itself, not a codegen subclass wrapping it."
        )
