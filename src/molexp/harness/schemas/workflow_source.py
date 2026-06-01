"""``WorkflowSource`` — generated runnable ``molexp.workflow`` source code.

The product of :class:`~molexp.harness.stages.generate_workflow_source.GenerateWorkflowSource`:
a Python program (text) that builds a ``molexp.workflow`` Workflow via the
public surface, plus the metadata needed to validate + trace it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["WorkflowSource"]


class WorkflowSource(BaseModel):
    """A generated ``molexp.workflow`` program + derivation metadata.

    Attributes:
        source: The emitted Python source. By convention it defines a
            module-level ``build_workflow()`` returning a ``WorkflowBuilder``,
            which :class:`ValidateWorkflowSource` calls and ``.build()``s.
        module_name: A suggested module name for the program.
        bound_workflow_id: The ``BoundWorkflow`` artifact id this derives from.
        symbols: The public ``molexp.workflow`` symbols the program uses
            (e.g. ``("WorkflowBuilder", "Task", "TaskContext")``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    module_name: str
    bound_workflow_id: str
    symbols: tuple[str, ...] = ()
