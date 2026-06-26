"""``WorkflowSource`` — generated runnable ``molexp.workflow`` source code.

The product of :class:`~molexp.harness.stages.generate_workflow_source.GenerateWorkflowSource`:
a Python program (text) that builds a ``molexp.workflow`` Workflow via the
public surface, plus the metadata needed to validate + trace it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["GeneratedFile", "WorkflowSource"]


class GeneratedFile(BaseModel):
    """One file of a multi-file generated program (relative path + source).

    Used by both :class:`WorkflowSource` (``workflow/<task>.py`` modules + the
    ``workflow/__init__.py`` assembly) and the test bundle
    (``tests/test_<task>.py`` files), so a complex workflow with many tasks is
    materialized as separate per-task files instead of one monolithic module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    source: str


class WorkflowSource(BaseModel):
    """A generated ``molexp.workflow`` program + derivation metadata.

    Attributes:
        source: The entry module's source — the ``build_workflow()`` assembly.
            By convention it defines a module-level ``build_workflow()``
            returning a ``WorkflowCompiler``. In multi-file mode this is the
            ``workflow/__init__.py`` content (it imports the per-task modules);
            in single-file mode it is the whole program. ``ValidateWorkflowSource``
            checks it and ``CompileWorkflow`` ``.compile()``s the assembled package.
        module_name: The importable entrypoint exposing ``build_workflow`` —
            the package name (e.g. ``"workflow"``) in multi-file mode, or a
            module name (e.g. ``"generated_workflow"``) in single-file mode.
        files: The per-task modules + the assembly, each at its relative path
            (e.g. ``workflow/make_data.py``, ``workflow/__init__.py``). Empty
            means single-file mode — ``source`` is the whole program written as
            ``{module_name}.py``.
        bound_workflow_id: The ``BoundWorkflow`` artifact id this derives from.
        symbols: The public ``molexp.workflow`` symbols the program uses
            (e.g. ``("WorkflowCompiler", "Task", "TaskContext")``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    module_name: str
    bound_workflow_id: str
    symbols: tuple[str, ...] = ()
    files: list[GeneratedFile] = Field(default_factory=list)
