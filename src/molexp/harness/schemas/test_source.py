"""``TestSource`` — generated pytest source for a materialized workflow module.

The product of :class:`~molexp.harness.stages.generate_test_code.GenerateTestCode`:
a pytest module (text) exercising the generated workflow program, plus the
metadata needed to validate + trace it. Mirrors
:class:`~molexp.harness.schemas.workflow_source.WorkflowSource`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.workflow_source import GeneratedFile

__all__ = ["TestSource"]


class TestSource(BaseModel):
    """A generated pytest program + derivation metadata.

    Attributes:
        source: The entry test module's source — in single-file mode the whole
            pytest program; in multi-file mode the first/primary test file.
            By convention test modules import ``build_workflow`` from the
            generated workflow package and define plain ``test_*`` functions.
        module_name: Module name for the entry program; starts with ``test_`` so
            pytest collects it.
        files: One test file per task (e.g. ``tests/test_make_data.py``), each
            at its relative path. Empty means single-file mode — ``source`` is
            written as ``{module_name}.py``.
        test_spec_id: The ``TestSpec`` id these tests realize.
        bound_workflow_id: The ``BoundWorkflow`` artifact id this derives from.
        symbols: The public symbols the program uses (e.g. ``("build_workflow",)``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    module_name: str
    test_spec_id: str
    bound_workflow_id: str
    symbols: tuple[str, ...] = ()
    files: list[GeneratedFile] = Field(default_factory=list)
