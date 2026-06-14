"""``TestSource`` — generated pytest source for a materialized workflow module.

The product of :class:`~molexp.harness.stages.generate_test_code.GenerateTestCode`:
a pytest module (text) exercising the generated workflow program, plus the
metadata needed to validate + trace it. Mirrors
:class:`~molexp.harness.schemas.workflow_source.WorkflowSource`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["TestSource"]


class TestSource(BaseModel):
    """A generated pytest program + derivation metadata.

    Attributes:
        source: The emitted pytest source. By convention it imports
            ``build_workflow`` from the sibling generated workflow module
            and defines plain module-level ``test_*`` functions.
        module_name: Module name for the program; starts with ``test_`` so
            pytest collects it.
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
