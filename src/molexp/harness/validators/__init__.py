"""Pure structural validators for the harness IRs.

Two functions, both sync, deterministic, no I/O:

- :func:`validate_workflow_ir` — checks a :class:`PlanWorkflowIR` for
  duplicate task ids, dangling edges, cycles, missing producers,
  unresolved inputs, agent-inferred parameters that aren't flagged for
  review, and a defense-in-depth grep for shell commands / backend leaks.
- :func:`validate_bound_workflow` — checks a :class:`BoundWorkflow`
  against its :class:`PlanWorkflowIR` for ir-task mapping consistency,
  input/output key agreement, allowed-path workspace containment,
  baseline deny-list floor, and edge topology equivalence.

Neither raises; both return a :class:`PlanValidationReport`. Phase-4 stage
wrappers (`ValidateWorkflowIR`, `ValidateBoundWorkflow`) own the
report-to-error lift.
"""

from __future__ import annotations

from molexp.harness.validators.bound_workflow import BoundWorkflowValidator
from molexp.harness.validators.experiment_spec import ExperimentSpecValidator
from molexp.harness.validators.input_set import InputSetValidator
from molexp.harness.validators.plan_form import PlanFormValidator
from molexp.harness.validators.provenance import ProvenanceValidator
from molexp.harness.validators.test_source import TestSourceValidator
from molexp.harness.validators.test_spec import TestSpecValidator
from molexp.harness.validators.workflow_ir import WorkflowIRValidator
from molexp.harness.validators.workflow_source import WorkflowSourceValidator

__all__ = [
    "BoundWorkflowValidator",
    "ExperimentSpecValidator",
    "InputSetValidator",
    "PlanFormValidator",
    "ProvenanceValidator",
    "TestSourceValidator",
    "TestSpecValidator",
    "WorkflowIRValidator",
    "WorkflowSourceValidator",
]
