"""Lower an agent-side ``PlanGraph`` into a workflow-layer ``WorkflowContract``.

This module owns the one-directional layering seam sub-spec 01 declared:
:class:`~molexp.agent.modes._planning.PlanGraph` (agent layer) is
*upstream* of :class:`~molexp.workflow.WorkflowContract` (workflow
layer). AuthorMode lowers a ``PlanGraph`` here and records the resulting
``workflow_id`` in :attr:`PlanGraph.compiled_contract_ref`. The reverse
never happens; this module imports the workflow layer through its public
surface only.

The lowering is deterministic — each :class:`~molexp.agent.modes._planning.PlanStep`
becomes one :class:`~molexp.workflow.TaskIO`:

- a step input with a ``source_step`` becomes a
  :class:`~molexp.workflow.TaskInputSpec` whose ``source`` is that
  upstream step id;
- a step output name becomes a
  :class:`~molexp.workflow.TaskOutputSpec`;
- a step artifact becomes an :class:`~molexp.workflow.ArtifactDecl`.

The lowered contract is run through
:func:`~molexp.workflow.validate_workflow_contract` and the normalizer
(:func:`~molexp.agent.modes.author.contract_normalize.normalize_contract`)
so AuthorMode never proceeds on a silently-broken contract.

Pure data + pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import PlanGraph, PlanStep
from molexp.agent.modes.author.contract_normalize import (
    ContractNormalizeReport,
    normalize_contract,
)
from molexp.workflow import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
    ValidationReport,
    WorkflowContract,
    default_validation_checks,
    validate_workflow_contract,
)

__all__ = ["LoweringResult", "lower_plan_graph"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_DEFAULT_IO_TYPE = "object"
"""Planning-stage I/O is untyped; the lowered contract uses a generic
placeholder type the workflow contract checks treat opaquely."""


class LoweringResult(BaseModel):
    """The outcome of one :func:`lower_plan_graph` call.

    Attributes:
        plan_graph: The input plan with
            :attr:`~molexp.agent.modes._planning.PlanGraph.compiled_contract_ref`
            set to the lowered ``workflow_id``.
        contract: The lowered (and normalized)
            :class:`~molexp.workflow.WorkflowContract`.
        validation_report: The
            :class:`~molexp.workflow.ValidationReport` from
            :func:`~molexp.workflow.validate_workflow_contract`.
        normalize_report: The normalizer's residual-issue report.
    """

    model_config = _FROZEN

    plan_graph: PlanGraph
    contract: WorkflowContract
    validation_report: ValidationReport
    normalize_report: ContractNormalizeReport

    @property
    def ok(self) -> bool:
        """Whether the lowered contract validates with no residual issues."""
        return self.validation_report.ok and self.normalize_report.ok


def _step_to_task_io(step: PlanStep) -> TaskIO:
    """Lower one :class:`PlanStep` into a :class:`TaskIO`."""
    inputs = tuple(
        TaskInputSpec(
            name=inp.name,
            type=_DEFAULT_IO_TYPE,
            required=True,
            source=inp.source_step,
        )
        for inp in step.io.inputs
    )
    outputs = tuple(TaskOutputSpec(name=name, type=_DEFAULT_IO_TYPE) for name in step.io.outputs)
    artifacts = tuple(
        ArtifactDecl(
            path=artifact.path,
            description=artifact.description,
            produced_by=step.id,
        )
        for artifact in step.artifacts
    )
    return TaskIO(task_id=step.id, inputs=inputs, outputs=outputs, artifacts=artifacts)


def lower_plan_graph(plan_graph: PlanGraph) -> LoweringResult:
    """Lower ``plan_graph`` into a validated, normalized ``WorkflowContract``.

    The contract's ``workflow_id`` is derived from the plan id. The
    returned :class:`LoweringResult` carries an updated ``PlanGraph``
    whose ``compiled_contract_ref`` equals the lowered ``workflow_id``,
    plus the validation report and the normalizer report so AuthorMode
    can surface residual issues instead of proceeding on a broken
    contract.
    """
    task_io = tuple(_step_to_task_io(step) for step in plan_graph.steps)
    workflow_id = _workflow_id_for(plan_graph)
    raw_contract = WorkflowContract(
        workflow_id=workflow_id,
        task_io=task_io,
        validation_checks=default_validation_checks(),
    )
    normalize_report = normalize_contract(raw_contract)
    contract = normalize_report.contract
    validation_report = validate_workflow_contract(contract)
    updated_plan = plan_graph.model_copy(update={"compiled_contract_ref": workflow_id})
    return LoweringResult(
        plan_graph=updated_plan,
        contract=contract,
        validation_report=validation_report,
        normalize_report=normalize_report,
    )


def _workflow_id_for(plan_graph: PlanGraph) -> str:
    """Derive a stable, deterministic ``workflow_id`` for a plan."""
    return f"wf_{plan_graph.plan_id}"
