"""Pure structural preflight over a typed ``PlanGraph``.

:func:`run_plan_graph_preflight` runs structural checks over a typed
:class:`~molexp.agent.modes._planning.PlanGraph` against its originating
:class:`~molexp.agent.modes._planning.IntentSpec` and the
:class:`~molexp.agent.modes._planning.CapabilityGraph`. It is distinct
from the *environment* ``check_plan_runtime`` preflight (process / file
availability) — this one is pure data inspection.

Six check classes:

1. ``graph_closed`` — every ``depends_on`` / input ``source_step``
   resolves to a step in the graph.
2. ``graph_acyclic`` — the ``depends_on`` DAG has no cycle.
3. ``outputs_consumed`` — every ``PlanStepIO`` output is consumed by a
   downstream step *or* is a terminal ``IntentSpec.required_outputs``
   entry.
4. ``capability_evidenced`` — every ``PlanStep.capability_id`` resolves
   to a :class:`CapabilityNode` with ``EvidenceState.evidenced``.
5. ``requirements_satisfiable`` — every ``IntentSpec.required_outputs``
   entry is produced by some step.
6. ``external_resources`` — bound capability nodes flagging an external
   resource limit are surfaced for a human decision (fails closed).
7. ``side_effects`` — every step with a side effect (a ``rollback``)
   must have a matching ``IntentSpec.allowed_side_effects`` entry.

Returns a frozen :class:`PlanGraphPreflightReport`. A failing report
transitions the plan to :data:`PlanState.preflight_failed`.

Pure functions + frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import (
    CapabilityGraph,
    EvidenceState,
    IntentSpec,
    PlanGraph,
)

__all__ = [
    "PlanGraphCheck",
    "PlanGraphPreflightReport",
    "run_plan_graph_preflight",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_EXTERNAL_RESOURCE_MARKERS: tuple[str, ...] = ("external", "install", "network", "credentials")
"""Substrings in a capability node's ``usage_limits`` that mark an
external-resource dependency a human must vet before approval."""


class PlanGraphCheck(BaseModel):
    """One row of a :class:`PlanGraphPreflightReport`.

    Attributes:
        name: Stable check identifier (``graph_acyclic``, …).
        passed: Whether the check passed.
        detail: Human-readable explanation, populated on failure.
    """

    model_config = _FROZEN

    name: str
    passed: bool
    detail: str = ""


class PlanGraphPreflightReport(BaseModel):
    """Aggregate result of the structural plan-graph preflight.

    Attributes:
        passed: ``True`` iff every check passed.
        checks: The per-check rows, in evaluation order.
    """

    model_config = _FROZEN

    passed: bool
    checks: tuple[PlanGraphCheck, ...]

    def failed_check_names(self) -> tuple[str, ...]:
        """Return the names of the failing checks."""
        return tuple(check.name for check in self.checks if not check.passed)


def run_plan_graph_preflight(
    *,
    plan_graph: PlanGraph,
    intent: IntentSpec,
    capabilities: CapabilityGraph,
) -> PlanGraphPreflightReport:
    """Run every structural check; return the aggregate report."""
    checks = (
        _check_graph_closed(plan_graph),
        _check_graph_acyclic(plan_graph),
        _check_outputs_consumed(plan_graph, intent),
        _check_capability_evidenced(plan_graph, capabilities),
        _check_requirements_satisfiable(plan_graph, intent),
        _check_external_resources(plan_graph, capabilities),
        _check_side_effects(plan_graph, intent),
    )
    return PlanGraphPreflightReport(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def _check_graph_closed(plan_graph: PlanGraph) -> PlanGraphCheck:
    """Every ``depends_on`` / input source resolves to a known step."""
    ids = {step.id for step in plan_graph.steps}
    dangling: list[str] = []
    for step in plan_graph.steps:
        for dep in step.depends_on:
            if dep not in ids:
                dangling.append(f"{step.id} depends_on unknown {dep!r}")
        for inp in step.io.inputs:
            if inp.source_step is not None and inp.source_step not in ids:
                dangling.append(f"{step.id} input {inp.name!r} from unknown {inp.source_step!r}")
    return PlanGraphCheck(
        name="graph_closed",
        passed=not dangling,
        detail="; ".join(dangling),
    )


def _check_graph_acyclic(plan_graph: PlanGraph) -> PlanGraphCheck:
    """The ``depends_on`` graph has no cycle."""
    acyclic = plan_graph.is_acyclic()
    return PlanGraphCheck(
        name="graph_acyclic",
        passed=acyclic,
        detail="" if acyclic else "the plan graph contains a dependency cycle",
    )


def _check_outputs_consumed(plan_graph: PlanGraph, intent: IntentSpec) -> PlanGraphCheck:
    """Every step output is consumed downstream or is a required output."""
    required = set(intent.required_outputs)
    consumed: set[tuple[str, str]] = set()
    for step in plan_graph.steps:
        for inp in step.io.inputs:
            if inp.source_step is not None:
                consumed.add((inp.source_step, inp.name))
    dangling: list[str] = []
    for step in plan_graph.steps:
        for output in step.io.outputs:
            if output in required:
                continue
            if (step.id, output) in consumed:
                continue
            dangling.append(f"{step.id} output {output!r} is unconsumed")
    return PlanGraphCheck(
        name="outputs_consumed",
        passed=not dangling,
        detail="; ".join(dangling),
    )


def _check_capability_evidenced(
    plan_graph: PlanGraph, capabilities: CapabilityGraph
) -> PlanGraphCheck:
    """Every bound ``capability_id`` resolves to an evidenced node."""
    problems: list[str] = []
    for step in plan_graph.steps:
        if step.capability_id is None:
            continue
        node = capabilities.node_by_id(step.capability_id)
        if node is None:
            problems.append(f"{step.id} binds unknown capability {step.capability_id!r}")
        elif node.evidence_state is not EvidenceState.evidenced:
            problems.append(
                f"{step.id} binds capability {step.capability_id!r} "
                f"with evidence state {node.evidence_state.value}"
            )
    return PlanGraphCheck(
        name="capability_evidenced",
        passed=not problems,
        detail="; ".join(problems),
    )


def _check_requirements_satisfiable(plan_graph: PlanGraph, intent: IntentSpec) -> PlanGraphCheck:
    """Every required output is produced by some step."""
    produced: set[str] = set()
    for step in plan_graph.steps:
        produced.update(step.io.outputs)
    unmet = [req for req in intent.required_outputs if req not in produced]
    return PlanGraphCheck(
        name="requirements_satisfiable",
        passed=not unmet,
        detail=("" if not unmet else f"no step produces required output(s): {', '.join(unmet)}"),
    )


def _check_external_resources(
    plan_graph: PlanGraph, capabilities: CapabilityGraph
) -> PlanGraphCheck:
    """Surface bound capabilities that depend on an external resource.

    Fails closed: an external-resource dependency must be confirmed by a
    human before the plan is approved.
    """
    flagged: list[str] = []
    for step in plan_graph.steps:
        if step.capability_id is None:
            continue
        node = capabilities.node_by_id(step.capability_id)
        if node is None:
            continue
        if _has_external_marker(node.usage_limits):
            flagged.append(
                f"{step.id} binds capability {step.capability_id!r} needing an external resource"
            )
    return PlanGraphCheck(
        name="external_resources",
        passed=not flagged,
        detail="; ".join(flagged),
    )


def _has_external_marker(usage_limits: tuple[str, ...]) -> bool:
    """Return whether any usage-limit string marks an external resource."""
    return any(
        marker in limit.lower() for limit in usage_limits for marker in _EXTERNAL_RESOURCE_MARKERS
    )


def _check_side_effects(plan_graph: PlanGraph, intent: IntentSpec) -> PlanGraphCheck:
    """Every step with a rollback (a side effect) must be sanctioned.

    A non-empty ``rollback`` means the step mutates state; the
    ``IntentSpec`` must declare a matching ``allowed_side_effects``
    entry (any entry — the plan-graph layer does not type side-effect
    kinds, so a non-empty allowed set sanctions a non-empty rollback).
    """
    if not intent.allowed_side_effects:
        offenders = [step.id for step in plan_graph.steps if step.rollback]
        return PlanGraphCheck(
            name="side_effects",
            passed=not offenders,
            detail=(
                ""
                if not offenders
                else (
                    f"step(s) {', '.join(offenders)} declare a side effect but "
                    "IntentSpec.allowed_side_effects is empty"
                )
            ),
        )
    return PlanGraphCheck(name="side_effects", passed=True)
