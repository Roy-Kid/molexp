"""Pure structural preflight over a typed ``PlanGraph``.

:func:`preflight_plan_graph` runs structural checks over a typed
:class:`~molexp.agent.modes._planning.PlanGraph` against its originating
:class:`~molexp.agent.modes._planning.IntentSpec.required_outputs`. It is
distinct from the *environment* ``check_plan_runtime`` preflight (process
/ file availability) — this one is pure data inspection.

Five check classes:

1. ``graph_closed`` — every ``depends_on`` / input ``source_step``
   resolves to a step in the graph.
2. ``graph_acyclic`` — the ``depends_on`` DAG has no cycle.
3. ``outputs_consumed`` — every ``PlanStepIO`` output is consumed by a
   downstream step *or* is a terminal required-outputs entry.
4. ``every_step_has_api_refs`` — every ``PlanStep`` carries at least one
   ``api_refs`` entry; the ``ResearchAndPlan`` agent's composition
   reasoning lives there.
5. ``every_step_isolated_testable`` — every ``PlanStep`` carries an
   ``IsolatedTestSketch`` marking it isolated-testable.

Returns a frozen :class:`PlanGraphPreflightReport`. A failing report
transitions the plan to :data:`PlanState.preflight_failed`.

Pure functions + frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import PlanGraph

__all__ = [
    "PlanGraphCheck",
    "PlanGraphPreflightReport",
    "preflight_plan_graph",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class PlanGraphCheck(BaseModel):
    """One row of a :class:`PlanGraphPreflightReport`."""

    model_config = _FROZEN

    name: str
    passed: bool
    detail: str = ""


class PlanGraphPreflightReport(BaseModel):
    """Aggregate result of the structural plan-graph preflight."""

    model_config = _FROZEN

    passed: bool
    checks: tuple[PlanGraphCheck, ...]

    def failed_check_names(self) -> tuple[str, ...]:
        """Return the names of the failing checks."""
        return tuple(check.name for check in self.checks if not check.passed)


def preflight_plan_graph(
    *,
    plan_graph: PlanGraph,
    required_outputs: tuple[str, ...],
) -> PlanGraphPreflightReport:
    """Run every structural check; return the aggregate report."""
    checks = (
        _check_graph_closed(plan_graph),
        _check_graph_acyclic(plan_graph),
        _check_outputs_consumed(plan_graph, required_outputs),
        _check_every_step_has_api_refs(plan_graph),
        _check_every_step_isolated_testable(plan_graph),
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


def _check_outputs_consumed(
    plan_graph: PlanGraph, required_outputs: tuple[str, ...]
) -> PlanGraphCheck:
    """Every step output is consumed downstream or is a required output."""
    required = set(required_outputs)
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


def _check_every_step_has_api_refs(plan_graph: PlanGraph) -> PlanGraphCheck:
    """Every PlanStep names at least one project symbol it composes.

    Replaces the prior ``capability_evidenced`` check: instead of looking
    up an external ``CapabilityGraph`` node, we trust the
    ``ResearchAndPlan`` agent's MCP-grounded ``PlanStep.api_refs``. A
    step with empty ``api_refs`` means the agent could not discover any
    toolchain primitive for that step — fail closed.
    """
    bare = [step.id for step in plan_graph.steps if not step.api_refs]
    return PlanGraphCheck(
        name="every_step_has_api_refs",
        passed=not bare,
        detail=(
            ""
            if not bare
            else f"step(s) {', '.join(bare)} carry no api_refs (no grounded primitive)"
        ),
    )


def _check_every_step_isolated_testable(plan_graph: PlanGraph) -> PlanGraphCheck:
    """Every step is decomposed to an isolated-testable granularity.

    A :class:`~molexp.agent.modes._planning.PlanStep` whose
    ``test_sketch.is_isolated_testable`` is ``False`` could only be
    exercised with the real output of an upstream step — it is too
    coarse and the plan is not terminably decomposed. The check fails
    closed so such a plan transitions to
    :data:`~molexp.agent.modes._planning.PlanState.preflight_failed`.
    """
    not_testable = [
        step.id for step in plan_graph.steps if not step.test_sketch.is_isolated_testable
    ]
    return PlanGraphCheck(
        name="every_step_isolated_testable",
        passed=not not_testable,
        detail=(
            ""
            if not not_testable
            else (
                "step(s) "
                + ", ".join(not_testable)
                + " are not decomposed to an isolated-testable granularity"
            )
        ),
    )
