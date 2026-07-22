"""Deterministic feasibility annotator for an emergent-plan task board.

:class:`PlanReachabilityProbe` grades each task on a
:class:`~molexp.harness.plan.TaskBoard` by searching a
:class:`~molexp.harness.registry.CapabilityRegistry` for capabilities that match
the task name, then attaching a :class:`~molexp.harness.plan.FeasibilityAnnotation`
via the pure :func:`~molexp.harness.plan.annotate_feasibility` transition.

The probe is **read-only and artifact-free**: it takes no store, produces no
artifacts, only calls the registry's read-only :meth:`search`, and returns a
**new** board — the input board (and its tasks) are never mutated.
"""

from __future__ import annotations

from molexp.harness.plan import (
    Difficulty,
    FeasibilityAnnotation,
    TaskBoard,
    annotate_feasibility,
)
from molexp.harness.registry import CapabilityRegistry

__all__ = ["PlanReachabilityProbe"]


class PlanReachabilityProbe:
    """Annotate a task board's feasibility from a capability registry.

    Attributes:
        max_refs: Cap on how many matched capability ids are recorded in a
            task's ``probed_refs`` (registry / search order preserved).
    """

    def __init__(self, max_refs: int = 5) -> None:
        self.max_refs = max_refs

    def annotate(self, board: TaskBoard, registry: CapabilityRegistry | None) -> TaskBoard:
        """Return a new board whose tasks carry probe verdicts; never mutates.

        Each task is graded on a deterministic difficulty ladder:

        - ``registry is None`` → ``reachable=False``, ``Difficulty.UNKNOWN``.
        - ``registry.search(task.name)`` empty → ``reachable=False``,
          ``Difficulty.HARD``.
        - one or more hits → ``reachable=True``, ``Difficulty.MODERATE`` when any
          hit declares ``side_effects`` else ``Difficulty.TRIVIAL``; the first
          :attr:`max_refs` hit ids become ``probed_refs``.

        The verdict is folded onto a running board with
        :func:`~molexp.harness.plan.annotate_feasibility`, so this method is a
        read-only projection — it takes no store, writes no artifacts, and
        leaves ``board`` and every input task object untouched.

        Args:
            board: The task board to grade (never mutated).
            registry: The grounded capability catalog, or ``None`` when no
                registry has been grounded yet.

        Returns:
            A new :class:`TaskBoard` with a :class:`FeasibilityAnnotation` on
            every task.
        """
        result = board
        for task in board.tasks:
            annotation = self._probe(task.name, registry)
            result = annotate_feasibility(result, task.id, annotation)
        return result

    def _probe(
        self, name: str, registry: CapabilityRegistry | None
    ) -> FeasibilityAnnotation:
        if registry is None:
            return FeasibilityAnnotation(
                reachable=False,
                difficulty=Difficulty.UNKNOWN,
                probed_refs=(),
                rationale="no capability registry grounded",
            )
        hits = registry.search(name)
        if not hits:
            return FeasibilityAnnotation(
                reachable=False,
                difficulty=Difficulty.HARD,
                probed_refs=(),
                rationale="no grounded capability matched",
            )
        probed_refs = tuple(cap.id for cap in hits[: self.max_refs])
        difficulty = (
            Difficulty.MODERATE
            if any(cap.side_effects for cap in hits)
            else Difficulty.TRIVIAL
        )
        rationale = "grounded on " + ", ".join(cap.id for cap in hits)
        return FeasibilityAnnotation(
            reachable=True,
            difficulty=difficulty,
            probed_refs=probed_refs,
            rationale=rationale,
        )
