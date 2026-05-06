"""Sweep-level fan-out: run replicas concurrently under a semaphore.

A sweep is the outer loop of a ``molexp run`` invocation: one replica
per ``(experiment, mol_run)`` pair. :func:`run_sweep` gathers replicas
concurrently, bounded by ``Semaphore(jobs)``, and aggregates outputs
and failures into a :class:`SweepState`.

The fan-out implementation is intentionally direct (``asyncio.gather``
under a semaphore) because spec 05 will replace it with the
workflow-level ``wf.parallel(...)`` primitive; at that point sweeps
are expressed as workflows and this module is folded into the
runtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mollog import get_logger

from molexp.config import ProfileConfig

logger = get_logger(__name__)


@dataclass
class SweepReplica:
    """One unit of work in the sweep.

    Pairs a molexp ``Run`` with the ``Experiment`` whose workflow
    should execute it. The run's ``id`` (fallback: ``id(mol_run)``)
    keys into :class:`SweepState`.

    Attributes:
        mol_run: A ``molexp.workspace.Run`` (or any object exposing ``id``).
        experiment: An object exposing ``workflow.execute(run=, profile_config=)``.
    """

    mol_run: Any
    experiment: Any


@dataclass
class SweepState:
    """Aggregated outcome of a sweep.

    Attributes:
        outputs: ``replica_id -> WorkflowResult`` for replicas that
            completed without raising.
        failures: ``replica_id -> "<ExcType>: <message>"`` for replicas
            that raised during ``workflow.execute``. Failures are
            captured, not re-raised, so one replica's error does not
            cancel its peers.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        """``True`` iff no replica raised during ``workflow.execute``."""
        return not self.failures


async def run_sweep(
    replicas: list[SweepReplica],
    *,
    profile_config: ProfileConfig | None = None,
    jobs: int = 1,
) -> SweepState:
    """Execute *replicas* concurrently, bounded by ``jobs``.

    Args:
        replicas: ``(mol_run, experiment)`` pairs to execute.
        profile_config: Active :class:`~molexp.config.ProfileConfig`
            forwarded unchanged to each ``workflow.execute`` call.
        jobs: Maximum concurrent replicas. Values ``<= 0`` are clamped
            up to ``1`` (``Semaphore(0)`` would deadlock).

    Returns:
        :class:`SweepState` with ``outputs`` and ``failures`` populated.
        An empty *replicas* list returns an empty state.
    """
    if not replicas:
        return SweepState()

    sem = asyncio.Semaphore(max(1, jobs))

    async def _run_one(
        replica: SweepReplica,
    ) -> tuple[str, Any, Exception | None]:
        rid = _replica_id(replica)
        try:
            async with sem:
                result = await replica.experiment.workflow.execute(
                    run=replica.mol_run,
                    profile_config=profile_config,
                )
            return rid, result, None
        except Exception as exc:
            logger.exception(f"Sweep replica {rid!r} failed")
            return rid, None, exc

    results = await asyncio.gather(*[_run_one(r) for r in replicas])
    state = SweepState()
    for rid, output, exc in results:
        if exc is not None:
            state.failures[rid] = f"{type(exc).__name__}: {exc}"
        else:
            state.outputs[rid] = output
    return state


def _replica_id(replica: SweepReplica) -> str:
    """Stable string ID for *replica* (``mol_run.id`` with fallback)."""
    rid = getattr(replica.mol_run, "id", None)
    if rid is not None:
        return str(rid)
    return f"replica-{id(replica.mol_run)}"
