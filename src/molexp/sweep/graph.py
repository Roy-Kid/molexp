"""Sweep-level pydantic-graph: fan out replicas with bounded concurrency.

Phase-1 design. See ``docs/spec/unified-pydantic-graph-dispatch.md``.

A sweep is the outer loop of a ``molexp run`` invocation: one replica per
``(experiment, mol_run)`` pair.  Before this module existed, the sweep
was a plain ``for`` loop in ``molexp.cli.run_cmd._execute_sweep`` that
serially ``asyncio.run``'d each replica's workflow — so ``molexp run
--local`` could not parallelise experiments.

:func:`run_sweep` replaces that loop with a one-node pydantic-graph
(:class:`SweepRoot`) whose body ``asyncio.gather``s replicas under a
``Semaphore(jobs)``.  The graph shape is deliberately trivial in this
phase; the point is to have pydantic-graph own sweep lifecycle so
snapshot, resume, and per-node backend routing (Phase 2/3) can extend it
without changing the public entry point.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mollog import get_logger
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from molexp.config import ProfileConfig

logger = get_logger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class SweepReplica:
    """One unit of work in the sweep graph.

    Pairs a molexp ``Run`` with the ``Experiment`` whose workflow should
    execute it.  The run's ``id`` (fallback: ``id(mol_run)``) keys into
    :class:`SweepState`.

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
        outputs: ``replica_id -> WorkflowResult`` for replicas that completed
            without raising.
        failures: ``replica_id -> "<ExcType>: <message>"`` for replicas that
            raised during ``workflow.execute``.  Failures are captured, not
            re-raised, so one replica's error does not cancel its peers.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        """``True`` iff no replica raised during ``workflow.execute``."""
        return not self.failures

    def _sync_from(self, other: "SweepState") -> None:
        """Mirror *other* into ``self`` in place (pydantic-graph snapshot)."""
        self.outputs = other.outputs
        self.failures = other.failures


@dataclass
class SweepDeps:
    """Dependencies injected into :class:`SweepRoot`."""

    replicas: list[SweepReplica]
    profile_config: ProfileConfig | None
    semaphore: asyncio.Semaphore


# ── Graph node ──────────────────────────────────────────────────────────────


@dataclass
class SweepRoot(BaseNode[SweepState, SweepDeps, SweepState]):
    """Single pydantic-graph node that fans out replicas concurrently.

    The body ``asyncio.gather``s one coroutine per replica; each coroutine
    acquires ``ctx.deps.semaphore`` before calling
    ``experiment.workflow.execute``.  Exceptions raised by a replica are
    caught and recorded in :attr:`SweepState.failures`.

    Future phases will split this into one node per replica to enable
    per-replica backend routing, snapshot/resume at replica granularity,
    and finer-grained observation.
    """

    async def run(
        self, ctx: GraphRunContext[SweepState, SweepDeps]
    ) -> End[SweepState]:
        replicas = ctx.deps.replicas
        sem = ctx.deps.semaphore
        cfg = ctx.deps.profile_config

        async def _run_one(
            replica: SweepReplica,
        ) -> tuple[str, Any, Exception | None]:
            rid = _replica_id(replica)
            try:
                async with sem:
                    result = await replica.experiment.workflow.execute(
                        run=replica.mol_run,
                        profile_config=cfg,
                    )
                return rid, result, None
            except Exception as exc:
                logger.exception(f"Sweep replica {rid!r} failed")
                return rid, None, exc

        results = await asyncio.gather(*[_run_one(r) for r in replicas])
        new_state = SweepState()
        for rid, output, exc in results:
            if exc is not None:
                new_state.failures[rid] = f"{type(exc).__name__}: {exc}"
            else:
                new_state.outputs[rid] = output

        ctx.state._sync_from(new_state)
        return End(ctx.state)


# ── Public entry point ──────────────────────────────────────────────────────


async def run_sweep(
    replicas: list[SweepReplica],
    *,
    profile_config: ProfileConfig | None = None,
    jobs: int = 1,
) -> SweepState:
    """Execute *replicas* concurrently, bounded by ``jobs``.

    Args:
        replicas: ``(mol_run, experiment)`` pairs to execute.
        profile_config: Active :class:`~molexp.config.ProfileConfig` forwarded
            unchanged to each ``workflow.execute`` call.
        jobs: Maximum concurrent replicas.  Values ``<= 0`` are clamped up
            to ``1`` (``Semaphore(0)`` would deadlock).

    Returns:
        :class:`SweepState` with ``outputs`` and ``failures`` populated.
        An empty *replicas* list returns an empty state without spinning
        up a graph.
    """
    if not replicas:
        return SweepState()

    sem = asyncio.Semaphore(max(1, jobs))
    deps = SweepDeps(
        replicas=replicas,
        profile_config=profile_config,
        semaphore=sem,
    )
    state = SweepState()
    graph: Graph[SweepState, SweepDeps, SweepState] = Graph(nodes=[SweepRoot])
    run_result = await graph.run(SweepRoot(), state=state, deps=deps)
    return run_result.output


# ── Helpers ─────────────────────────────────────────────────────────────────


def _replica_id(replica: SweepReplica) -> str:
    """Stable string ID for *replica* (``mol_run.id`` with fallback)."""
    rid = getattr(replica.mol_run, "id", None)
    if rid is not None:
        return str(rid)
    return f"replica-{id(replica.mol_run)}"
