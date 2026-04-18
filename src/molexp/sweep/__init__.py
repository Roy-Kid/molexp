"""Sweep-level orchestration built on pydantic-graph.

Phase-1 scope: a single-node graph that fans out replicas
(``experiment × mol_run``) concurrently, bounded by a ``jobs`` semaphore.
Every replica currently executes *in-process* via
``experiment.workflow.execute(...)``.

Future phases (see ``docs/spec/unified-pydantic-graph-dispatch.md``):

* **Phase 2** — replace the local-only body with backend-aware dispatch
  (``local`` / ``slurm`` / ``pbs`` / ``lsf``) routed through molq.
* **Phase 3** — expose per-node ``backend=`` on inner ``@wf.task``.
"""

from .graph import (
    SweepDeps,
    SweepReplica,
    SweepRoot,
    SweepState,
    run_sweep,
)

__all__ = [
    "SweepDeps",
    "SweepReplica",
    "SweepRoot",
    "SweepState",
    "run_sweep",
]
