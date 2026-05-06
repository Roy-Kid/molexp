"""Sweep-level fan-out — run replicas concurrently under a semaphore.

The fan-out is bounded by a ``jobs`` semaphore; replicas execute
in-process via ``experiment.workflow.execute(...)``. Spec 05 will
replace this with the workflow-level ``wf.parallel(...)`` primitive,
at which point sweeps are expressed as workflows.
"""

from .graph import SweepReplica, SweepState, run_sweep

__all__ = ["SweepReplica", "SweepState", "run_sweep"]
