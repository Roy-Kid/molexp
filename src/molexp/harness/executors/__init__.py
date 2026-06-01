"""Executor backends for ``molexp.harness`` (Phase 9 §6).

- :class:`Executor` Protocol — runtime-checkable contract.
- :class:`DryRunExecutor` — no real subprocess; for pipeline smoke tests.
- :class:`LocalExecutor` — real ``subprocess.run`` with timeout + stdout/stderr capture.

``SlurmExecutor`` lands in a later phase (HPC submission lifecycle is its
own beast).
"""

from __future__ import annotations

from molexp.harness.executors.dry_run import DryRunExecutor
from molexp.harness.executors.executor import Executor
from molexp.harness.executors.local import LocalExecutor

__all__ = ["DryRunExecutor", "Executor", "LocalExecutor"]
