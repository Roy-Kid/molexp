"""Concrete :class:`molexp.harness.Mode` subclasses.

Ships :class:`PlanMode` — the idea→experiment-plan→WorkflowIR→runnable
``molexp.workflow`` source pipeline — and :class:`RunMode`, its back half:
generated tests gate a real (executor-subprocess) workflow execution, ending
in a final report + audit on the same ``workspace.Run``.
"""

from __future__ import annotations

from molexp.harness.modes.plan import PlanMode
from molexp.harness.modes.run import RunMode

__all__ = ["PlanMode", "RunMode"]
