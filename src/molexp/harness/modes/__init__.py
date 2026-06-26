"""Concrete :class:`molexp.harness.Mode` subclasses.

Ships the single :class:`PlanMode` — the idea → verified plan → execution
report pipeline (9 steps). Real scientific execution is its opt-in
``execute=True`` tail (the folded-in former RunMode back half), gated by the
step-8 review; it runs the workflow as executor subprocesses and writes the
final + audit reports on the same ``workspace.Run``.
"""

from __future__ import annotations

from molexp.harness.modes.plan import PlanMode

__all__ = ["PlanMode"]
