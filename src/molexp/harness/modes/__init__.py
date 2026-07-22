"""Concrete plan-pipeline drivers.

Ships the :class:`EmergentPlanOrchestrator` — the two-phase planning pipeline.
**Phase 1 — emergent planning:** an interactive agent loop drives a task board
from the draft, a deterministic form guard keeps a malformed plan from ever
reaching the human, and a hard review gate freezes the approved board into a
content-addressed experiment plan (suspending store-first when there is no
approver). **Phase 2 — deterministic realization** turns that frozen plan into
executable artifacts; it is a separate phase, not driven here.

The orchestrator is not a ``Mode`` subclass and holds no completion ledger —
its resume correctness rides on the hard gate's store-first replay.
"""

from __future__ import annotations

from molexp.harness.modes.emergent_plan import EmergentPlanOrchestrator

__all__ = ["EmergentPlanOrchestrator"]
