"""Plan-diff-centric repair state carried across PlanMode loop iterations.

The repair loop is a plain ``while`` loop bounded by a max-iterations
budget. Between iterations it threads a mutable :class:`PlanRuntimeState`
scratchpad whose ``repair_signal`` slot carries the
:class:`~molexp.agent.modes._planning.PlanDiff` produced by a preflight
failure or a rejected ``approve_direction`` gate.

:class:`RepairSignal` is a frozen pydantic value (it wraps a frozen
``PlanDiff``); :class:`PlanRuntimeState` is a plain mutable runtime
container because the loop mutates it in place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import PlanDiff

__all__ = ["PlanRuntimeState", "RepairSignal"]


class RepairSignal(BaseModel):
    """A planted repair request — wraps the :class:`PlanDiff` to apply.

    Attributes:
        plan_diff: The diff the repair loop applies via
            :meth:`PlanDiff.apply` before re-running the affected
            stages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_diff: PlanDiff


class PlanRuntimeState:
    """Mutable scratchpad threaded across PlanMode repair-loop iterations.

    Plain runtime container — the loop mutates ``iteration`` and
    ``repair_signal`` between rounds, so this is not a frozen pydantic
    model.

    Attributes:
        iteration: Zero on the first pass; bumped per repair round.
        repair_signal: The pending :class:`RepairSignal`, or ``None``
            when no repair is queued.
    """

    def __init__(
        self,
        *,
        iteration: int = 0,
        repair_signal: RepairSignal | None = None,
    ) -> None:
        self.iteration = iteration
        self.repair_signal = repair_signal

    def plant(self, signal: RepairSignal) -> None:
        """Queue ``signal`` for the loop to consume next iteration."""
        self.repair_signal = signal

    def consume(self) -> RepairSignal | None:
        """Return and clear the pending repair signal."""
        signal, self.repair_signal = self.repair_signal, None
        return signal
