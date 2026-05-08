"""Resume policy system for checkpoint-based run resumption.

Provides policy interface and built-in policies for determining
whether to resume from a checkpoint or start fresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .run import RunStatus

if TYPE_CHECKING:
    from .checkpoint import CheckpointState
    from .run import Run


class WalltimeChunk:
    """Step-budget tracker for walltime-aware chunking.

    Used together with :meth:`RunContext.checkpoint_step` and
    :meth:`RunContext.suspend` to break long step iterations across
    multiple executions of the same run.  See
    :func:`chunk_at`.

    Usage::

        with run.start() as ctx:
            budget = chunk_at(step_budget=400)
            for s in range(ctx.resumed_step, target_steps):
                if budget.exhausted():
                    ctx.suspend(at_step=s)
                    return
                # ... work ...
                budget.tick()
                ctx.checkpoint_step(s + 1)
    """

    __slots__ = ("budget", "consumed")

    def __init__(self, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"chunk_at: step_budget must be >= 1; got {budget!r}")
        self.budget = int(budget)
        self.consumed = 0

    def tick(self) -> None:
        """Record one step consumed against the budget."""
        self.consumed += 1

    def exhausted(self) -> bool:
        """Return True once the budget is fully consumed."""
        return self.consumed >= self.budget


@runtime_checkable
class ResumePolicy(Protocol):
    """Interface for checkpoint resume policies.

    Policies determine whether a run should resume from a checkpoint
    or start fresh based on run state and checkpoint properties.

    The walltime-chunking helper :func:`chunk_at` is a module-level
    function — it is not part of the structural protocol because it
    has no ``self`` argument and concrete policy classes need not
    redeclare it.
    """

    def should_resume(self, run: Run, checkpoint: CheckpointState) -> bool:
        """Decide whether to resume from checkpoint.

        Args:
            run: Run instance
            checkpoint: Checkpoint state to potentially resume from

        Returns:
            True if should resume from checkpoint, False to start fresh
        """
        ...


def chunk_at(*, step_budget: int) -> WalltimeChunk:
    """Build a :class:`WalltimeChunk` budget tracker for this chunk.

    The returned object is used by user code inside the run body to
    decide when to call :meth:`RunContext.suspend`. Lives at module
    scope (rather than on :class:`ResumePolicy`) so concrete policy
    classes do not need to redeclare it just to satisfy the Protocol.
    """
    return WalltimeChunk(step_budget)


class AlwaysResumePolicy:
    """Always resume from latest checkpoint if one exists.

    This policy will always attempt to resume from the most recent
    checkpoint, regardless of run status or other factors.
    """

    def should_resume(self, run: Run, checkpoint: CheckpointState) -> bool:
        """Always return True to resume from checkpoint."""
        return True


class NeverResumePolicy:
    """Never resume from checkpoints, always start fresh.

    This policy ignores all checkpoints and always starts runs
    from the beginning. Useful for ensuring clean runs or debugging.
    """

    def should_resume(self, run: Run, checkpoint: CheckpointState) -> bool:
        """Always return False to start fresh."""
        return False


class StatusBasedPolicy:
    """Resume only if previous run failed or was cancelled.

    This policy resumes from checkpoints only when the run status
    indicates a failure or cancellation, allowing recovery from
    interrupted runs while ensuring successful runs start fresh.
    """

    def should_resume(self, run: Run, checkpoint: CheckpointState) -> bool:
        """Resume if run status is FAILED or CANCELLED."""
        return run.status in [RunStatus.FAILED, RunStatus.CANCELLED]


# Policy registry for string lookup
RESUME_POLICIES: dict[str, ResumePolicy] = {
    "always": AlwaysResumePolicy(),
    "never": NeverResumePolicy(),
    "status": StatusBasedPolicy(),
}


def get_resume_policy(policy: str | ResumePolicy) -> ResumePolicy:
    """Get resume policy instance from string or object.

    Args:
        policy: Policy name string or ResumePolicy instance

    Returns:
        ResumePolicy instance

    Raises:
        ValueError: If policy string is not recognized
    """
    if isinstance(policy, str):
        if policy not in RESUME_POLICIES:
            raise ValueError(
                f"Unknown resume policy: {policy}. "
                f"Available policies: {list(RESUME_POLICIES.keys())}"
            )
        return RESUME_POLICIES[policy]

    return policy
