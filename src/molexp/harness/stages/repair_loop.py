"""``RepairLoop`` — generate → validate → repair until the plan converges.

LLM generation is stochastic and imperfect: a malformed ``WorkflowIR``, an
off-signature binding, or a workflow that does not faithfully implement the
report (caught by ``ReviewPlan``) would otherwise fail the whole plan on a
single bad sample. ``RepairLoop`` wraps a generator stage and the validators
that gate its output: it runs them, and on any validation failure it records the
failure as a ``<output>_feedback`` artifact — which the generator reads on its
next attempt — and re-generates, up to ``attempts`` times. So the structural and
semantic validators *drive the plan to convergence* instead of merely rejecting
it. After the budget is exhausted it re-raises the last failure (never silently
passes a plan that did not validate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas import ArtifactRef

__all__ = ["RepairLoop"]

_LOG = get_logger(__name__)


class RepairLoop(Stage):
    """Run ``generate`` + ``validators``, regenerating with feedback on failure."""

    name: ClassVar[str] = "repair_loop"

    def __init__(
        self,
        *,
        name: str,
        generate: Stage,
        validators: list[Stage],
        feedback_kind: str,
        attempts: int = 3,
    ) -> None:
        if attempts < 1:
            raise ValueError("RepairLoop attempts must be >= 1")
        if not validators:
            raise ValueError("RepairLoop needs at least one validator")
        # Shadow the ClassVar on this instance so the Mode ledger keys on the
        # wrapped stage's name (mirrors ApprovalGate's instance naming).
        object.__setattr__(self, "name", name)
        self._generate = generate
        self._validators = list(validators)
        self._feedback_kind = feedback_kind
        self._attempts = attempts

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        last: StageExecutionError | None = None
        for attempt in range(1, self._attempts + 1):
            # The generated artifact is this stage's output; the validators are
            # gates whose own (report) artifacts stay in the store but are not
            # returned, so downstream stages resolve the real generated artifact.
            generated: ArtifactRef = await self._generate.run(ctx)
            try:
                for validator in self._validators:
                    await validator.run(ctx)
            except StageExecutionError as exc:
                last = exc
                self._record_feedback(ctx, exc)
                _LOG.warning(
                    f"[repair:{self.name}] attempt {attempt}/{self._attempts} failed "
                    f"validation; regenerating with feedback — {exc}"
                )
                continue
            return generated
        assert last is not None  # attempts >= 1, so the loop body ran at least once
        raise last

    def _record_feedback(self, ctx: HarnessRunContext, exc: StageExecutionError) -> None:
        """Persist the failure as a ``<output>_feedback`` artifact the generator
        reads on its next attempt (the validator already persisted the report)."""
        if isinstance(exc, StagePersistedFailureError):
            text = ctx.artifact_store.get(exc.persisted_ref.id).decode("utf-8")
        else:
            text = str(exc)
        ctx.artifact_store.put_text(
            kind=self._feedback_kind,
            text=text,
            created_by=self.name,
            parent_ids=[],
        )
