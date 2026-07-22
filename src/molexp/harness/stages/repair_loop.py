"""``RepairLoop`` — generate → validate → diagnose → evidence → repair.

LLM generation is stochastic: a malformed IR, off-signature binding, or
workflow that fails structural checks would otherwise fail the whole plan on
one bad sample. ``RepairLoop`` wraps a generator and its validators, and on
failure:

1. records the failure as ``feedback_kind`` for the next attempt;
2. diagnoses the failure (capability gaps vs retryable LLM errors);
3. on a **hard capability gap** (confirmed missing molpy API / SYMBOL_NOT_FOUND),
   raises :class:`~molexp.harness.errors.CapabilityGapError` immediately —
   do not burn attempts inventing missing symbols. An *unknown*
   ``capability_id`` from a live catalog is **retryable** (re-bind);
4. otherwise enriches feedback with molmcp evidence when symbols are present
   and re-generates, up to ``attempts`` times (default 3).

Raising the attempt budget is not a quality strategy; fix diagnosis / evidence
instead of lottery-retrying.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.harness.core.stage import Stage
from molexp.harness.errors import (
    CapabilityGapError,
    StageExecutionError,
    StagePersistedFailureError,
)
from molexp.harness.stages.evidence import collect_evidence_text, diagnose_failure

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas import PlanArtifactRef

__all__ = ["RepairLoop"]

_LOG = get_logger(__name__)


class RepairLoop(Stage):
    """Run ``generate`` + ``validators``, regenerating with evidence on failure."""

    name: ClassVar[str] = "repair_loop"

    def __init__(
        self,
        *,
        name: str,
        generate: Stage,
        validators: list[Stage],
        feedback_kind: str,
        attempts: int = 3,
        hard_stop_gaps: bool = True,
        collect_evidence: bool = True,
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
        self._hard_stop_gaps = hard_stop_gaps
        self._collect_evidence = collect_evidence

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        last: StageExecutionError | None = None
        loop_t0 = time.monotonic()
        for attempt in range(1, self._attempts + 1):
            # The generated artifact is this stage's output; the validators are
            # gates whose own (report) artifacts stay in the store but are not
            # returned, so downstream stages resolve the real generated artifact.
            attempt_t0 = time.monotonic()
            _LOG.info(f"[repair:{self.name}] attempt {attempt}/{self._attempts} start")
            generated: PlanArtifactRef = await self._generate.run(ctx)
            try:
                for validator in self._validators:
                    await validator.run(ctx)
            except CapabilityGapError:
                raise  # already diagnosed; do not burn remaining attempts
            except StageExecutionError as exc:
                gap = self._record_feedback(ctx, exc)
                if gap is not None:
                    raise gap from exc
                last = exc
                _LOG.warning(
                    f"[repair:{self.name}] attempt {attempt}/{self._attempts} failed "
                    f"validation duration_s={time.monotonic() - attempt_t0:.2f}; "
                    f"regenerating with feedback — {exc}"
                )
                continue
            _LOG.info(
                f"[repair:{self.name}] attempt {attempt}/{self._attempts} ok "
                f"duration_s={time.monotonic() - attempt_t0:.2f} "
                f"total_s={time.monotonic() - loop_t0:.2f}"
            )
            return generated
        assert last is not None  # attempts >= 1, so the loop body ran at least once
        _LOG.error(
            f"[repair:{self.name}] exhausted attempts={self._attempts} "
            f"total_s={time.monotonic() - loop_t0:.2f}"
        )
        raise last

    def _record_feedback(
        self, ctx: HarnessRunContext, exc: StageExecutionError
    ) -> CapabilityGapError | None:
        """Persist failure + diagnosis (+ optional molmcp evidence) as feedback.

        Returns a :class:`CapabilityGapError` when the failure is a hard gap
        (caller must raise it); otherwise ``None`` and the loop may retry.
        """
        if isinstance(exc, StagePersistedFailureError):
            text = ctx.artifact_store.get(exc.persisted_ref.id).decode("utf-8")
        else:
            text = str(exc)

        diagnosis = diagnose_failure(exc, text)
        # Always gather evidence first — confirmed symbol misses may promote
        # to capability gaps (api_symbol_missing).
        if self._collect_evidence and (diagnosis.symbols or diagnosis.codes or diagnosis.gaps):
            evidence = collect_evidence_text(diagnosis, ctx=ctx)
            body = f"{text}\n\n{evidence}"
        else:
            body = text

        ctx.artifact_store.put_text(
            kind=self._feedback_kind,
            text=body,
            created_by=self.name,
            parent_ids=[],
        )

        if self._hard_stop_gaps and diagnosis.is_capability_gap:
            gap = diagnosis.gaps[0]
            return CapabilityGapError(
                gap.user_message(),
                code=gap.code,
                symbol=gap.symbol,
            )
        return None
