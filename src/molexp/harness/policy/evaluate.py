"""Pure evaluator for :class:`ApprovalPolicy` (Phase 6 §7.5).

Given an ``ApprovalPolicy`` + an optional ``BoundWorkflow`` (and
optionally its source ``WorkflowIR``), walks the workflow and emits one
:class:`ApprovalRequest` per triggered ``require_for_*`` clause.

Auto-trigger rules (each gated by the matching policy flag):

- ``hpc_submission`` — ``bw.execution_backend in {"slurm", "pbs", "lsf"}``.
- ``agent_inferred_scientific_parameters`` — **one per BoundTask** that
  has any ``ParameterValue.source == "agent_inferred"``.
- ``large_resource_request`` — ``bw.resource_policy.max_runtime_s > 86400``
  OR ``bw.resource_policy.max_memory_gb > 256`` (thresholds hard-coded
  in Phase 6; parameterize on ``ApprovalPolicy`` in a future phase).
- ``full_execution`` — emitted whenever the flag is True and ``bw`` is
  not None. The caller decides timing (before/after dry-run).
- ``overwrite`` — emitted when ``"overwrite" in bw.review_flags`` OR
  any ``TaskIR.review_flags`` (via the optional ``ir``) contains
  ``"overwrite"``. ``BoundTask`` has no ``review_flags`` field; the IR
  argument is how task-level overwrite hints reach this validator.

``final_report`` is **not** auto-triggered (no workflow signal).
Use :func:`make_final_report_approval_request` for that intent.

Deterministic ordering of the returned list:
``[hpc_submission, full_execution, large_resource_request, overwrite,
agent_inferred_scientific_parameters (per BoundTask in bw.tasks order)]``.

Request ``id`` is the first 12 hex chars of ``uuid.uuid4()``; ``created_at``
is set to the current UTC time at request construction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from molexp.harness.schemas.approval import ApprovalIntent, ApprovalRequest
from molexp.harness.schemas.bound_workflow import BoundWorkflow
from molexp.harness.schemas.policy import ApprovalPolicy
from molexp.harness.schemas.workflow_ir import WorkflowIR

__all__ = ["evaluate_approval_policy", "make_final_report_approval_request"]


_HPC_BACKENDS = frozenset({"slurm", "pbs", "lsf"})
_RUNTIME_THRESHOLD_S = 86_400  # 24 hours
_MEMORY_THRESHOLD_GB = 256.0


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


def _make_request(
    *,
    intent: ApprovalIntent,
    reason: str,
    triggered_by_policy: str,
    metadata: dict[str, object] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        id=_new_id(),
        intent=intent,
        reason=reason,
        triggered_by_policy=triggered_by_policy,
        metadata=dict(metadata or {}),
        created_at=_now(),
    )


def evaluate_approval_policy(
    policy: ApprovalPolicy,
    *,
    bw: BoundWorkflow | None = None,
    ir: WorkflowIR | None = None,
) -> list[ApprovalRequest]:
    """Walk ``policy`` + ``bw`` (+ optional ``ir``) and emit approval requests."""
    requests: list[ApprovalRequest] = []

    if bw is None:
        # Auto-triggers all need a BoundWorkflow to reason about.
        return requests

    # 1. hpc_submission
    if policy.require_for_hpc_submission and bw.execution_backend in _HPC_BACKENDS:
        requests.append(
            _make_request(
                intent="hpc_submission",
                reason=(
                    f"BoundWorkflow.execution_backend={bw.execution_backend!r} "
                    "is an HPC scheduler; submission requires approval"
                ),
                triggered_by_policy="require_for_hpc_submission",
                metadata={"execution_backend": bw.execution_backend},
            )
        )

    # 2. full_execution
    if policy.require_for_full_execution:
        requests.append(
            _make_request(
                intent="full_execution",
                reason="Full workflow execution requires approval per policy",
                triggered_by_policy="require_for_full_execution",
                metadata={"bound_workflow_id": bw.id},
            )
        )

    # 3. large_resource_request
    if policy.require_for_large_resource_request:
        rp = bw.resource_policy
        runtime_too_high = rp.max_runtime_s > _RUNTIME_THRESHOLD_S
        memory_too_high = rp.max_memory_gb is not None and rp.max_memory_gb > _MEMORY_THRESHOLD_GB
        if runtime_too_high or memory_too_high:
            requests.append(
                _make_request(
                    intent="large_resource_request",
                    reason=(
                        f"Resource policy exceeds threshold "
                        f"(runtime={rp.max_runtime_s}s, memory={rp.max_memory_gb}GB)"
                    ),
                    triggered_by_policy="require_for_large_resource_request",
                    metadata={
                        "max_runtime_s": rp.max_runtime_s,
                        "max_memory_gb": rp.max_memory_gb,
                    },
                )
            )

    # 4. overwrite (deduped across bw.review_flags + ir.tasks[*].review_flags)
    if policy.require_for_overwrite:
        bw_overwrite = "overwrite" in bw.review_flags
        ir_overwrite = ir is not None and any("overwrite" in t.review_flags for t in ir.tasks)
        if bw_overwrite or ir_overwrite:
            requests.append(
                _make_request(
                    intent="overwrite",
                    reason=(
                        "Workflow declares overwrite intent in review_flags "
                        "(bw or upstream IR task)"
                    ),
                    triggered_by_policy="require_for_overwrite",
                )
            )

    # 5. agent_inferred_scientific_parameters — one per BoundTask
    if policy.require_for_agent_inferred_scientific_parameters:
        for bt in bw.tasks:
            inferred_keys = [
                key for key, param in bt.parameters.items() if param.source == "agent_inferred"
            ]
            if inferred_keys:
                requests.append(
                    _make_request(
                        intent="agent_inferred_scientific_parameters",
                        reason=(
                            f"BoundTask {bt.id!r} has agent-inferred parameters: {inferred_keys}"
                        ),
                        triggered_by_policy="require_for_agent_inferred_scientific_parameters",
                        metadata={"bound_task_id": bt.id, "inferred_keys": inferred_keys},
                    )
                )

    return requests


def make_final_report_approval_request(
    policy: ApprovalPolicy,
) -> ApprovalRequest | None:
    """Emit the lone ``final_report`` intent when policy demands it.

    This intent has no workflow signal — the caller invokes this helper
    explicitly when generating the final report. Returns ``None`` if
    ``policy.require_for_final_report is False``.
    """
    if not policy.require_for_final_report:
        return None
    return _make_request(
        intent="final_report",
        reason="Final report publication requires approval per policy",
        triggered_by_policy="require_for_final_report",
    )
