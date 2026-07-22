"""Scope-generalized ``ApprovalRequest`` schema (spec plan-emergent-07).

plan-emergent-07 adds a *resume-routing discriminator* to the approval schema:
``ApprovalRequest`` gains an explicit frozen ``scope`` (``approval_gate`` vs
``intervention_request``) plus a ``target_agent_id`` naming the phase-2 codegen
subagent, and ``ApprovalIntent`` gains ``"task_intervention"``. These are
typed fields — not ``metadata`` — because they drive deterministic resume
dispatch and must round-trip through the store's typed columns.

RED before GREEN: these pin the schema the production change must satisfy.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from molexp.harness.schemas import ApprovalRequest, ApprovalScope


def _created_at() -> datetime:
    return datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)


def _gate_request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        intent="experiment_spec",
        reason="approve the concrete experiment spec",
        triggered_by_policy="PlanMode",
        created_at=_created_at(),
    )


class TestApprovalRequestScope:
    def test_default_scope_is_approval_gate(self) -> None:
        # Backward-compat default: a phase-1 gate omits scope entirely.
        assert _gate_request().scope == "approval_gate"

    def test_default_target_agent_id_is_none(self) -> None:
        assert _gate_request().target_agent_id is None

    def test_construct_intervention_request(self) -> None:
        request = ApprovalRequest(
            id="req-int",
            intent="task_intervention",
            reason="loop stuck — need human guidance",
            triggered_by_policy="StepAuditLoop",
            created_at=_created_at(),
            scope="intervention_request",
            target_agent_id="codegen-task-T",
        )
        assert request.scope == "intervention_request"
        assert request.target_agent_id == "codegen-task-T"
        assert request.intent == "task_intervention"

    def test_scope_field_is_frozen(self) -> None:
        request = _gate_request()
        with pytest.raises(ValidationError):
            request.scope = "intervention_request"  # type: ignore[misc]

    def test_target_agent_id_field_is_frozen(self) -> None:
        request = ApprovalRequest(
            id="req-int",
            intent="task_intervention",
            reason="r",
            triggered_by_policy="p",
            created_at=_created_at(),
            scope="intervention_request",
            target_agent_id="codegen-task-T",
        )
        with pytest.raises(ValidationError):
            request.target_agent_id = "codegen-task-U"  # type: ignore[misc]

    def test_illegal_scope_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApprovalRequest(
                id="req-bad",
                intent="experiment_spec",
                reason="r",
                triggered_by_policy="p",
                created_at=_created_at(),
                scope="not-a-scope",  # type: ignore[arg-type]
            )

    def test_task_intervention_is_a_valid_intent(self) -> None:
        request = ApprovalRequest(
            id="req-int",
            intent="task_intervention",
            reason="r",
            triggered_by_policy="p",
            created_at=_created_at(),
        )
        assert request.intent == "task_intervention"


class TestApprovalScopeExport:
    def test_approval_scope_importable_from_schemas(self) -> None:
        # ApprovalScope is a first-class exported Literal alias.
        from molexp.harness import schemas

        assert "ApprovalScope" in schemas.__all__
        assert ApprovalScope is schemas.ApprovalScope
