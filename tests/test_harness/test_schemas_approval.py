"""Tests for ApprovalIntent / ApprovalRequest / ApprovalDecision (Phase 6 §7.5).

Locks:
- frozen pydantic round-trip
- ApprovalIntent carries exactly six values
- ApprovalRequest defaults (metadata={}, etc.)
- ApprovalDecision defaults (reason=None)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------- ApprovalIntent


def test_approval_intent_carries_six_values() -> None:
    from typing import Literal

    from molexp.harness.schemas.approval import ApprovalIntent

    assert get_origin(ApprovalIntent) is Literal
    expected = {
        "agent_inferred_scientific_parameters",
        "full_execution",
        "hpc_submission",
        "large_resource_request",
        "overwrite",
        "final_report",
    }
    assert set(get_args(ApprovalIntent)) == expected


# ---------------------------------------------------------- ApprovalRequest


def _make_request():
    from molexp.harness.schemas.approval import ApprovalRequest

    return ApprovalRequest(
        id="req-001",
        intent="hpc_submission",
        reason="Workflow targets slurm backend",
        triggered_by_policy="require_for_hpc_submission",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )


def test_approval_request_round_trip() -> None:
    from molexp.harness.schemas.approval import ApprovalRequest

    req = _make_request()
    dumped = req.model_dump_json()
    rehydrated = ApprovalRequest.model_validate_json(dumped)
    assert rehydrated == req


def test_approval_request_with_metadata_round_trip() -> None:
    from molexp.harness.schemas.approval import ApprovalRequest

    req = ApprovalRequest(
        id="req-002",
        intent="agent_inferred_scientific_parameters",
        reason="Task t1 has 2 agent-inferred parameters",
        triggered_by_policy="require_for_agent_inferred_scientific_parameters",
        metadata={"bound_task_id": "b1", "inferred_keys": ["n_chains", "rho"]},
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    dumped = req.model_dump_json()
    rehydrated = ApprovalRequest.model_validate_json(dumped)
    assert rehydrated == req


def test_approval_request_defaults() -> None:
    req = _make_request()
    assert req.metadata == {}


def test_approval_request_is_frozen() -> None:
    req = _make_request()
    with pytest.raises(ValidationError):
        req.intent = "full_execution"  # type: ignore[misc]


def test_approval_request_rejects_unknown_intent() -> None:
    from molexp.harness.schemas.approval import ApprovalRequest

    with pytest.raises(ValidationError):
        ApprovalRequest(
            id="x",
            intent="not_a_real_intent",  # type: ignore[arg-type]
            reason="x",
            triggered_by_policy="x",
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
        )


def test_approval_request_default_factory_independent() -> None:
    a = _make_request()
    b = _make_request()
    assert a.metadata is not b.metadata


# ---------------------------------------------------------- ApprovalDecision


def _make_decision(*, granted: bool = True):
    from molexp.harness.schemas.approval import ApprovalDecision

    return ApprovalDecision(
        request_id="req-001",
        granted=granted,
        decided_by="user:roy",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
        reason="Looks safe",
    )


def test_approval_decision_round_trip() -> None:
    from molexp.harness.schemas.approval import ApprovalDecision

    dec = _make_decision()
    dumped = dec.model_dump_json()
    rehydrated = ApprovalDecision.model_validate_json(dumped)
    assert rehydrated == dec


def test_approval_decision_defaults() -> None:
    from molexp.harness.schemas.approval import ApprovalDecision

    dec = ApprovalDecision(
        request_id="req-001",
        granted=False,
        decided_by="harness:auto",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    assert dec.reason is None


def test_approval_decision_is_frozen() -> None:
    dec = _make_decision()
    with pytest.raises(ValidationError):
        dec.granted = False  # type: ignore[misc]


# -------------------------------------------- re-exports


def test_three_approval_types_re_exported_from_schemas_package() -> None:
    from molexp.harness.schemas import (
        ApprovalDecision as via_pkg_dec,
    )
    from molexp.harness.schemas import (
        ApprovalIntent as via_pkg_intent,
    )
    from molexp.harness.schemas import (
        ApprovalRequest as via_pkg_req,
    )
    from molexp.harness.schemas.approval import (
        ApprovalDecision as via_mod_dec,
    )
    from molexp.harness.schemas.approval import (
        ApprovalIntent as via_mod_intent,
    )
    from molexp.harness.schemas.approval import (
        ApprovalRequest as via_mod_req,
    )

    assert via_pkg_intent is via_mod_intent
    assert via_pkg_req is via_mod_req
    assert via_pkg_dec is via_mod_dec


def test_three_approval_types_re_exported_from_top_level() -> None:
    from molexp.harness import (  # noqa: F401
        ApprovalDecision,
        ApprovalIntent,
        ApprovalRequest,
    )
