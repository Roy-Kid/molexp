"""Tests for ParameterValue + ParameterSource (Phase 1 schema layer).

Locks the wire format per spec §4.3:
- frozen pydantic round-trip
- ParameterSource is the 7-value Literal alias from harness-goal.md §1.6
- unknown source raises ValidationError
- optional fields default to None / False
- approved defaults False
"""

from __future__ import annotations

from typing import get_args, get_origin

import pytest
from pydantic import ValidationError


def test_parameter_value_round_trip() -> None:
    from molexp.harness.schemas.parameter import ParameterValue

    pv = ParameterValue(
        value=1e6,
        source="literature_default",
        reason="LAMMPS NEMD example",
        confidence=0.85,
        citation="doi:10.1234/lammps-nemd",
        approved=True,
    )
    dumped = pv.model_dump_json()
    rehydrated = ParameterValue.model_validate_json(dumped)
    assert rehydrated == pv


def test_parameter_value_is_frozen() -> None:
    from molexp.harness.schemas.parameter import ParameterValue

    pv = ParameterValue(value=1, source="user_provided")
    with pytest.raises(ValidationError):
        pv.value = 2  # type: ignore[misc]


def test_parameter_value_optional_defaults() -> None:
    from molexp.harness.schemas.parameter import ParameterValue

    pv = ParameterValue(value="hello", source="user_provided")
    assert pv.reason is None
    assert pv.confidence is None
    assert pv.citation is None
    assert pv.approved is False


def test_parameter_value_rejects_unknown_source() -> None:
    from molexp.harness.schemas.parameter import ParameterValue

    with pytest.raises(ValidationError):
        ParameterValue(
            value=1,
            source="vibes",  # type: ignore[arg-type]
        )


def test_parameter_value_accepts_any_value_type() -> None:
    """The value field is intentionally untyped (Any) — harness-goal §4.3."""
    from molexp.harness.schemas.parameter import ParameterValue

    for v in [None, True, 0, 1.5, "x", [1, 2, 3], {"k": "v"}]:
        ParameterValue(value=v, source="user_provided")


def test_parameter_source_is_literal_not_enum() -> None:
    from typing import Literal

    from molexp.harness.schemas import parameter as parameter_mod

    assert get_origin(parameter_mod.ParameterSource) is Literal
    import enum

    for name in dir(parameter_mod):
        obj = getattr(parameter_mod, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            pytest.fail(f"schemas/parameter.py must not define enum: {name}")


def test_parameter_source_seven_value_set() -> None:
    """ParameterSource ships the 7-value set from harness-goal.md §1.6."""
    from molexp.harness.schemas.parameter import ParameterSource

    expected = {
        "user_provided",
        "agent_inferred",
        "project_default",
        "package_default",
        "literature_default",
        "manual_override",
        "runtime_detected",
    }
    assert set(get_args(ParameterSource)) == expected
