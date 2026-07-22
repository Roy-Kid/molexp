"""Tests for :mod:`molexp.workflow.contract` value types.

The frozen-pydantic types (`WorkflowContract`, `TaskIO`, …) carry no molexp
behavior beyond pydantic's own frozen / extra-forbid config, so the sole
molexp-owned invariant here is that the baseline check set is *complete*:
:func:`default_validation_checks` must ship a default for every
:class:`ValidationCheckId` member (a new id without a default would be caught).
"""

from __future__ import annotations

from molexp.workflow.contract import (
    ValidationCheckId,
    default_validation_checks,
)


class TestDefaultValidationChecks:
    def test_covers_every_validation_check_id(self) -> None:
        defaults = default_validation_checks()
        assert {c.id for c in defaults} == set(ValidationCheckId)
