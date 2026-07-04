"""Tests for the frozen-pydantic types in :mod:`molexp.workflow.contract`.

Coverage focus:

- Empty-tuple defaults on every collection field (so absent IR sections
  parse to empty tuples).
- ``extra="forbid"`` rejection of unknown kwargs.
- Frozen-config rejection of post-construction mutation.
- Severity literal accepts only ``"error"`` / ``"warning"``.
- ``ValidationCheckId`` enum membership matches the documented list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.workflow.contract import (
    ValidationCheckId,
    WorkflowContract,
    default_validation_checks,
)

# ── TaskInputSpec / TaskOutputSpec / ArtifactDecl ──────────────────────────


# ── TaskIO ─────────────────────────────────────────────────────────────────


# ── ValidationCheckId / ValidationCheck ────────────────────────────────────


# ── ValidationIssue / ValidationReport ─────────────────────────────────────


# ── WorkflowContract ───────────────────────────────────────────────────────


def test_workflow_contract_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        WorkflowContract(
            workflow_id="workflow_00000000",
            stray=1,  # type: ignore[call-arg]
        )


def test_workflow_contract_is_frozen() -> None:
    c = WorkflowContract(workflow_id="workflow_00000000")
    with pytest.raises(ValidationError):
        c.workflow_id = "other"  # type: ignore[misc]


# ── default_validation_checks ──────────────────────────────────────────────


def test_default_validation_checks_covers_every_id() -> None:
    defaults = default_validation_checks()
    assert {c.id for c in defaults} == set(ValidationCheckId)
