"""Tests for ExpectedOutput (Phase 3 §4.6).

Locks the regression pinned by spec 01 janitor finding #1b: opening
``ArtifactKind = str`` must not lose the non-empty constraint that the
former ``Literal[...]`` enforced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_expected_output_rejects_empty_kind() -> None:
    """ac-008 — ``ExpectedOutput(kind="")`` is rejected by the
    ``min_length=1`` constraint, consistent with ``PlanArtifactRef.kind``.

    Spec 01 janitor finding #1b: opening ``ArtifactKind = str`` lost
    the implicit non-empty constraint that ``Literal[...]`` enforced;
    ``PlanArtifactRef.kind`` already pins ``min_length=1`` and
    ``ExpectedOutput.kind`` now matches.
    """
    from molexp.harness.schemas.workflow_ir import ExpectedOutput

    with pytest.raises(ValidationError):
        ExpectedOutput(name="x", kind="", description="x")
