"""Tests for ``ExpectedOutput`` (harness workflow-IR schema).

Regression (spec 01 janitor finding #1b): opening ``ArtifactKind = str`` must
not lose the non-empty constraint the former ``Literal[...]`` enforced —
``ExpectedOutput.kind`` pins ``min_length=1`` to match ``PlanArtifactRef.kind``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.harness.schemas.workflow_ir import ExpectedOutput


class TestExpectedOutput:
    def test_rejects_empty_kind(self) -> None:
        """``ExpectedOutput(kind="")`` is rejected by the ``min_length=1`` constraint."""
        with pytest.raises(ValidationError):
            ExpectedOutput(name="x", kind="", description="x")
