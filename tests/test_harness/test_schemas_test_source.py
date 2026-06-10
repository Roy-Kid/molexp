"""Tests for the ``TestSource`` schema (spec ``harness-run-mode-01-substrate``, T01).

RED before implementation: ``TestSource`` does not exist yet, so every
in-function ``from molexp.harness import TestSource`` fails. (``TestSource``
matches pytest's ``Test*`` collection pattern, so — exactly like the house
``test_schemas_test_spec.py`` does for ``TestSpec`` / ``TestResult`` — it is
imported inside test functions, never at module scope.)

After GREEN these assert:

- ``TestSource`` is a frozen pydantic model carrying a generated pytest
  module + derivation metadata, mirroring ``WorkflowSource``;
- ``symbols`` defaults to the empty tuple;
- unknown fields are rejected (``extra="forbid"``);
- ``"test_source"`` is a registered artifact kind AND a ``ValidationReport``
  target kind;
- the schema is re-exported from ``molexp.harness`` and
  ``molexp.harness.schemas``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

_TEST_SOURCE = (
    "from generated_workflow import build_workflow\n"
    "\n"
    "\n"
    "def test_ok():\n"
    "    assert callable(build_workflow)\n"
)


# ----------------------------------------------------------- schema shape


def test_test_source_schema_fields() -> None:
    from molexp.harness import TestSource

    ts = TestSource(
        source=_TEST_SOURCE,
        module_name="test_generated_workflow",
        test_spec_id="ts-001",
        bound_workflow_id="bw-x",
        symbols=("build_workflow",),
    )
    assert ts.source == _TEST_SOURCE
    assert ts.module_name == "test_generated_workflow"
    assert ts.test_spec_id == "ts-001"
    assert ts.bound_workflow_id == "bw-x"
    assert ts.symbols == ("build_workflow",)


def test_test_source_symbols_default_to_empty_tuple() -> None:
    from molexp.harness import TestSource

    ts = TestSource(
        source=_TEST_SOURCE,
        module_name="test_generated_workflow",
        test_spec_id="ts-001",
        bound_workflow_id="bw-x",
    )
    assert ts.symbols == ()


def test_test_source_is_frozen() -> None:
    from molexp.harness import TestSource

    ts = TestSource(
        source=_TEST_SOURCE,
        module_name="test_generated_workflow",
        test_spec_id="ts-001",
        bound_workflow_id="bw-x",
    )
    with pytest.raises(ValidationError):
        ts.source = "mutated"  # type: ignore[misc]


def test_test_source_forbids_extra_fields() -> None:
    from molexp.harness import TestSource

    with pytest.raises(ValidationError):
        TestSource(
            source=_TEST_SOURCE,
            module_name="test_generated_workflow",
            test_spec_id="ts-001",
            bound_workflow_id="bw-x",
            unexpected="boom",  # type: ignore[call-arg]
        )


def test_test_source_reexported_from_harness() -> None:
    from molexp.harness.schemas.test_source import TestSource as Canonical

    import molexp.harness as h
    from molexp.harness.schemas import TestSource as FromSchemas

    assert h.TestSource is Canonical
    assert FromSchemas is Canonical


# ------------------------------------------------- kind + target kind


def test_test_source_in_well_known_artifact_kinds() -> None:
    from molexp.harness import WELL_KNOWN_ARTIFACT_KINDS

    assert "test_source" in WELL_KNOWN_ARTIFACT_KINDS


def test_validation_report_accepts_test_source_target_kind() -> None:
    from molexp.harness import ValidationReport

    report = ValidationReport.from_violations(
        target_kind="test_source",
        target_id="x",
        violations=[],
    )
    assert report.target_kind == "test_source"
    assert report.target_id == "x"
    assert report.passed is True
