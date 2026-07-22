"""Tests for step-audit form/review schemas (``molexp.harness.schemas.step_audit``)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from molexp.harness.schemas import (
    FormDocument,
    FormField,
    ReviewDecision,
    ReviewPack,
    approval_decision_to_review,
    review_decision_to_approval,
)
from molexp.harness.schemas.step_audit import (
    FormArtifactRefField,
    FormBooleanField,
    FormKeyValueField,
    FormMarkdownField,
    FormMultiSelectField,
    FormNumberField,
    FormSelectField,
    FormSelectOption,
    FormTableColumn,
    FormTableField,
    FormTextAreaField,
    FormTextField,
    ReviewFinding,
)


def _now() -> datetime:
    return datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)


# One instance of every FormField variant — the discriminated union must cover
# all ten (the table IS the point: each kind is a distinct model in the union).
_FIELD_FIXTURES: list[FormField] = [
    FormTextField(id="t", label="Text", default="x"),
    FormTextAreaField(id="ta", label="Area", rows=3),
    FormNumberField(id="n", label="Num", default=1.5, unit="K"),
    FormBooleanField(id="b", label="Flag", default=True),
    FormSelectField(
        id="s",
        label="Pick",
        options=[FormSelectOption(value="a", label="A")],
        default="a",
    ),
    FormMultiSelectField(
        id="ms",
        label="Multi",
        options=[FormSelectOption(value="a", label="A")],
        default=["a"],
    ),
    FormTableField(
        id="tbl",
        label="Table",
        columns=[FormTableColumn(id="c1", label="C1")],
        default_rows=[{"c1": "v"}],
    ),
    FormKeyValueField(id="kv", label="KV", default=[{"key": "k", "value": "v"}]),
    FormMarkdownField(id="md", label="MD", content="**hi**", readonly=True),
    FormArtifactRefField(id="ar", label="Art", default="art-1", allowed_kinds=["log"]),
]


class TestFormField:
    def test_all_ten_kinds_round_trip_through_discriminated_union(self) -> None:
        adapter: TypeAdapter[FormField] = TypeAdapter(FormField)
        assert len(_FIELD_FIXTURES) == 10
        for field in _FIELD_FIXTURES:
            dumped = field.model_dump(mode="json")
            restored = adapter.validate_python(dumped)
            assert restored.kind == field.kind
            assert restored.id == field.id
            assert restored.model_dump(mode="json") == dumped

    def test_unknown_kind_rejected(self) -> None:
        adapter: TypeAdapter[FormField] = TypeAdapter(FormField)
        with pytest.raises(ValidationError):
            adapter.validate_python({"kind": "wizard", "id": "x", "label": "X"})


class TestFormDocument:
    def test_duplicate_field_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate FormField id"):
            FormDocument(
                fields=[
                    FormTextField(id="same", label="A"),
                    FormBooleanField(id="same", label="B"),
                ]
            )


class TestReviewPack:
    def test_round_trips_with_findings(self) -> None:
        pack = ReviewPack(
            pack_id="p1",
            step_id="draft_spec",
            step_title="Draft spec",
            policy="hard",
            summary_md="summary",
            form=FormDocument(fields=[FormTextField(id="t", label="T")]),
            decision_options=["approve", "reject", "revise"],
            findings=[ReviewFinding(finding_id="f1", severity="warning", summary="check me")],
            audit_hints=["hint"],
        )
        restored = ReviewPack.model_validate(pack.model_dump(mode="json"))
        assert restored.policy == "hard"
        assert restored.findings[0].finding_id == "f1"

    def test_empty_decision_options_rejected(self) -> None:
        """A pack must offer at least one action (``min_length=1``)."""
        with pytest.raises(ValidationError):
            ReviewPack(
                pack_id="p1",
                step_id="s",
                step_title="S",
                policy="hard",
                summary_md="",
                decision_options=[],
            )


class TestReviewDecisionProjection:
    def test_approve_and_reject_round_trip_through_approval(self) -> None:
        """review → approval → review preserves identity; action maps to ``granted``."""
        when = _now()
        for action, granted in (("approve", True), ("reject", False)):
            review = ReviewDecision(
                pack_id="pack",
                action=action,  # type: ignore[arg-type]
                decided_by="alice",
                decided_at=when,
                reason="why",
            )
            approval = review_decision_to_approval(review, request_id="req")
            assert approval.granted is granted
            assert approval.request_id == "req"
            assert approval.decided_by == "alice"
            assert approval.decided_at == when
            assert approval.reason == "why"

            back = approval_decision_to_review(approval, pack_id="pack")
            assert back.action == action
            assert back.decided_by == "alice"
            assert back.decided_at == when
            assert back.reason == "why"
            assert back.pack_id == "pack"

    def test_revise_action_has_no_binary_mapping(self) -> None:
        decision = ReviewDecision(
            pack_id="p1",
            action="revise",
            decided_by="op",
            decided_at=_now(),
            field_values={"t": "new"},
        )
        with pytest.raises(ValueError, match="revise"):
            review_decision_to_approval(decision, request_id="req-1")
