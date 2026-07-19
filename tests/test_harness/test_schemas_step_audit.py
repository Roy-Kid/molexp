"""Unit tests for harness step-audit form/review schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from molexp.harness.schemas import (
    ApprovalDecision,
    FormDocument,
    FormField,
    ReviewDecision,
    ReviewPack,
    StepAuditPolicy,
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


def test_form_field_ten_kinds_round_trip() -> None:
    adapter: TypeAdapter[FormField] = TypeAdapter(FormField)
    for field in _FIELD_FIXTURES:
        dumped = field.model_dump(mode="json")
        restored = adapter.validate_python(dumped)
        assert restored.kind == field.kind
        assert restored.id == field.id
        assert restored.model_dump(mode="json") == dumped


def test_form_document_round_trip_all_kinds() -> None:
    doc = FormDocument(title="t", description_md="d", fields=list(_FIELD_FIXTURES))
    restored = FormDocument.model_validate(doc.model_dump(mode="json"))
    assert len(restored.fields) == 10
    assert [f.kind for f in restored.fields] == [f.kind for f in _FIELD_FIXTURES]


def test_unknown_form_field_kind_raises() -> None:
    adapter: TypeAdapter[FormField] = TypeAdapter(FormField)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "wizard", "id": "x", "label": "X"})


def test_form_field_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        FormTextField.model_validate({"kind": "text", "id": "x", "label": "X", "surprise": True})


def test_form_document_duplicate_field_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate FormField id"):
        FormDocument(
            fields=[
                FormTextField(id="same", label="A"),
                FormBooleanField(id="same", label="B"),
            ]
        )


@pytest.mark.parametrize("policy", ["auto", "soft", "hard"])
def test_review_pack_accepts_policies(policy: StepAuditPolicy) -> None:
    pack = ReviewPack(
        pack_id="p1",
        step_id="draft_spec",
        step_title="Draft spec",
        policy=policy,
        summary_md="summary",
        form=FormDocument(fields=[FormTextField(id="t", label="T")]),
        decision_options=["approve", "reject", "revise"],
        findings=[
            ReviewFinding(finding_id="f1", severity="warning", summary="check me"),
        ],
        audit_hints=["hint"],
    )
    restored = ReviewPack.model_validate(pack.model_dump(mode="json"))
    assert restored.policy == policy
    assert restored.findings[0].finding_id == "f1"


def test_review_pack_empty_decision_options_raises() -> None:
    with pytest.raises(ValidationError):
        ReviewPack(
            pack_id="p1",
            step_id="s",
            step_title="S",
            policy="hard",
            summary_md="",
            decision_options=[],
        )


def test_review_pack_illegal_policy_raises() -> None:
    with pytest.raises(ValidationError):
        ReviewPack.model_validate(
            {
                "pack_id": "p1",
                "step_id": "s",
                "step_title": "S",
                "policy": "elevated",
                "summary_md": "",
                "decision_options": ["approve"],
            }
        )


def test_review_decision_approve_reject_project() -> None:
    when = _now()
    approve = ReviewDecision(
        pack_id="p1",
        action="approve",
        decided_by="op",
        decided_at=when,
        reason="ok",
    )
    approval = review_decision_to_approval(approve, request_id="req-1")
    assert approval.granted is True
    assert approval.request_id == "req-1"
    assert approval.decided_by == "op"
    assert approval.decided_at == when
    assert approval.reason == "ok"

    reject = ReviewDecision(
        pack_id="p1",
        action="reject",
        decided_by="op",
        decided_at=when,
        reason="no",
    )
    denied = review_decision_to_approval(reject, request_id="req-1")
    assert denied.granted is False


def test_review_decision_revise_has_no_binary_map() -> None:
    decision = ReviewDecision(
        pack_id="p1",
        action="revise",
        decided_by="op",
        decided_at=_now(),
        field_values={"t": "new"},
    )
    with pytest.raises(ValueError, match="revise"):
        review_decision_to_approval(decision, request_id="req-1")


def test_illegal_review_action_raises() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "pack_id": "p1",
                "action": "defer",
                "decided_by": "op",
                "decided_at": _now().isoformat(),
            }
        )


def test_approval_decision_binary_unchanged() -> None:
    d = ApprovalDecision(
        request_id="r",
        granted=True,
        decided_by="x",
        decided_at=_now(),
        reason=None,
    )
    assert d.granted is True


def test_bidirectional_approve_reject_projection() -> None:
    when = _now()
    for action in ("approve", "reject"):
        review = ReviewDecision(
            pack_id="pack",
            action=action,  # type: ignore[arg-type]
            decided_by="alice",
            decided_at=when,
            reason="why",
        )
        approval = review_decision_to_approval(review, request_id="req")
        back = approval_decision_to_review(approval, pack_id="pack")
        assert back.action == action
        assert back.decided_by == "alice"
        assert back.decided_at == when
        assert back.reason == "why"
        assert back.pack_id == "pack"
