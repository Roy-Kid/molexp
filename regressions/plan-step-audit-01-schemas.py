"""Regression: ReviewPack + approve projection via public schemas API only."""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.harness.schemas import (
    FormDocument,
    ReviewDecision,
    ReviewPack,
    review_decision_to_approval,
)
from molexp.harness.schemas.step_audit import FormBooleanField, FormTextField


def main() -> int:
    pack = ReviewPack(
        pack_id="pack-reg-01",
        step_id="draft_spec",
        step_title="Draft spec",
        policy="hard",
        summary_md="minimal pack",
        form=FormDocument(
            title="Reg",
            fields=[
                FormTextField(id="name", label="Name", default="demo"),
                FormBooleanField(id="ok", label="OK", default=True),
            ],
        ),
        decision_options=["approve", "reject", "revise"],
    )
    restored = ReviewPack.model_validate(pack.model_dump(mode="json"))
    assert restored.pack_id == pack.pack_id
    assert len(restored.form.fields) == 2

    decision = ReviewDecision(
        pack_id=pack.pack_id,
        action="approve",
        decided_by="regression",
        decided_at=datetime(2026, 7, 19, tzinfo=UTC),
        reason="ok",
    )
    approval = review_decision_to_approval(decision, request_id="req-reg")
    assert approval.granted is True
    print("plan-step-audit-01-schemas: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
