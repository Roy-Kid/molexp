"""Step-audit form/review schemas — expert review packs for PlanMode gates.

Pure data (``frozen=True``). Used by :class:`~molexp.harness.stages.step_audit_loop.StepAuditLoop`
and services/CLI/UI consumers. Binary :class:`~molexp.harness.schemas.approval.ApprovalDecision`
remains the store-facing grant shape; :class:`ReviewDecision` is the three-action
expert surface with projection helpers for approve/reject only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molexp.harness.schemas.approval import ApprovalDecision
from molexp.harness.schemas.artifact import PlanArtifactRef

__all__ = [
    "FindingResolution",
    "FormArtifactRefField",
    "FormBooleanField",
    "FormDocument",
    "FormField",
    "FormKeyValueField",
    "FormMarkdownField",
    "FormMultiSelectField",
    "FormNumberField",
    "FormSelectField",
    "FormSelectOption",
    "FormTableColumn",
    "FormTableField",
    "FormTextAreaField",
    "FormTextField",
    "ReviewDecision",
    "ReviewFinding",
    "ReviewPack",
    "StepAuditPolicy",
    "approval_decision_to_review",
    "review_decision_to_approval",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

StepAuditPolicy = Literal["auto", "soft", "hard"]
"""How a step-audit loop treats human involvement.

* ``auto`` — synthetic approve; no suspend.
* ``soft`` — pack recorded; pipeline continues without blocking.
* ``hard`` — store-first suspend until a decision lands.
"""

ReviewAction = Literal["approve", "reject", "revise"]
FieldSeverity = Literal["info", "warning", "error"]
FindingSeverity = Literal["error", "warning", "info"]
FindingResolutionStatus = Literal["accepted", "dismissed", "fixed"]


class _FormFieldBase(BaseModel):
    """Shared shell for every :data:`FormField` variant."""

    model_config = _FROZEN

    id: str
    label: str
    help: str = ""
    required: bool = False
    readonly: bool = False
    severity: FieldSeverity | None = None
    group: str | None = None


class FormSelectOption(BaseModel):
    """One choice in a select / multi_select field."""

    model_config = _FROZEN

    value: str
    label: str
    description: str | None = None


class FormTableColumn(BaseModel):
    """Column definition for a table field."""

    model_config = _FROZEN

    id: str
    label: str
    value_kind: str | None = None


class FormTextField(_FormFieldBase):
    """Single-line text input."""

    kind: Literal["text"] = "text"
    placeholder: str | None = None
    default: str | None = None
    min_length: int | None = None
    max_length: int | None = None


class FormTextAreaField(_FormFieldBase):
    """Multi-line text input."""

    kind: Literal["textarea"] = "textarea"
    placeholder: str | None = None
    default: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    rows: int | None = None


class FormNumberField(_FormFieldBase):
    """Numeric input."""

    kind: Literal["number"] = "number"
    default: float | int | None = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    unit: str | None = None


class FormBooleanField(_FormFieldBase):
    """Boolean toggle."""

    kind: Literal["boolean"] = "boolean"
    default: bool | None = None


class FormSelectField(_FormFieldBase):
    """Single-select options."""

    kind: Literal["select"] = "select"
    options: list[FormSelectOption]
    default: str | None = None


class FormMultiSelectField(_FormFieldBase):
    """Multi-select options."""

    kind: Literal["multi_select"] = "multi_select"
    options: list[FormSelectOption]
    default: list[str] | None = None


class FormTableField(_FormFieldBase):
    """Tabular rows with typed columns."""

    kind: Literal["table"] = "table"
    columns: list[FormTableColumn]
    default_rows: list[dict[str, Any]] | None = None
    min_rows: int | None = None
    max_rows: int | None = None


class FormKeyValueField(_FormFieldBase):
    """List of key/value pairs."""

    kind: Literal["key_value"] = "key_value"
    key_label: str = "key"
    value_label: str = "value"
    default: list[dict[str, str]] | None = None


class FormMarkdownField(_FormFieldBase):
    """Display-only markdown body."""

    kind: Literal["markdown"] = "markdown"
    content: str


class FormArtifactRefField(_FormFieldBase):
    """Reference to a harness artifact by id."""

    kind: Literal["artifact_ref"] = "artifact_ref"
    allowed_kinds: list[str] | None = None
    default: str | None = None


FormField = Annotated[
    FormTextField
    | FormTextAreaField
    | FormNumberField
    | FormBooleanField
    | FormSelectField
    | FormMultiSelectField
    | FormTableField
    | FormKeyValueField
    | FormMarkdownField
    | FormArtifactRefField,
    Field(discriminator="kind"),
]
"""Discriminated union of form field variants (discriminator ``kind``)."""


class FormDocument(BaseModel):
    """Structured form a UI or CLI can render without guessing field shapes."""

    model_config = _FROZEN

    title: str | None = None
    description_md: str | None = None
    fields: list[FormField] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_field_ids(self) -> FormDocument:
        seen: set[str] = set()
        for field in self.fields:
            if field.id in seen:
                raise ValueError(f"duplicate FormField id: {field.id!r}")
            seen.add(field.id)
        return self


class ReviewFinding(BaseModel):
    """One pre-audit finding attached to a :class:`ReviewPack`."""

    model_config = _FROZEN

    finding_id: str
    severity: FindingSeverity
    summary: str
    detail: str = ""
    field_id: str | None = None


class FindingResolution(BaseModel):
    """How an expert resolved one :class:`ReviewFinding`."""

    model_config = _FROZEN

    finding_id: str
    status: FindingResolutionStatus
    note: str | None = None


class ReviewPack(BaseModel):
    """One step's review package — the single UI/CLI presentation unit."""

    model_config = _FROZEN

    pack_id: str
    step_id: str
    step_title: str
    policy: StepAuditPolicy
    summary_md: str
    subject_refs: list[PlanArtifactRef] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    form: FormDocument = Field(default_factory=FormDocument)
    decision_options: list[ReviewAction] = Field(min_length=1)
    audit_hints: list[str] | None = None


class ReviewDecision(BaseModel):
    """Expert (or auto) answer to a :class:`ReviewPack`."""

    model_config = _FROZEN

    pack_id: str
    action: ReviewAction
    decided_by: str
    decided_at: datetime
    reason: str | None = None
    field_values: dict[str, Any] = Field(default_factory=dict)
    edits: dict[str, Any] | None = None
    finding_resolutions: list[FindingResolution] | None = None


def review_decision_to_approval(
    decision: ReviewDecision,
    *,
    request_id: str,
) -> ApprovalDecision:
    """Project approve/reject onto binary :class:`ApprovalDecision`.

    Args:
        decision: Three-action review decision.
        request_id: Approval request id for the store/event row.

    Returns:
        Binary approval decision with matching identity fields.

    Raises:
        ValueError: If ``decision.action`` is ``revise`` (no binary mapping).
    """
    if decision.action == "revise":
        raise ValueError(
            "review_decision_to_approval cannot map action='revise'; "
            "persist ReviewDecision and re-enter StepAuditLoop instead"
        )
    return ApprovalDecision(
        request_id=request_id,
        granted=decision.action == "approve",
        decided_by=decision.decided_by,
        decided_at=decision.decided_at,
        reason=decision.reason,
    )


def approval_decision_to_review(
    decision: ApprovalDecision,
    *,
    pack_id: str,
    field_values: dict[str, Any] | None = None,
) -> ReviewDecision:
    """Lift a binary approval decision into a :class:`ReviewDecision`.

    Args:
        decision: Store-facing grant/deny row.
        pack_id: Pack this decision answers.
        field_values: Optional form values (default empty).

    Returns:
        ReviewDecision with action approve or reject.
    """
    return ReviewDecision(
        pack_id=pack_id,
        action="approve" if decision.granted else "reject",
        decided_by=decision.decided_by,
        decided_at=decision.decided_at,
        reason=decision.reason,
        field_values=dict(field_values or {}),
    )
