"""ReviewPack builders + markdown projection for plan approval gates.

Shared by the CLI interactive approver and the server's approvals inbox
(Python = UI). Structured packs come from the latest ``review_pack`` artifact
when StepAuditLoop has already written one; otherwise a best-effort pack is
synthesised from legacy artifact contents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.harness.schemas import FormDocument, ReviewPack
from molexp.harness.schemas.step_audit import FormMarkdownField, FormTextField

if TYPE_CHECKING:
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.workspace.run import Run

__all__ = [
    "build_review_pack",
    "render_approval_preview",
    "render_review_pack",
]

_SOURCE_PREVIEW_LINES = 120


def build_review_pack(run: Run, intent: str) -> ReviewPack:
    """Return a :class:`ReviewPack` for the gated content of *intent*.

    Prefer the latest ``review_pack`` artifact written by StepAuditLoop; fall
    back to a synthesised pack from legacy previews. Never raises on missing
    artifacts.
    """
    store = _store(run)
    ref = store.latest_by_kind("review_pack")
    if ref is not None:
        try:
            return ReviewPack.model_validate_json(store.get(ref.id))
        except Exception:
            pass
    if intent == "experiment_spec":
        return _synthetic_spec_pack(store)
    return _synthetic_plan_pack(store)


def render_review_pack(pack: ReviewPack) -> str:
    """Project a :class:`ReviewPack` to operator-readable markdown/plain text."""
    lines: list[str] = []
    if pack.step_title:
        lines.append(f"# {pack.step_title}")
        lines.append("")
    if pack.summary_md:
        lines.append(pack.summary_md.rstrip())
        lines.append("")
    if pack.findings:
        lines.append("## Findings")
        for finding in pack.findings:
            lines.append(f"- [{finding.severity}] {finding.summary}")
        lines.append("")
    if pack.form.fields:
        lines.append("## Form")
        for field in pack.form.fields:
            if field.kind == "markdown":
                lines.append(f"### {field.label}")
                lines.append(field.content)
            else:
                lines.append(f"- {field.label} (`{field.id}`, {field.kind})")
        lines.append("")
    if pack.decision_options:
        lines.append("Options: " + ", ".join(pack.decision_options))
    if pack.audit_hints:
        lines.append("")
        lines.append("## Audit hints")
        for hint in pack.audit_hints:
            lines.append(f"- {hint}")
    text = "\n".join(lines).strip()
    return text or "(empty review pack)"


def render_approval_preview(run: Run, intent: str) -> str:
    """Render the text an operator should review before deciding *intent*.

    Thin wrapper: :func:`render_review_pack` over :func:`build_review_pack`.
    Best-effort — missing artifacts yield a stated absence, never an exception.
    """
    try:
        return render_review_pack(build_review_pack(run, intent))
    except Exception:
        return "(preview unavailable)"


def _store(run: Run) -> FileArtifactStore:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")


def _synthetic_spec_pack(store: FileArtifactStore) -> ReviewPack:
    ref = store.latest_by_kind("experiment_spec")
    if ref is None:
        body = "(no experiment_spec artifact found)"
        refs = []
    else:
        try:
            spec = json.loads(store.get(ref.id))
            lines = [
                f"title     : {spec.get('title')}",
                f"objective : {spec.get('objective')}",
            ]
            for variable in spec.get("variables", []):
                value = (variable.get("value") or {}).get("value")
                unit = variable.get("unit") or ""
                lines.append(f"variable  : {variable.get('name')} = {value} {unit}".rstrip())
            for question in spec.get("resolved_questions", []):
                lines.append(f"resolved  : {question.get('question')} -> {question.get('answer')}")
            body = "\n".join(lines)
        except Exception:
            body = "(experiment_spec unreadable)"
        refs = [ref]
    return ReviewPack(
        pack_id=f"pack-spec-synth-{ref.id if ref else 'none'}",
        step_id="draft_spec",
        step_title="Draft spec",
        policy="hard",
        summary_md=body,
        subject_refs=refs,
        form=FormDocument(
            fields=[
                FormMarkdownField(id="spec", label="Spec", content=body, readonly=True),
                FormTextField(id="operator_notes", label="Operator notes"),
            ]
        ),
        decision_options=["approve", "reject", "revise"],
    )


def _synthetic_plan_pack(store: FileArtifactStore) -> ReviewPack:
    lines: list[str] = []
    dry = store.latest_by_kind("execution_result")
    tests = store.latest_by_kind("test_result")
    lines.append(f"compiled / dry-ran : {dry is not None}")
    if tests is not None:
        try:
            status = json.loads(store.get(tests.id)).get("status", "unknown")
        except Exception:
            status = "unreadable"
        lines.append(f"generated tests    : {status}")
    source_ref = store.latest_by_kind("workflow_source")
    refs = [r for r in (dry, tests, source_ref) if r is not None]
    if source_ref is None:
        lines.append("workflow source    : (not generated)")
    else:
        try:
            source = json.loads(store.get(source_ref.id)).get("source", "")
        except Exception:
            source = ""
        lines.append("workflow source    :")
        lines.append("")
        src_lines = source.splitlines()
        lines.extend(src_lines[:_SOURCE_PREVIEW_LINES])
        if len(src_lines) > _SOURCE_PREVIEW_LINES:
            lines.append(f"… (+{len(src_lines) - _SOURCE_PREVIEW_LINES} more lines)")
    body = "\n".join(lines)
    anchor = dry or source_ref
    return ReviewPack(
        pack_id=f"pack-plan-synth-{anchor.id if anchor else 'none'}",
        step_id="review",
        step_title="Review",
        policy="hard",
        summary_md=body,
        subject_refs=refs,
        form=FormDocument(
            fields=[
                FormMarkdownField(id="plan", label="Plan", content=body, readonly=True),
                FormTextField(id="operator_notes", label="Operator notes"),
            ]
        ),
        decision_options=["approve", "reject", "revise"],
    )
