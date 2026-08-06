"""Deterministic :class:`ReviewPack` builders for PlanMode step audits.

No LLM: packs are assembled from the latest subject artifacts so the audit
trail always has a structured form even when the gateway is offline.
"""

from __future__ import annotations

import json

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError
from molexp.harness.plan import ExperimentPlan
from molexp.harness.schemas import (
    FormDocument,
    FormField,
    FormMarkdownField,
    FormTextField,
    PlanArtifactRef,
    ReviewPack,
)
from molexp.harness.schemas.step_audit import FormBooleanField

__all__ = [
    "build_experiment_plan_review_pack",
    "build_experiment_spec_review_pack",
    "build_plan_review_pack",
]


def _require_kind(
    ctx: HarnessRunContext, kind: str, *, stage: str
) -> tuple[PlanArtifactRef, bytes]:
    ref = ctx.artifact_store.latest_by_kind(kind)
    if ref is None:
        raise StageExecutionError(f"{stage}: missing required artifact kind {kind!r}")
    return ref, ctx.artifact_store.get(ref.id)


def build_experiment_spec_review_pack(ctx: HarnessRunContext) -> ReviewPack:
    """Build a hard-review pack for the concrete ``experiment_spec`` artifact."""
    ref, raw = _require_kind(ctx, "experiment_spec", stage="build_experiment_spec_review_pack")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = {}
    title = str(data.get("title") or data.get("id") or ref.id)
    objective = str(data.get("objective") or "")
    summary = f"## Experiment spec\n\n**{title}**\n\n{objective}".strip()
    fields: list[FormField] = [
        FormMarkdownField(
            id="spec_summary",
            label="Specification summary",
            content=summary,
            readonly=True,
        ),
        FormTextField(
            id="operator_notes",
            label="Operator notes",
            help="Optional notes recorded with the approval decision.",
            required=False,
        ),
        FormBooleanField(
            id="confirm_parameters",
            label="I reviewed the resolved parameters",
            required=True,
            default=False,
        ),
    ]
    return ReviewPack(
        pack_id=f"pack-spec-{ref.id}",
        step_id="draft_spec",
        step_title="Draft spec",
        policy="hard",
        summary_md=summary,
        subject_refs=[ref],
        form=FormDocument(title=title, description_md=objective or None, fields=fields),
        decision_options=["approve", "reject", "revise"],
        audit_hints=[
            "Confirm every agent-inferred parameter before capability discovery.",
            "Reject to regenerate the spec; revise to pin field values and re-audit.",
        ],
    )


def _clean_plan_display_text(raw: str, *, max_len: int = 240) -> str:
    """Strip revision wrappers / multi-copy noise for human review UI.

    Plan drafts sometimes carry ``Original request:… Revise the previous plan…``
    templates; those must never appear as the form title or description.
    """
    text = " ".join((raw or "").split()).strip()
    if not text:
        return ""
    # Drop common revision scaffolding prefixes (keep the first human goal).
    for marker in (
        "Revise the previous plan",
        "Original request:",
        "using this feedback:",
    ):
        if marker.lower() in text.lower():
            # Prefer the segment before the first revision boilerplate.
            lower = text.lower()
            idx = lower.find("revise the previous plan")
            if idx > 0:
                text = text[:idx].strip()
            idx = lower.find("original request:")
            if idx == 0:
                rest = text[len("Original request:") :].strip()
                # Cut again if revise appears after
                ridx = rest.lower().find("revise the previous plan")
                text = rest[:ridx].strip() if ridx > 0 else rest
            break
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def build_experiment_plan_review_pack(ctx: HarnessRunContext) -> ReviewPack:
    """Build a hard-review pack for the pre-approval ``experiment_plan``.

    Anchors on the **pre-approval** ``experiment_plan`` artifact — the frozen
    plan is a post-approval consequence and does not exist yet at gate-render,
    so this reads the mutable plan (spec + task board) via ``_require_kind`` and
    renders a compact board checklist the operator confirms before freeze.

    Args:
        ctx: The harness run context whose artifact store holds the plan.

    Returns:
        A hard-policy :class:`ReviewPack` over the ``experiment_plan`` artifact.

    Raises:
        StageExecutionError: If no ``experiment_plan`` artifact is present.
    """
    ref, raw = _require_kind(ctx, "experiment_plan", stage="build_experiment_plan_review_pack")
    plan = ExperimentPlan.model_validate(json.loads(raw))
    raw_title = str(plan.spec.get("title") or plan.spec.get("id") or "Experiment plan")
    raw_objective = str(plan.spec.get("objective") or "")
    title = _clean_plan_display_text(raw_title, max_len=120) or "Experiment plan"
    objective = _clean_plan_display_text(raw_objective, max_len=280)

    from molexp.harness.plan.document import render_experiment_plan_document

    # Full 12-section experiment plan book for audit / preview (not form fields).
    summary = render_experiment_plan_document(plan, title_fallback=title)

    # Prior knowledge (close-loop-01): surface the assembled digest so the human
    # gate can see which workspace knowledge shaped this plan.
    knowledge_ref = ctx.artifact_store.latest_by_kind("knowledge_context")
    subject_refs: list[PlanArtifactRef] = [ref]
    if knowledge_ref is not None:
        try:
            digest = ctx.artifact_store.get(knowledge_ref.id).decode("utf-8").strip()
        except Exception:
            digest = ""
        if digest:
            # Cap the pack surface — full digest stays on the artifact.
            max_digest_chars = 4000
            clipped = (
                digest
                if len(digest) <= max_digest_chars
                else (digest[:max_digest_chars] + "\n… (digest truncated in review pack)")
            )
            summary = (
                f"{summary.rstrip()}\n\n"
                f"## Prior knowledge\n\n"
                f"_Artifact `{knowledge_ref.id}` (kind=knowledge_context)_\n\n"
                f"{clipped}\n"
            )
            subject_refs.append(knowledge_ref)

    # Operator form: comment only — Priority / keep_tasks / confirm checkbox removed.
    fields: list[FormField] = [
        FormTextField(
            id="operator_notes",
            label="Comment",
            required=False,
            placeholder="Optional comment for this decision…",
        ),
    ]
    return ReviewPack(
        pack_id=f"pack-experiment-plan-{ref.id}",
        step_id="review_plan",
        step_title="Review experiment plan",
        policy="hard",
        summary_md=summary,
        subject_refs=subject_refs,
        form=FormDocument(title=title, description_md=objective or None, fields=fields),
        decision_options=["approve", "reject", "revise"],
        audit_hints=[
            "Approve freezes the board; revise re-opens planning with your comment.",
            "Prior knowledge section lists workspace FailureAnalysis/Finding digests that grounded planning.",
        ],
    )


def build_plan_review_pack(ctx: HarnessRunContext) -> ReviewPack:
    """Build a hard-review pack for the verified plan (step 8).

    Anchors on the latest compile ``execution_result`` when present, otherwise
    ``workflow_source`` — both prove the plan was materialised before final
    review.
    """
    subject = ctx.artifact_store.latest_by_kind("execution_result")
    if subject is None:
        subject = ctx.artifact_store.latest_by_kind("workflow_source")
    if subject is None:
        raise StageExecutionError(
            "build_plan_review_pack: need execution_result or workflow_source subject"
        )

    extras: list[PlanArtifactRef] = [subject]
    for kind in (
        "experiment_spec",
        "workflow_source",
        "test_spec",
        "input_set",
        "execution_result",
    ):
        ref = ctx.artifact_store.latest_by_kind(kind)
        if ref is not None and ref.id not in {r.id for r in extras}:
            extras.append(ref)

    lines = ["## Plan review", "", f"Primary subject: `{subject.kind}` (`{subject.id}`)", ""]
    for ref in extras:
        lines.append(f"- `{ref.kind}`: `{ref.id}`")
    summary = "\n".join(lines)

    fields: list[FormField] = [
        FormMarkdownField(
            id="plan_inventory",
            label="Plan artifacts",
            content=summary,
            readonly=True,
        ),
        FormTextField(
            id="operator_notes",
            label="Operator notes",
            required=False,
        ),
        FormBooleanField(
            id="confirm_plan",
            label="I reviewed the verified plan (source, tests, dry-run)",
            required=True,
            default=False,
        ),
    ]
    return ReviewPack(
        pack_id=f"pack-plan-{subject.id}",
        step_id="review",
        step_title="Review",
        policy="hard",
        summary_md=summary,
        subject_refs=list(extras),
        form=FormDocument(title="Verified plan", fields=fields),
        decision_options=["approve", "reject", "revise"],
        audit_hints=[
            "Approve only after dry-run and per-task tests look correct.",
            "Reject blocks finalisation; revise re-enters the audit loop with field_values.",
        ],
    )
