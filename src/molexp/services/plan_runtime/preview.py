"""``render_approval_preview`` — what an approval gate asks the operator to read.

The ONE gate-time preview renderer shared by the CLI's interactive approver
and the server's approvals inbox (Python = UI): plain text derived from the
suspended run's persisted artifacts, so the operator reviews the actual gated
content instead of approving blind.

Intents map to the plan gates: ``experiment_spec`` (pre-compile — the concrete
spec fields), ``final_report`` / anything else (the whole-plan review — the
generated workflow source plus the compile/dry-run verdict).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.workspace.run import Run

__all__ = ["render_approval_preview"]

_SOURCE_PREVIEW_LINES = 120


def render_approval_preview(run: Run, intent: str) -> str:
    """Render the text an operator should review before deciding *intent*.

    Best-effort by design: the preview is a convenience view over artifacts
    that already passed validation — a missing artifact renders as a stated
    absence, never an exception (the decision itself stays fully functional).
    """
    store = _store(run)
    if intent == "experiment_spec":
        return _spec_preview(store)
    return _plan_preview(store)


def _store(run: Run) -> FileArtifactStore:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")


def _spec_preview(store: FileArtifactStore) -> str:
    """The concrete experiment_spec the pre-compile gate approves."""
    ref = store.latest_by_kind("experiment_spec")
    if ref is None:
        return "(no experiment_spec artifact found)"
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
    return "\n".join(lines)


def _plan_preview(store: FileArtifactStore) -> str:
    """The whole-plan review: generated source + compile/test verdict."""
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
    if source_ref is None:
        lines.append("workflow source    : (not generated)")
        return "\n".join(lines)
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
    return "\n".join(lines)
