"""Persistent review queue for agent/task approvals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

REVIEWS_DIR_NAME = "reviews"
METADATA_FILE = "metadata.json"

ReviewKind = Literal["plan", "patch", "permission", "run_submission", "dangerous_action"]
ReviewStatus = Literal["pending", "approved", "rejected", "expired"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PersistedReviewItem:
    review_id: str
    kind: ReviewKind
    title: str
    status: ReviewStatus
    target_type: str
    target_id: str
    created_at: str
    task_id: str | None = None
    session_id: str | None = None
    description: str | None = None
    risk_level: RiskLevel = "medium"
    resolved_at: str | None = None
    resolution_comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def reviews_dir(workspace_root: str | Path) -> Path:
    path = Path(workspace_root) / REVIEWS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_review_metadata(workspace_root: str | Path) -> list[PersistedReviewItem]:
    root = reviews_dir(workspace_root)
    rows: list[PersistedReviewItem] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / METADATA_FILE
        if not meta_path.exists():
            continue
        item = read_review_metadata(workspace_root, entry.name)
        if item is not None:
            rows.append(item)
    rows.sort(key=lambda r: r.resolved_at or r.created_at, reverse=True)
    return rows


def read_review_metadata(
    workspace_root: str | Path,
    review_id: str,
) -> PersistedReviewItem | None:
    meta_path = Path(workspace_root) / REVIEWS_DIR_NAME / review_id / METADATA_FILE
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    metadata = raw.get("metadata")
    return PersistedReviewItem(
        review_id=str(raw.get("review_id") or review_id),
        kind=_coerce_kind(raw.get("kind")),
        title=str(raw.get("title") or "Untitled review"),
        status=_coerce_status(raw.get("status")),
        target_type=str(raw.get("target_type") or ""),
        target_id=str(raw.get("target_id") or ""),
        created_at=str(raw.get("created_at") or _now_iso()),
        task_id=raw.get("task_id") if isinstance(raw.get("task_id"), str) else None,
        session_id=raw.get("session_id") if isinstance(raw.get("session_id"), str) else None,
        description=raw.get("description") if isinstance(raw.get("description"), str) else None,
        risk_level=_coerce_risk(raw.get("risk_level")),
        resolved_at=raw.get("resolved_at") if isinstance(raw.get("resolved_at"), str) else None,
        resolution_comment=(
            raw.get("resolution_comment")
            if isinstance(raw.get("resolution_comment"), str)
            else None
        ),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def write_review_metadata(
    workspace_root: str | Path,
    item: PersistedReviewItem,
) -> None:
    target_dir = Path(workspace_root) / REVIEWS_DIR_NAME / item.review_id
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "review_id": item.review_id,
        "kind": item.kind,
        "title": item.title,
        "description": item.description,
        "risk_level": item.risk_level,
        "status": item.status,
        "target_type": item.target_type,
        "target_id": item.target_id,
        "task_id": item.task_id,
        "session_id": item.session_id,
        "created_at": item.created_at,
        "resolved_at": item.resolved_at,
        "resolution_comment": item.resolution_comment,
        "metadata": item.metadata,
    }
    path = target_dir / METADATA_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def plan_review_id(task_id: str, request_id: str) -> str:
    return f"review-plan-{_safe_id(task_id)}-{_safe_id(request_id)}"


def ensure_plan_review(
    workspace_root: str | Path,
    *,
    task_id: str,
    session_id: str,
    task_title: str,
    request_id: str,
    plan_markdown: str,
    workflow_preview: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> PersistedReviewItem:
    review_id = plan_review_id(task_id, request_id)
    existing = read_review_metadata(workspace_root, review_id)
    if existing is not None and existing.status != "pending":
        return existing
    item = PersistedReviewItem(
        review_id=review_id,
        kind="plan",
        title=f"Plan: {task_title}",
        description=_first_line(plan_markdown),
        risk_level="medium",
        status="pending",
        target_type="plan",
        target_id=request_id,
        task_id=task_id,
        session_id=session_id,
        created_at=created_at or _now_iso(),
        metadata={
            "plan_markdown": plan_markdown,
            "workflow_preview": workflow_preview or {},
        },
    )
    write_review_metadata(workspace_root, item)
    return item


def resolve_review(
    workspace_root: str | Path,
    item: PersistedReviewItem,
    *,
    status: Literal["approved", "rejected"],
    comment: str = "",
) -> PersistedReviewItem:
    resolved = PersistedReviewItem(
        review_id=item.review_id,
        kind=item.kind,
        title=item.title,
        description=item.description,
        risk_level=item.risk_level,
        status=status,
        target_type=item.target_type,
        target_id=item.target_id,
        task_id=item.task_id,
        session_id=item.session_id,
        created_at=item.created_at,
        resolved_at=_now_iso(),
        resolution_comment=comment or None,
        metadata=item.metadata,
    )
    write_review_metadata(workspace_root, resolved)
    return resolved


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)


def _coerce_kind(value: Any) -> ReviewKind:
    if value in {"plan", "patch", "permission", "run_submission", "dangerous_action"}:
        return value
    return "plan"


def _coerce_status(value: Any) -> ReviewStatus:
    if value in {"pending", "approved", "rejected", "expired"}:
        return value
    return "pending"


def _coerce_risk(value: Any) -> RiskLevel:
    if value in {"low", "medium", "high"}:
        return value
    return "medium"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
