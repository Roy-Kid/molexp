"""``analyze_run_failure`` — ordinary Run → sourced FailureAnalysis.

Shared by CLI, server, and (optionally) lifecycle tools so Python ≡ UI.
Deterministic narrative path needs **no** LLM: error.txt / metadata.error /
execution inventory. Optional ``narrative=`` overrides the template.

Default domain is ``failed`` only; ``cancelled`` requires ``force=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.workspace import harvest_run

if TYPE_CHECKING:
    from molexp.workspace.knowledge_item import KnowledgeItem
    from molexp.workspace.run import Run

__all__ = ["analyze_run_failure", "build_failure_narrative"]

_DEFAULT_NAME_PREFIX = "failure-analysis"
_MAX_ERROR_CHARS = 4000
_MAX_TAIL_LINES = 80


def build_failure_narrative(run: Run) -> str:
    """Build a non-empty deterministic FailureAnalysis narrative for *run*.

    Prefers ``executions/<last>/error.txt``, then run metadata error, then a
    status-only summary. Never returns empty string.
    """
    lines = [
        f"Run `{run.id}` finished with status `{run.status}`.",
        "",
        "## Evidence",
        "",
    ]
    error_text = _read_error_text(run)
    if error_text:
        lines.append("### error.txt / error metadata")
        lines.append("")
        lines.append("```")
        lines.append(error_text)
        lines.append("```")
        lines.append("")
    else:
        lines.append("(no error.txt or metadata.error on disk)")
        lines.append("")

    history = list(run.execution_history)
    if history:
        lines.append("### Executions")
        lines.append("")
        for rec in history:
            exec_id = getattr(rec, "execution_id", None) or "?"
            status = getattr(rec, "status", "?")
            lines.append(f"- `{exec_id}`: {status}")
        lines.append("")

    lines.append("## Resume")
    lines.append("")
    lines.append(
        "Use `resume` to reopen the last execution and recompute unfinished nodes, "
        "or `rerun` for a fresh attempt. Re-run analyze-failure after the next "
        "terminal failure to update this note."
    )
    return "\n".join(lines).rstrip() + "\n"


def analyze_run_failure(
    run: Run,
    *,
    created_by: str,
    narrative: str | None = None,
    force: bool = False,
    name: str | None = None,
) -> KnowledgeItem:
    """Write/update a FailureAnalysis KnowledgeItem for a failed *run*.

    Args:
        run: Workspace Run to interpret.
        created_by: Author string (``cli``, ``ui``, ``agent:…``).
        narrative: Optional override; when omitted a deterministic template is used.
        force: When True, also accept ``cancelled``; default is failed-only.
        name: Explicit KnowledgeItem name; default ``failure-analysis-{run.id}``.

    Returns:
        The written :class:`~molexp.workspace.knowledge_item.KnowledgeItem`.

    Raises:
        ValueError: Status domain refusal or empty effective narrative.
    """
    status = run.status
    if status == "failed" or (force and status == "cancelled"):
        pass
    else:
        raise ValueError(
            f"run {run.id} is {status!r} — analyze_run_failure requires status "
            f"'failed' (or 'cancelled' with force=True)"
        )

    text = (narrative or "").strip() or build_failure_narrative(run)
    item_name = name or f"{_DEFAULT_NAME_PREFIX}-{run.id}"
    return harvest_run(
        run,
        kind="FailureAnalysis",
        narrative=text,
        created_by=created_by,
        name=item_name,
    )


def _read_error_text(run: Run) -> str:
    """Best-effort error body from last execution error.txt or metadata."""
    chunks: list[str] = []
    history = list(run.execution_history)
    if history:
        last = history[-1]
        exec_id = getattr(last, "execution_id", None)
        if exec_id:
            path = Path(run.run_dir) / "executions" / str(exec_id) / "error.txt"
            if path.is_file():
                raw = path.read_text(encoding="utf-8", errors="replace")
                chunks.append(_clip(raw))
    # Fallback: scan executions/*/error.txt when history is empty but files exist.
    if not chunks:
        exec_root = Path(run.run_dir) / "executions"
        if exec_root.is_dir():
            for err_path in sorted(exec_root.glob("*/error.txt")):
                chunks.append(_clip(err_path.read_text(encoding="utf-8", errors="replace")))
    meta_error = getattr(run.metadata, "error", None)
    if meta_error:
        chunks.append(_clip(str(meta_error)))
    # Dedup if both channels hold the same string.
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return "\n---\n".join(unique)


def _clip(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > _MAX_TAIL_LINES:
        text = "\n".join(lines[-_MAX_TAIL_LINES:])
        text = f"… ({len(lines) - _MAX_TAIL_LINES} earlier lines omitted)\n{text}"
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + f"\n… (+{len(text) - _MAX_ERROR_CHARS} chars omitted)"
    return text
