"""Surface a PlanMode run on the Agents + Knowledge tabs.

Five raising writers make the AI activity visible in the UI — the agent-task
entry + session transcript (Agents tab), and three typed ``KnowledgeItem``
records mounted under the experiment (Knowledge graph): the ``Decision``
experiment record, the ``Finding`` harvested from an execute tail's
``final_report``, and the ``FailureAnalysis`` for a terminally-failed plan.

Each writer RAISES on failure — with ONE deliberate exception: a failed
provenance ``cite`` edge is warn-logged, never fatal, because the meta + body
already written must not be lost over a missing link. Everything else raises. The
per-record catch lives in :func:`materialize_plan_records`, which collects
failures into a structured ``PlanRecordOutcome`` (one broken record never
aborts its siblings, and the caller decides how loudly to surface it). The
record layer stays a projection: record errors never flip the plan's own
exit code, but they are always visible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.workspace.edges import EdgeRole
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.folder import Folder
    from molexp.workspace.knowledge_item import KnowledgeItem
    from molexp.workspace.run import Run

__all__ = [
    "emit_artifact_stage_events",
    "has_artifact",
    "write_agent_task_record",
    "write_experiment_record",
    "write_failure_analysis_record",
    "write_finding_record",
    "write_plan_task_status",
    "write_session_events_record",
]

_LOG = get_logger(__name__)


def write_agent_task_record(
    *,
    run: Run,
    workspace_root: str,
    task_id: str,
    draft: str,
    failed: bool = False,
) -> None:
    """Write the Agents-tab task entry (status ``failed`` for a failed plan)."""
    report = _read_artifact_json(run, "experiment_report")
    title = _title(report, draft, run.id)
    _write_agent_task(
        workspace_root, task_id=task_id, title=title, draft=draft, run=run, failed=failed
    )


def write_plan_task_status(
    workspace_root: str,
    *,
    task_id: str,
    draft: str,
    created_at: str,
    status: str,
    active_plan_task_id: str | None = None,
    turn_id: str | None = None,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Sync an IN-FLIGHT plan task's coarse status into the agent-task store.

    Called at every ``PlanTask`` lifecycle transition (launch → ``running``,
    gate suspension → ``waiting_approval``, …) so the Agents hub lists the
    task while it runs — previously a plan was invisible there until it
    finished. Terminal states are still written by
    :func:`write_agent_task_record` (which upgrades the title from the
    generated report). Best-effort: a store failure must not break the run.

    Scope ids (``project_id`` / ``experiment_id`` / ``run_id``) are the plan
    run's mount — pass them from the live ``PlanTask`` so the Agents hub can
    deep-link without waiting for terminal materialize.
    """
    from molexp.services.agent_task_store import (
        PersistedAgentTask,
        read_agent_task_metadata,
        write_agent_task_metadata,
    )

    title = draft.strip().splitlines()[0][:80] if draft.strip() else task_id
    try:
        current = read_agent_task_metadata(workspace_root, task_id)
        write_agent_task_metadata(
            workspace_root,
            PersistedAgentTask(
                task_id=task_id,
                session_id=current.session_id if current is not None else task_id,
                title=current.title if current is not None else title,
                goal=current.goal if current is not None else draft,
                status=status,
                created_at=current.created_at if current is not None else created_at,
                updated_at=datetime.now(UTC).isoformat(),
                plan_mode=True,
                active_mode="plan",
                active_turn_id=turn_id,
                active_plan_task_id=active_plan_task_id,
                pending_plan_draft=current.pending_plan_draft if current is not None else None,
                skill_id=current.skill_id if current is not None else None,
                project_id=(
                    project_id
                    if project_id is not None
                    else (current.project_id if current is not None else None)
                ),
                experiment_id=(
                    experiment_id
                    if experiment_id is not None
                    else (current.experiment_id if current is not None else None)
                ),
                run_id=(
                    run_id
                    if run_id is not None
                    else (current.run_id if current is not None else None)
                ),
            ),
        )
    except Exception as exc:  # the record is a convenience view, never load-bearing
        _LOG.warning(f"[plan-task {task_id}] status sync failed: {exc!r}")


def write_session_events_record(
    *,
    run: Run,
    experiment: Experiment,
    workspace_root: str,
    task_id: str,
    draft: str,
    turn_id: str | None = None,
    failure_stage: str | None = None,
    failure_error: str | None = None,
) -> None:
    """Write the synthesized session transcript the Agents session view renders.

    On failure pass ``failure_error`` (and optional ``failure_stage``) so the
    chat ends with an ``error`` + failed ``loop_completed`` carrying the real
    reason — not a green "plan ready" summary.
    """
    report = _read_artifact_json(run, "experiment_report")
    _write_session_events(
        workspace_root,
        task_id=task_id,
        run=run,
        experiment=experiment,
        draft=draft,
        report=report,
        turn_id=turn_id,
        failure_stage=failure_stage,
        failure_error=failure_error,
    )


# ── agent-task session ───────────────────────────────────────────────────────


def _write_agent_task(
    workspace_root: str,
    *,
    task_id: str,
    title: str,
    draft: str,
    run: Run,
    failed: bool = False,
) -> None:
    from molexp.services.agent_task_store import (
        PersistedAgentTask,
        read_agent_task_metadata,
        write_agent_task_metadata,
    )

    created = _created_at(run)
    current = read_agent_task_metadata(workspace_root, task_id)
    is_parent_chat_task = (
        current is not None
        and current.active_plan_task_id is not None
        and current.task_id != current.active_plan_task_id
    )
    if failed:
        status = "failed"
    else:
        status = run.status if run.status in {"succeeded", "failed"} else "completed"
    # Scope is the plan Run's experiment — always authoritative (chat parent
    # may have a broader mount; the plan itself is experiment-scoped).
    project_id = run.experiment.project.id
    experiment_id = run.experiment.id
    write_agent_task_metadata(
        workspace_root,
        PersistedAgentTask(
            task_id=task_id,
            session_id=current.session_id if current is not None else task_id,
            title=current.title if is_parent_chat_task else title,
            goal=current.goal if is_parent_chat_task else draft,
            status=status,
            created_at=current.created_at if current is not None else created,
            updated_at=datetime.now(UTC).isoformat(),
            plan_mode=True,
            active_mode="plan",
            active_turn_id=current.active_turn_id if current is not None else None,
            active_plan_task_id=(current.active_plan_task_id if current is not None else None),
            pending_plan_draft=None,
            skill_id=current.skill_id if current is not None else None,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run.id,
        ),
    )


# PlanOrchestrator stages (artifact kind → step label). Mirrors UI planStages.ts.
_STAGE_LABELS: list[tuple[str, str]] = [
    ("experiment_plan", "Built the task board"),
    ("review_pack", "Opened the review gate"),
    ("analysis_result", "Recorded the plan review"),
    ("frozen_experiment_plan", "Froze the experiment plan"),
    ("plan_report", "Rendered the plan report"),
    ("experiment_spec", "Materialized the experiment spec"),
    ("bound_workflow", "Bound tasks for realization"),
    ("workflow_source", "Generated workflow source"),
    ("test_source", "Generated per-task tests"),
    ("execution_result", "Compiled the workflow"),
    ("intervention_request", "Requested realization intervention"),
    # optional execute tail:
    ("final_report", "Executed the workflow & wrote the final report"),
    ("audit_report", "Generated the audit report"),
]


def _write_session_events(
    workspace_root: str,
    *,
    task_id: str,
    run: Run,
    experiment: Experiment,
    draft: str,
    report: dict[str, Any] | None,
    turn_id: str | None = None,
    failure_stage: str | None = None,
    failure_error: str | None = None,
) -> None:
    """Write a synthesized session transcript for the Agents *session view*.

    The transcript is deliberately lean: one step per PlanMode stage plus a short
    final summary. The full deliverables — spec, every task, and the runnable
    source — are NOT crammed into the chat answer; the session view's Deliverables
    panel fetches them structurally from ``GET /plans/{run_id}``. The terminal
    ``loop_completed`` carries a ``plan`` locator so the panel knows which plan to
    open.

    Failure path: still lists completed stages, then emits a typed ``error``
    event and a failed ``loop_completed`` whose text is the human-readable
    reason (pytest output / stage message) — never a green "plan ready".
    """
    from molexp.services.agent_task_store import append_agent_task_events, read_agent_task_events

    ts = _created_at(run)
    kinds = set(_artifact_kinds(run))
    event_context = {"turn_id": turn_id, "mode": "plan"}
    existing = read_agent_task_events(workspace_root, task_id)
    has_started = any(
        event.get("type") == "loop_started"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("turn_id") == turn_id
        for event in existing
    )

    def _payload(event: dict[str, Any]) -> dict[str, Any]:
        p = event.get("payload")
        return p if isinstance(p, dict) else {}

    def _same_turn(event: dict[str, Any]) -> bool:
        return _payload(event).get("turn_id") == turn_id

    has_failed_terminal = any(
        event.get("type") == "loop_completed"
        and _same_turn(event)
        and _payload(event).get("failed") is True
        for event in existing
    )
    has_success_terminal = any(
        event.get("type") == "loop_completed"
        and _same_turn(event)
        and _payload(event).get("failed") is not True
        for event in existing
    )
    events: list[dict[str, Any]] = []
    if not has_started:
        events.append(
            {
                "type": "loop_started",
                "ts": ts,
                "payload": {"user_input": draft, **event_context},
            }
        )
    for kind, label in _STAGE_LABELS:
        if kind not in kinds:
            continue
        already = False
        for event in existing:
            if event.get("type") != "tool_call_completed":
                continue
            result = _payload(event).get("result")
            if isinstance(result, dict) and result.get("artifact") == kind:
                already = True
                break
        if already:
            continue
        events.append(
            {
                "type": "tool_call_completed",
                "ts": ts,
                "payload": {
                    "tool_name": label,
                    "result": {"artifact": kind},
                    **event_context,
                },
            }
        )

    tasks = _read_workflow_tasks(experiment)
    source = _read_workflow_source(run)
    project_id = experiment.project.id if hasattr(experiment, "project") else ""
    title = _title(report, draft, run.id)
    plan_ref = {
        "run_id": run.id,
        "project_id": project_id,
        "experiment_id": experiment.id,
        "title": title,
        "step_count": len(tasks),
        "has_workflow": bool(source and source.strip())
        or ("workflow_source" in kinds)
        or ("workflow_ir" in kinds),
    }

    if failure_error is not None and not has_failed_terminal:
        detail = _failure_detail(run, failure_error)
        stage = failure_stage or _infer_failure_stage(kinds)
        events.append(
            {
                "type": "error",
                "ts": ts,
                "payload": {
                    "message": failure_error.strip() or "plan failed",
                    "stage": stage,
                    "detail": detail,
                    **event_context,
                },
            }
        )
        events.append(
            {
                "type": "loop_completed",
                "ts": ts,
                "payload": {
                    "text": _failure_summary(title, stage, failure_error, detail),
                    "failed": True,
                    "error": failure_error.strip() or "plan failed",
                    "stage": stage,
                    "plan": plan_ref,
                    **event_context,
                },
            }
        )
    elif (
        failure_error is None
        and report is not None
        and not has_success_terminal
        and not has_failed_terminal
    ):
        # Locator the Deliverables panel uses to fetch the structured plan
        # (`GET /projects/{p}/experiments/{e}/plans/{run_id}`). Carried on the
        # open `payload` so no schema / OpenAPI surface has to change.
        events.append(
            {
                "type": "loop_completed",
                "ts": ts,
                "payload": {
                    "text": _summary(title, tasks, source),
                    "plan": plan_ref,
                    **event_context,
                },
            }
        )
    if events:
        append_agent_task_events(workspace_root, task_id, events)


def _summary(title: str, tasks: list[str], source: str | None) -> str:
    """A short chat-answer summary; the full content lives in the panel."""
    did = ["drafted the experiment spec"]
    if tasks:
        did.append(f"bound {len(tasks)} workflow task{'s' if len(tasks) != 1 else ''}")
    if source and source.strip():
        did.append("generated the runnable workflow source")
    return (
        f"**{title}** — experiment plan ready.\n\n"
        f"PlanMode {', '.join(did)}.\n\n"
        "Open the **Deliverables** panel to review the spec, plan, and workflow script."
    )


def _failure_summary(title: str, stage: str | None, error: str, detail: str | None) -> str:
    """Human-readable failed-turn answer for the Agents chat."""
    stage_bit = f" at stage `{stage}`" if stage else ""
    lines = [
        f"**{title}** — plan failed{stage_bit}.",
        "",
        "## Error",
        "",
        (error.strip() or "(no error text)"),
    ]
    if detail and detail.strip() and detail.strip() not in error:
        lines += ["", "## Detail", "", "```", detail.strip()[:4000], "```"]
    lines += [
        "",
        "Partial deliverables (spec / workflow) may still be available in the "
        "**Deliverables** panel. Re-run the same draft to resume from the stage ledger.",
    ]
    return "\n".join(lines)


def _failure_detail(run: Run, _error: str) -> str | None:
    """Best-effort pytest / stage stderr to surface in chat (not just Knowledge)."""
    # Prefer the harness feedback artifact left by ExecuteTests for the repair loop.
    for kind in ("test_code_feedback", "stdout", "stderr"):
        text = _read_artifact_text(run, kind)
        if text and text.strip():
            return text.strip()
    # Fall back to the execution error.txt if present.
    try:
        err_path = Path(run.run_dir) / "executions"
        if err_path.is_dir():
            candidates = sorted(err_path.glob("*/error.txt"), key=lambda p: p.stat().st_mtime)
            if candidates:
                body = candidates[-1].read_text(encoding="utf-8", errors="replace")
                if body.strip():
                    return body.strip()
    except OSError:
        pass
    return None


def _read_artifact_text(run: Run, kind: str) -> str | None:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    root = Path(run.run_dir) / "artifacts"
    store = FileArtifactStore(root=root)
    ref = store.latest_by_kind(kind)
    if ref is None:
        return None
    try:
        raw = store.get(ref.id)
    except Exception:
        return None
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")
    return str(raw) if raw is not None else None


def _infer_failure_stage(kinds: set[str]) -> str | None:
    """Guess the failed stage from what is already on disk (when caller omitted it)."""
    if "workflow_source" in kinds and "execution_result" not in kinds:
        return "execute_tests"
    if "experiment_spec" in kinds and "workflow_ir" not in kinds:
        return "extract_workflow_ir"
    if "experiment_report" in kinds and "experiment_spec" not in kinds:
        return "generate_experiment_spec"
    return None


def _artifact_kinds(run: Run) -> list[str]:
    index_dir = Path(run.run_dir) / "artifacts" / "_index"
    if not index_dir.is_dir():
        return []
    return sorted(p.stem for p in index_dir.glob("*.json"))


def emit_artifact_stage_events(
    workspace_root: str,
    task_id: str,
    run: Run,
    *,
    turn_id: str | None = None,
    mode: str = "plan",
) -> int:
    """Append ``tool_call_completed`` rows for each on-disk plan artifact kind.

    The Agents progress rail only advances on these events (not ``stage_started``).
    Call at the review gate and after realization so the UI does not stay frozen
    on empty circles while artifacts already exist under ``run/artifacts/``.

    Returns:
        Number of new events written (0 when all kinds already recorded).
    """
    from molexp.services.agent_task_store import (
        append_agent_task_events,
        read_agent_task_events,
    )

    kinds = _artifact_kinds(run)
    if not kinds:
        return 0
    existing = read_agent_task_events(workspace_root, task_id)
    already: set[str] = set()
    for event in existing:
        if event.get("type") != "tool_call_completed":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            art = result.get("artifact")
            if isinstance(art, str) and art:
                already.add(art)
    labels = dict(_STAGE_LABELS)
    ts = _created_at(run)
    events: list[dict[str, Any]] = []
    for kind in kinds:
        if kind in already:
            continue
        label = labels.get(kind, kind.replace("_", " "))
        events.append(
            {
                "type": "tool_call_completed",
                "ts": ts,
                "payload": {
                    "tool_name": label,
                    "result": {"artifact": kind},
                    "turn_id": turn_id,
                    "mode": mode,
                },
            }
        )
    if events:
        append_agent_task_events(workspace_root, task_id, events)
    return len(events)


# ── knowledge experiment-record note ─────────────────────────────────────────
# Disk write + knowledge.created + guarded cite all go through
# ``molexp.workspace.write_knowledge_item`` (agent-record-export-02). Plan-only
# rendering stays here.


def write_experiment_record(
    *,
    run: Run,
    experiment: Experiment,
    draft: str,
    model: str,
) -> KnowledgeItem:
    """Write the Decision-kind experiment record via ``write_knowledge_item``.

    Raises when no ``experiment_report`` artifact exists — the record renders
    from it, and a missing report is the caller's signal to skip this writer
    (checked via :func:`has_artifact`), never a silent no-op.
    """
    from molexp.workspace.knowledge_item import SourceRef
    from molexp.workspace.knowledge_write import write_knowledge_item

    report = _read_artifact_json(run, "experiment_report")
    if report is None:
        raise ValueError(f"run {run.id} has no experiment_report artifact to record")
    title = _title(report, draft, run.id)
    tasks = _read_workflow_tasks(experiment)
    source = _read_workflow_source(run)
    body = _render_markdown(
        title,
        draft,
        report,
        run,
        model,
        experiment_id=experiment.id,
        project_id=experiment.project.id if hasattr(experiment, "project") else "",
        tasks=tasks,
        source=source,
    )
    item_name = f"experiment-record-{experiment.id}-{run.id}"
    sources = [
        SourceRef(kind="run", ref=run.id),
        SourceRef(kind="experiment", ref=experiment.id),
    ]
    report_ref = _artifact_ref_id(run, "experiment_report")
    if report_ref is not None:
        sources.append(SourceRef(kind="artifact", ref=report_ref))
    return write_knowledge_item(
        experiment,
        name=item_name,
        kind="Decision",
        sources=sources,
        created_by=f"PlanMode/{model}",
        body=body,
        cite=[(run, "derived_from")],
        title=title,
        actor="plan-record",
    )


def write_finding_record(
    *,
    run: Run,
    experiment: Experiment,
    draft: str,
    model: str,
) -> KnowledgeItem:
    """Write the Finding from the execute tail's ``final_report``.

    Raises when no ``final_report`` exists (caller gates via
    :func:`has_artifact`). Sources cite the run, the experiment, and the
    exact final_report (+ audit_report when present) artifacts; a
    ``references`` edge connects the Finding to the Decision record when
    that record exists — plan → outcome stays traversable.
    """
    from molexp.workspace.knowledge_item import KnowledgeItem, SourceRef
    from molexp.workspace.knowledge_write import write_knowledge_item

    final_report = _read_artifact_json(run, "final_report")
    if final_report is None:
        raise ValueError(f"run {run.id} has no final_report artifact to harvest")
    title = _title(final_report, draft, run.id)
    item_name = f"finding-{experiment.id}-{run.id}"
    sources = [
        SourceRef(kind="run", ref=run.id),
        SourceRef(kind="experiment", ref=experiment.id),
    ]
    for kind in ("final_report", "audit_report"):
        ref_id = _artifact_ref_id(run, kind)
        if ref_id is not None:
            sources.append(SourceRef(kind="artifact", ref=ref_id))
    lines = [f"# Finding: {title}", ""]
    for key, label in _FINAL_REPORT_FIELDS:
        block = _render_value(final_report.get(key))
        if block:
            lines += [f"## {label}", "", block, ""]
    cites: list[tuple[Folder, EdgeRole]] = [(run, "derived_from")]
    try:
        decision = experiment.get_folder(
            f"experiment-record-{experiment.id}-{run.id}", cls=KnowledgeItem
        )
    except Exception:
        decision = None  # no Decision record (its write raced/failed) — Finding stands alone
    if decision is not None:
        cites.append((decision, "references"))
    return write_knowledge_item(
        experiment,
        name=item_name,
        kind="Finding",
        sources=sources,
        created_by=f"PlanMode/{model}",
        body="\n".join(lines).rstrip() + "\n",
        cite=cites,
        title=title,
        actor="plan-record",
    )


def write_failure_analysis_record(
    *,
    run: Run,
    experiment: Experiment,
    model: str,
    failure_stage: str | None,
    failure_error: str,
) -> KnowledgeItem:
    """Write the FailureAnalysis for a plan that terminally failed.

    Never called for an approval suspension — a suspension is not a failure
    (the callers carve ``ApprovalPendingError`` out before reaching this).
    """
    from molexp.workspace.knowledge_item import SourceRef
    from molexp.workspace.knowledge_write import write_knowledge_item

    item_name = f"failure-{experiment.id}-{run.id}"
    completed = _artifact_kinds(run)
    lines = [
        f"# Failure analysis: plan run {run.id}",
        "",
        f"failed stage: {failure_stage or 'unknown'}",
        "",
        "## Error",
        "",
        failure_error.strip() or "(no error text)",
        "",
        "## Completed before the failure",
        "",
        ("\n".join(f"- {kind}" for kind in completed) if completed else "(no artifacts)"),
        "",
        "## Resume",
        "",
        "Re-running the same draft resumes the stage ledger from here; artifacts "
        "a validator rejected are regenerated, not reused.",
    ]
    title = f"Failure analysis: plan run {run.id}"
    return write_knowledge_item(
        experiment,
        name=item_name,
        kind="FailureAnalysis",
        sources=[
            SourceRef(kind="run", ref=run.id),
            SourceRef(kind="experiment", ref=experiment.id),
        ],
        created_by=f"PlanMode/{model}",
        body="\n".join(lines).rstrip() + "\n",
        cite=[(run, "derived_from")],
        title=title,
        actor="plan-record",
    )


_FINAL_REPORT_FIELDS: list[tuple[str, str]] = [
    ("conclusion", "Conclusion"),
    ("hypothesis_verdict", "Hypothesis verdict"),
    ("summary", "Summary"),
    ("metrics", "Metrics"),
    ("caveats", "Caveats"),
]


def has_artifact(run: Run, kind: str) -> bool:
    """Whether the run's artifact store holds at least one *kind* artifact."""
    return _artifact_ref_id(run, kind) is not None


def _read_workflow_tasks(experiment: Experiment) -> list[str]:
    """The generated workflow's task ids (the spec), from the persisted IR."""
    raw = (
        experiment.metadata.workflow_source
        if hasattr(experiment.metadata, "workflow_source")
        else None
    )
    if not isinstance(raw, str) or not raw:
        return []
    try:
        ir = json.loads(raw)
    except ValueError, TypeError:
        return []
    tcs = ir.get("task_configs") if isinstance(ir, dict) else None
    if not isinstance(tcs, list):
        return []
    return [
        tc["task_id"] for tc in tcs if isinstance(tc, dict) and isinstance(tc.get("task_id"), str)
    ]


def _artifact_ref_id(run: Run, kind: str) -> str | None:
    """The content-addressed id of the run's latest *kind* artifact (or None)."""
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    ref = store.latest_by_kind(kind)
    return ref.id if ref is not None else None


def _read_workflow_source(run: Run) -> str | None:
    data = _read_artifact_json(run, "workflow_source")
    source = data.get("source") if isinstance(data, dict) else None
    return source if isinstance(source, str) else None


_FIELD_ORDER: list[tuple[str, str]] = [
    ("objective", "Objective"),
    ("background", "Background"),
    ("system_description", "System description"),
    ("scientific_hypothesis", "Scientific hypothesis"),
    ("experimental_design", "Experimental design"),
    ("variables", "Variables"),
    ("controlled_conditions", "Controlled conditions"),
    ("expected_outputs", "Expected outputs"),
    ("assumptions", "Assumptions"),
    ("risks_or_uncertainties", "Risks & uncertainties"),
    ("user_questions", "Open questions"),
]


def _render_markdown(
    title: str,
    draft: str,
    report: dict[str, Any],
    run: Run,
    model: str,
    *,
    experiment_id: str,
    project_id: str = "",
    tasks: list[str] | None = None,
    source: str | None = None,
    include_request: bool = True,
) -> str:
    lines = [
        f"# {title}",
        "",
        (
            f"> Experiment record from PlanMode · experiment `{experiment_id}` · "
            f"run `{run.id}` · model `{model}` · {_created_at(run)}"
        ),
        "",
    ]
    # Where this plan landed in the workspace.
    where = " / ".join(f"`{part}`" for part in (project_id, experiment_id, run.id) if part)
    if where:
        lines += ["## Where", "", f"Project / Experiment / Run: {where}", ""]
    if include_request and draft.strip():
        lines += ["## Original request", "", draft.strip(), ""]
    known = {k for k, _ in _FIELD_ORDER} | {"title"}
    for key, label in _FIELD_ORDER:
        block = _render_value(report.get(key))
        if block:
            lines += [f"## {label}", "", block, ""]
    for key, value in report.items():
        if key in known:
            continue
        block = _render_value(value)
        if block:
            lines += [f"## {key.replace('_', ' ').title()}", "", block, ""]
    # The generated workflow spec — every task + the runnable source.
    if tasks:
        bullets = "\n".join(f"{i + 1}. `{t}`" for i, t in enumerate(tasks))
        lines += ["## Generated workflow", "", f"{len(tasks)} tasks:", "", bullets, ""]
    if source and source.strip():
        lines += ["## Workflow source", "", "```python", source.strip(), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def _render_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [str(v) if not isinstance(v, dict) else _render_dict(v) for v in value]
        return "\n".join(f"- {item}" for item in items if item)
    if isinstance(value, dict):
        return _render_dict(value)
    return str(value)


def _render_dict(value: dict) -> str:  # gradual: values come from isinstance-narrowed object
    return "; ".join(f"**{k}**: {v}" for k, v in value.items())


# ── helpers ──────────────────────────────────────────────────────────────────


def _title(report: dict[str, Any] | None, draft: str, fallback: str) -> str:
    if report is not None:
        title = report.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    first_line = draft.strip().splitlines()[0] if draft.strip() else ""
    return first_line[:80] if first_line else fallback


def _created_at(run: Run) -> str:
    try:
        return run.metadata.created_at.isoformat()
    except Exception:
        return datetime.now(UTC).isoformat()


def _read_artifact_json(run: Run, kind: str) -> dict[str, Any] | None:
    """The run's latest *kind* artifact parsed as a JSON object (or None)."""
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    root = Path(run.run_dir) / "artifacts"
    store = FileArtifactStore(root=root)
    ref = store.latest_by_kind(kind)
    if ref is None:
        return None
    direct = root / kind / f"{ref.id}.json"
    raw: str | bytes | None = None
    try:
        raw = direct.read_text()
    except OSError:
        try:
            raw = store.get(ref.id)
        except Exception:
            return None
    try:
        data = json.loads(raw)
    except ValueError, TypeError:
        return None
    return data if isinstance(data, dict) else None
