"""Surface a completed PlanMode run on the Agents + Knowledge tabs.

After PlanMode finishes, two records make the AI activity visible in the UI:

1. an **agent-task session** entry (``agent_tasks/<task_id>/metadata.json``) so the
   plan shows in the Agents tab session list alongside chat tasks; and
2. a **Knowledge experiment-record** ``Note`` (an OKF Concept under the
   experiment) rendered from the ``experiment_report`` so the Knowledge tab shows
   a readable record of what was planned.

Both writes are best-effort: a failure here is logged and swallowed so it never
fails the plan run itself.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["record_plan_outputs"]

_LOG = get_logger(__name__)


def record_plan_outputs(
    *,
    run: Run,
    experiment: Experiment,
    workspace_root: str,
    task_id: str,
    draft: str,
    model: str,
) -> None:
    """Write the agent-task session + Knowledge experiment-record for a plan run."""
    report = _read_experiment_report(run)
    title = _title(report, draft, run.id)

    try:
        _write_agent_task(workspace_root, task_id=task_id, title=title, draft=draft, run=run)
        _write_session_events(
            workspace_root,
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            report=report,
        )
    except Exception as exc:
        _LOG.warning(f"[plan-record {run.id}] agent-task entry failed: {exc!r}")

    if report is not None:
        try:
            _write_experiment_record(
                workspace_root,
                experiment,
                run,
                title=title,
                draft=draft,
                report=report,
                model=model,
            )
        except Exception as exc:
            _LOG.warning(f"[plan-record {run.id}] knowledge note failed: {exc!r}")


# ── agent-task session ───────────────────────────────────────────────────────


def _write_agent_task(
    workspace_root: str, *, task_id: str, title: str, draft: str, run: Run
) -> None:
    from molexp.server.routes.agent_task_store import (
        PersistedAgentTask,
        write_agent_task_metadata,
    )

    created = _created_at(run)
    write_agent_task_metadata(
        workspace_root,
        PersistedAgentTask(
            task_id=task_id,
            session_id=task_id,
            title=title,
            goal=draft,
            status=run.status if run.status in {"succeeded", "failed"} else "completed",
            created_at=created,
            updated_at=datetime.now(UTC).isoformat(),
            plan_mode=True,
        ),
    )


# The nine PlanMode steps (artifact kind → step label). The session transcript
# emits one event per kind present, and the UI progress rail keys its per-step
# "done" state on these artifact kinds — so this list mirrors the rail's
# ``planStages.ts`` order exactly. Each step is keyed on the representative
# artifact it produces (step 5 bundles bind/source/tests under workflow_source;
# step 7's compile dry run is an execution_result; step 8's gate is an
# analysis_result).
_STAGE_LABELS: list[tuple[str, str]] = [
    ("experiment_report", "Drafted the experiment proposal"),
    ("experiment_spec", "Drafted the concrete spec"),
    ("capability_catalog", "Resolved molcrafts capabilities"),
    ("workflow_ir", "Drafted the workflow spec"),
    ("workflow_source", "Generated tasks + per-task tests"),
    ("input_set", "Generated the input set"),
    ("execution_result", "Compiled & dry-ran the workflow"),
    ("analysis_result", "Reviewed the plan"),
    ("execution_report", "Produced the execution report"),
]


def _write_session_events(
    workspace_root: str,
    *,
    task_id: str,
    run: Run,
    experiment: Experiment,
    draft: str,
    report: dict[str, Any] | None,
) -> None:
    """Write a synthesized session transcript for the Agents *session view*.

    The transcript is deliberately lean: one step per PlanMode stage plus a short
    final summary. The full deliverables — spec, every task, and the runnable
    source — are NOT crammed into the chat answer; the session view's Deliverables
    panel fetches them structurally from ``GET /plans/{run_id}``. The terminal
    ``loop_completed`` carries a ``plan`` locator so the panel knows which plan to
    open.
    """
    from molexp.server.routes.agent_task_store import write_agent_task_events

    ts = _created_at(run)
    kinds = set(_artifact_kinds(run))
    events: list[dict[str, Any]] = []
    for kind, label in _STAGE_LABELS:
        if kind in kinds:
            events.append(
                {
                    "type": "tool_call_completed",
                    "ts": ts,
                    "payload": {"tool_name": label, "result": {"artifact": kind}},
                }
            )
    if report is not None:
        tasks = _read_workflow_tasks(experiment)
        source = _read_workflow_source(run)
        project_id = experiment.project.id if hasattr(experiment, "project") else ""
        title = _title(report, draft, run.id)
        # Locator the Deliverables panel uses to fetch the structured plan
        # (`GET /projects/{p}/experiments/{e}/plans/{run_id}`). Carried on the
        # open `payload` (which is `Record[str, Any]` on the wire) so no schema
        # or OpenAPI surface has to change.
        plan_ref = {
            "run_id": run.id,
            "project_id": project_id,
            "experiment_id": experiment.id,
            "title": title,
            "step_count": len(tasks),
            "has_workflow": bool(source and source.strip()),
        }
        events.append(
            {
                "type": "loop_completed",
                "ts": ts,
                "payload": {"text": _summary(title, tasks, source), "plan": plan_ref},
            }
        )
    write_agent_task_events(workspace_root, task_id, events)


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


def _artifact_kinds(run: Run) -> list[str]:
    index_dir = Path(run.run_dir) / "artifacts" / "_index"
    if not index_dir.is_dir():
        return []
    return sorted(p.stem for p in index_dir.glob("*.json"))


# ── knowledge experiment-record note ─────────────────────────────────────────


def _write_experiment_record(
    workspace_root: str,
    experiment: Experiment,
    run: Run,
    *,
    title: str,
    draft: str,
    report: dict[str, Any],
    model: str,
) -> None:
    from molexp.workspace import Workspace
    from molexp.workspace.concepts import Note

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
    # Mount the record Note at the WORKSPACE ROOT (idempotent on the slugified
    # name). Mounting under the nested Experiment trips a Bundle path-doubling
    # bug for deeply-nested concepts, so root-mount keeps it reliably readable in
    # the Knowledge tab; the experiment id is encoded in the name + the body.
    ws = Workspace(root=Path(workspace_root), name=Path(workspace_root).name)
    note = ws.add_folder(Note(parent=ws, name=f"experiment-record-{experiment.id}-{run.id}"))
    note.set_body(body)


def _read_workflow_tasks(experiment: Experiment) -> list[str]:
    """The generated workflow's task ids (the spec), from the persisted IR."""
    raw = getattr(experiment.metadata, "workflow_source", None)
    if not isinstance(raw, str) or not raw:
        return []
    try:
        ir = json.loads(raw)
    except (ValueError, TypeError):
        return []
    tcs = ir.get("task_configs") if isinstance(ir, dict) else None
    if not isinstance(tcs, list):
        return []
    return [
        tc["task_id"] for tc in tcs if isinstance(tc, dict) and isinstance(tc.get("task_id"), str)
    ]


def _read_workflow_source(run: Run) -> str | None:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    root = Path(run.run_dir) / "artifacts"
    store = FileArtifactStore(root=root)
    ref = store.latest_by_kind("workflow_source")
    if ref is None:
        return None
    direct = root / "workflow_source" / f"{ref.id}.json"
    try:
        data = json.loads(direct.read_text())
    except (OSError, ValueError, TypeError):
        return None
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


def _render_dict(value: dict[str, Any]) -> str:
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


def _read_experiment_report(run: Run) -> dict[str, Any] | None:
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    root = Path(run.run_dir) / "artifacts"
    store = FileArtifactStore(root=root)
    ref = store.latest_by_kind("experiment_report")
    if ref is None:
        return None
    direct = root / "experiment_report" / f"{ref.id}.json"
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
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None
