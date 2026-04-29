"""Native pydantic-ai tools exposed by the molexp agent.

Three purposes:

1. **Task management** — deterministic side-effecting operations on
   workspace state (``submit_run``, ``get_run_status``, ``wait_for_run``,
   ``retry_run``, ``execute_run``). Some require approval; see
   ``catalog.DEFAULT_APPROVAL_TOOLS``.

2. **Workspace structure** — create / list projects, experiments, runs
   so the agent can drive the full workspace lifecycle from chat. The
   ``create_experiment`` tool attaches a built-in workflow template
   (see :mod:`.workflow_templates`) so users don't have to author
   Python files just to smoke-test the system.

3. **Chat plumbing** — ``ask_user`` lets the agent pause the run and
   ask the user for clarification, returning the reply.

Heavy analytic / plotting code is intentionally *not* native: install
``molcrafts-mcp`` (read tools) and a code-exec MCP server (aggregation,
plotting) when those are needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import RunContext

from .deps import MolexpDeps
from .workflow_templates import TEMPLATES, list_templates


def _read_run_results(run: Any) -> dict[str, Any]:
    """Pull the persisted ``context.results`` dict off a freshly-loaded Run.

    The :class:`~molexp.workspace.Run` instance returned by ``get_run`` is
    metadata-only; it does not enter a :class:`RunContext`, so the
    in-memory ``context`` attribute isn't available. Read straight from
    ``run.json`` (the canonical on-disk representation) instead.
    """
    import json

    run_json_path = run.run_dir / "run.json"
    if not run_json_path.exists():
        return {}
    try:
        payload = json.loads(run_json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        return {}
    results = context.get("results")
    return dict(results) if isinstance(results, dict) else {}

# ── Task-management tools (write side effects) ────────────────────────────────


async def submit_run(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create a new run with the given parameters in an experiment.

    Mirrors ``POST /api/executions``: the run record is materialized on
    disk; actual workflow execution is delegated to whatever launches
    runs (molq scheduler, sweep CLI, etc.). Use ``wait_for_run`` to
    monitor progress.

    Args:
        project_id: Target project ID.
        experiment_id: Target experiment ID.
        parameters: Hyperparameters / configuration for the new run.

    Returns:
        ``{"run_id", "status", "parameters"}`` on success;
        ``{"error", ...}`` on failure.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    run = experiment.run(parameters=parameters)
    return {
        "run_id": run.id,
        "status": str(run.status),
        "parameters": dict(parameters),
    }


async def get_run_status(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Read the current status of a run from on-disk metadata.

    Args:
        project_id: Project containing the run.
        experiment_id: Experiment containing the run.
        run_id: Run identifier.

    Returns:
        ``{"run_id", "status", "started_at", "finished_at", "error"}``.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    run = experiment.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    err_meta = getattr(run.metadata, "error", None)
    error_payload: dict[str, str] | None = None
    if err_meta is not None:
        error_payload = {
            "type": getattr(err_meta, "type", "Error"),
            "message": getattr(err_meta, "message", ""),
        }

    started_at = getattr(run.metadata, "created_at", None)
    finished_at = getattr(run.metadata, "finished_at", None)
    return {
        "run_id": run.id,
        "status": str(run.status),
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "error": error_payload,
    }


_TERMINAL_STATUSES = {"succeeded", "completed", "failed", "cancelled", "error"}


async def wait_for_run(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
    timeout_seconds: float = 300.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Poll a run until it reaches a terminal status or the timeout fires.

    Returns the final ``get_run_status`` payload, plus a ``timed_out``
    flag when the timeout is hit before the run terminates.

    Args:
        project_id: Project containing the run.
        experiment_id: Experiment containing the run.
        run_id: Run identifier.
        timeout_seconds: Max wall-clock seconds to wait. Default 5 min.
        poll_interval: Seconds between status checks. Default 2 s.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        status_payload = await get_run_status(ctx, project_id, experiment_id, run_id)
        if "error" in status_payload and status_payload.get("status") is None:
            return status_payload
        status = str(status_payload.get("status", "")).lower()
        if status in _TERMINAL_STATUSES:
            status_payload["timed_out"] = False
            return status_payload
        if time.monotonic() >= deadline:
            status_payload["timed_out"] = True
            return status_payload
        await asyncio.sleep(max(0.1, float(poll_interval)))


async def retry_run(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Clone an existing run's parameters into a fresh run.

    Mirrors ``POST /api/projects/{p}/experiments/{e}/runs/{r}/rerun``.

    Returns:
        ``{"source_run_id", "new_run_id", "status"}``.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    run = experiment.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    new_run = experiment.run(parameters=dict(run.parameters))
    return {
        "source_run_id": run.id,
        "new_run_id": new_run.id,
        "status": str(new_run.status),
    }


# ── Chat plumbing (no I/O, just hands a prompt to the user) ───────────────────


async def ask_user(ctx: RunContext[MolexpDeps], prompt: str) -> dict[str, Any]:
    """Pause the run and prompt the user for free-form input.

    Use this when the goal is ambiguous — for example, when the user
    asked for an analysis without specifying scope (workspace / project
    / experiment). The session emits a ``UserMessageRequestEvent`` and
    suspends until the user replies via the chat input.

    Args:
        prompt: The question to display to the user.

    Returns:
        ``{"content": str}`` with the user's reply.
    """
    session = getattr(ctx.deps, "session", None)
    if session is None or not hasattr(session, "await_user_message"):
        return {"error": "ask_user requires an interactive session"}
    reply = await session.await_user_message(prompt)
    return {"content": reply}


# ── Workspace structure tools (read + write) ─────────────────────────────────


async def list_projects(ctx: RunContext[MolexpDeps]) -> list[dict[str, Any]]:
    """List every project in the current workspace.

    Returns a list of ``{"id", "name", "description"}`` dicts. Empty when the
    workspace has no projects yet.
    """
    rows: list[dict[str, Any]] = []
    for project in ctx.deps.workspace.list_projects():
        rows.append(
            {
                "id": project.id,
                "name": project.name,
                "description": getattr(project, "description", "") or "",
            }
        )
    return rows


async def list_experiments(
    ctx: RunContext[MolexpDeps], project_id: str
) -> list[dict[str, Any]] | dict[str, Any]:
    """List experiments inside ``project_id``."""
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    rows: list[dict[str, Any]] = []
    for exp in project.list_experiments():
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "params": dict(getattr(exp, "params", {}) or {}),
                "has_workflow": exp.workflow is not None,
            }
        )
    return rows


async def list_runs(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
) -> list[dict[str, Any]] | dict[str, Any]:
    """List runs inside ``project_id`` / ``experiment_id``."""
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    rows: list[dict[str, Any]] = []
    for run in experiment.list_runs():
        rows.append(
            {
                "id": run.id,
                "status": str(run.status),
                "parameters": dict(run.parameters or {}),
            }
        )
    return rows


async def get_run_results(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Read the final ``ctx.set_result`` payload of a run.

    Returns ``{"run_id", "status", "parameters", "results"}``. Use this after
    ``execute_run`` (or once a CLI/molq run has finished) to display values
    back to the user.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    run = experiment.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}
    return {
        "run_id": run.id,
        "status": str(run.status),
        "parameters": dict(run.parameters or {}),
        "results": _read_run_results(run),
    }


async def create_project(
    ctx: RunContext[MolexpDeps],
    name: str,
    description: str = "",
) -> dict[str, Any]:
    """Create (or get) a project by name. Idempotent.

    Args:
        name: Human-readable project name. The workspace slugifies this
            into the on-disk id.
        description: Optional one-line description stored alongside.
    """
    project = ctx.deps.workspace.project(name)
    return {
        "project_id": project.id,
        "name": project.name,
        "description": description,
    }


async def list_workflow_templates(
    ctx: RunContext[MolexpDeps],
) -> list[dict[str, Any]]:
    """Return the catalog of built-in workflow templates the agent can attach.

    Each entry is ``{"name", "description", "parameters"}``. Pass ``name``
    to :func:`create_experiment` as the ``template`` argument.
    """
    return list_templates()


async def create_experiment(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    name: str,
    template: str | None = None,
) -> dict[str, Any]:
    """Create an experiment, optionally binding a built-in workflow template.

    Two flows:

    - **Quick demo**: pass ``template`` (one of :func:`list_workflow_templates`)
      to bind a tiny in-memory workflow. The binding lives only in the
      current server process — a restart drops it.
    - **Real workflows**: omit ``template`` and follow up with
      :func:`set_workflow_from_ir` to bind a JSON IR. The IR is
      persisted to disk and survives restarts.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}

    if template is None:
        experiment = project.experiment(name)
        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "workflow_template": None,
            "has_workflow": experiment.workflow is not None,
        }

    if template not in TEMPLATES:
        return {
            "error": (
                f"Unknown workflow template '{template}'. "
                f"Available: {sorted(TEMPLATES)}"
            )
        }
    fn, description, expected_params = TEMPLATES[template]
    experiment = project.experiment(name)
    if experiment.workflow is None:
        experiment.set_workflow(fn)
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "workflow_template": template,
        "description": description,
        "expected_parameters": list(expected_params),
    }


async def list_task_types(ctx: RunContext[MolexpDeps]) -> list[dict[str, str]]:
    """Return every task-type slug that can appear in a workflow IR.

    Use this before calling :func:`set_workflow_from_ir` so the IR's
    ``task_type`` fields reference real, server-side-resolvable
    capabilities. Each entry is ``{"slug", "description"}``.
    """
    from molexp.workflow.registry import default_registry

    return [
        {"slug": slug, "description": description}
        for slug, description in default_registry.items()
    ]


async def set_workflow_from_ir(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    workflow_json: dict[str, Any],
) -> dict[str, Any]:
    """Bind a JSON workflow IR to an experiment and persist it to disk.

    The IR shape matches ``schema/workflow.json``:

    - ``task_configs[]``: each entry has ``task_id`` (unique within the
      workflow), ``task_type`` (one of :func:`list_task_types`), and
      ``config`` (constructor kwargs for the task).
    - ``links[]``: edges with ``source`` / ``target`` ``task_id`` pairs.
    - ``metadata``: optional label / tags.

    The server compiles the IR into a ``WorkflowSpec`` (validating that
    every slug is registered and every link references known task IDs)
    and writes the IR to ``<exp_dir>/workflow.json`` so the binding
    survives restart. Errors before the experiment is touched are
    surfaced as ``{"error": "..."}``.

    Use this *instead of* :func:`create_experiment`'s ``template`` for
    real workflows — the template path is for demos only.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    if experiment.workflow is not None:
        return {
            "error": (
                f"Experiment '{experiment_id}' already has a workflow bound. "
                "Delete the experiment to rebind."
            )
        }
    try:
        experiment.set_workflow(workflow_json)
    except (KeyError, ValueError) as exc:
        return {"error": f"Invalid workflow IR: {exc}"}
    spec = experiment.workflow
    return {
        "experiment_id": experiment.id,
        "workflow_id": spec.workflow_id if spec is not None else None,
        "task_count": len(workflow_json.get("task_configs", [])),
        "persisted_to": str(experiment.workflow_ir_path),
    }


async def execute_run(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Run the workflow attached to ``experiment_id`` against an existing run.

    Blocks until the workflow finishes (templates are intentionally tiny —
    sub-second). Returns the final status and any persisted results so the
    agent can include them in its reply.

    Errors when the experiment has no workflow attached — that means the
    server was restarted since the experiment was created. Re-call
    ``create_experiment`` with the same name + template to rebind.
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}
    run = experiment.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}
    workflow = experiment.workflow
    if workflow is None:
        return {
            "error": (
                f"Experiment '{experiment_id}' has no workflow attached "
                "(server may have restarted). Re-call create_experiment "
                "with the same name and template to rebind."
            )
        }
    try:
        await workflow.execute(run=run)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the agent
        return {"error": f"Execution failed: {exc!r}", "run_id": run.id}
    return {
        "run_id": run.id,
        "status": str(run.status),
        "parameters": dict(run.parameters or {}),
        "results": _read_run_results(run),
    }


# Tool groups — explicit so plan mode can disable writes by simply omitting
# WRITE_TOOLS from the catalog. Keep these in sync with the system prompt's
# tool surface section in :mod:`system_prompt`.
READ_ONLY_TOOLS: list = [
    list_projects,
    list_experiments,
    list_runs,
    get_run_results,
    list_workflow_templates,
    list_task_types,
    get_run_status,
]

WRITE_TOOLS: list = [
    create_project,
    create_experiment,
    set_workflow_from_ir,
    submit_run,
    execute_run,
    wait_for_run,
    retry_run,
]

CHAT_TOOLS: list = [ask_user]


def get_all_builtin_tools() -> list:
    """Return every native tool function for catalog registration.

    Order matters only for the agent's "available tools" listing —
    keep read-only tools first so the system prompt's free-form
    discovery surfaces inspection options before destructive ones.
    """
    return [*READ_ONLY_TOOLS, *WRITE_TOOLS, *CHAT_TOOLS]


def get_read_only_tools() -> list:
    """Return only the read-only and chat tools, used in plan mode."""
    return [*READ_ONLY_TOOLS, *CHAT_TOOLS]
