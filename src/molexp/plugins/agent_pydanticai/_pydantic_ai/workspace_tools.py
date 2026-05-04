"""Native pydantic-ai tools exposed by the molexp agent.

Each tool here registers itself with the package-level
:class:`~molexp.plugins.agent_pydanticai.tool_registry.ToolRegistry`
via the :func:`~molexp.plugins.agent_pydanticai.tool_registry.native_tool`
decorator. Categories:

- ``workspace`` — create/list projects, experiments, runs (workspace
  structure manipulation).
- ``workflow`` — workflow IR binding + run lifecycle (``submit_run``,
  ``set_workflow_from_ir``, ``execute_run``, …). Some are mutating;
  the ``mutates`` flag drives plan-mode filtering.
- ``chat`` — ``ask_user`` for clarification mid-run.
- ``control`` — session-control tools like ``exit_plan_mode`` (wired in
  the session layer to halt + emit a structured event).

Heavy analytic / plotting code is intentionally *not* native: install
``molcrafts-mcp`` (read tools) and a code-exec MCP server (aggregation,
plotting) when those are needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import RunContext

from ..tool_registry import native_tool
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


@native_tool(category="workflow", mutates=True)
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


@native_tool(category="workflow", mutates=False)
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


@native_tool(category="workflow", mutates=False)
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


@native_tool(category="workflow", mutates=True)
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


@native_tool(category="chat", mutates=False)
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


@native_tool(category="workspace", mutates=False)
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


@native_tool(category="workspace", mutates=False)
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


@native_tool(category="workspace", mutates=False)
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


@native_tool(category="workflow", mutates=False)
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


@native_tool(category="workspace", mutates=True)
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


@native_tool(category="workflow", mutates=False)
async def list_workflow_templates(
    ctx: RunContext[MolexpDeps],
) -> list[dict[str, Any]]:
    """Return the catalog of built-in workflow templates the agent can attach.

    Each entry is ``{"name", "description", "parameters"}``. Pass ``name``
    to :func:`create_experiment` as the ``template`` argument.
    """
    return list_templates()


@native_tool(category="workflow", mutates=True)
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


@native_tool(category="workflow", mutates=False)
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


@native_tool(category="workflow", mutates=True)
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


@native_tool(category="workflow", mutates=True)
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


# ── Session-control tools ───────────────────────────────────────────────────


@native_tool(category="control", mutates=False)
async def exit_plan_mode(
    ctx: RunContext[MolexpDeps],
    plan_markdown: str,
    workflow_preview: dict[str, Any],
) -> dict[str, Any]:
    """Hand a finalized plan back to the user for explicit approval.

    Halts the agent until the user approves, rejects, or edits the
    plan via the chat UI. The same session resumes after the decision
    — see :meth:`PydanticAISession.respond_plan`.

    Every plan is a workflow. The numbered steps in ``plan_markdown``
    are the prose view; ``workflow_preview.workflow_ir`` is the
    structured view of the same nodes. They MUST be in lockstep:
    every step number in the prose corresponds to one
    ``task_configs[]`` node, including investigation-style steps like
    ``read_paper``, ``inspect_dataset``, or ``survey_runs``. On
    approval the session flips out of plan mode and the agent
    proceeds to bind / execute the workflow (the IR compiles directly
    to a runnable Python script — see ``workflow_preview.python_script``
    if you want to ship the script alongside the IR).

    Args:
        plan_markdown: Numbered step plan as markdown. One step per
            node in ``workflow_ir.task_configs``, in the same order.
        workflow_preview: Structured workflow preview with shape::

                {
                    "workflow_ir": {
                        "name": "<short name>",
                        "task_configs": [
                            {"task_id": "<unique>",
                             "task_type": "<slug from list_task_types>",
                             "config": {...}},
                            ...
                        ],
                        "links": [{"source": "<task_id>",
                                   "target": "<task_id>"}, ...],
                        "metadata": {}
                    },
                    "python_script": "<optional; renderable from IR>",
                    "mermaid": "<optional; UI auto-derives a graph>",
                    "intervention_points": ["rename A to fetch", ...]
                }

            ``task_configs`` MUST contain at least one node — empty
            workflows are not valid plans.

    Returns:
        On approval: ``{"approved": True, "edited_plan": "...",
        "edited_workflow_ir": {...} | None}``.
        On rejection: ``{"approved": False, "feedback": "..."}``.
        On contract violation: ``{"error": "..."}`` — fix the call
        and try again in the same turn.
    """
    if not isinstance(workflow_preview, dict):
        return {
            "error": (
                "exit_plan_mode: workflow_preview must be an object with "
                "shape {workflow_ir: {...}, python_script?, mermaid?, "
                "intervention_points?}."
            )
        }
    ir = workflow_preview.get("workflow_ir")
    if not isinstance(ir, dict):
        return {
            "error": (
                "exit_plan_mode: workflow_preview.workflow_ir must be an "
                "object with task_configs[] and links[]."
            )
        }
    task_configs = ir.get("task_configs")
    if not isinstance(task_configs, list) or not task_configs:
        return {
            "error": (
                "exit_plan_mode: workflow_preview.workflow_ir.task_configs "
                "MUST contain at least one node. Investigation-style steps "
                "(read literature, grep codebase, inspect runs) belong in "
                "the IR as investigation tasks — every step is a node."
            )
        }
    session = getattr(ctx.deps, "session", None)
    if session is None or not hasattr(session, "await_plan_decision"):
        return {
            "error": (
                "exit_plan_mode requires an interactive session with "
                "plan-mode support."
            )
        }
    decision = await session.await_plan_decision(
        plan_markdown=plan_markdown,
        workflow_preview=workflow_preview,
    )
    return decision


# NOTE: the static READ_ONLY_TOOLS / WRITE_TOOLS / CHAT_TOOLS lists were
# replaced by :class:`~molexp.plugins.agent_pydanticai.tool_registry.ToolRegistry`.
# Each function above self-registers via :func:`@native_tool`. Catalogs and
# the settings UI now query the registry directly, so adding a new tool
# requires no edits to module-level lists.
