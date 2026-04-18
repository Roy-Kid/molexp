"""Built-in molexp workspace tools for the agent layer.

Three access levels (per proposal §6.5):

Level 1 — Workspace tools (read-only, never require approval)
  list_projects, list_experiments, list_runs, get_run_summary

Level 2 — Workflow tools (execution, Phase 3 will add these)
  workflow_execute (stubbed — NotImplementedError until Phase 3)

Level 3 — Product tools (write operations, some require approval)
  create_run

All functions take `RunContext[MolexpDeps]` as first argument so
pydantic-ai recognises them as context-aware tools.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from .deps import MolexpDeps

# ── Level 1: Workspace read-only tools ────────────────────────────────────────

async def list_projects(ctx: RunContext[MolexpDeps]) -> list[dict[str, Any]]:
    """List all projects in the workspace.

    Returns a list of project summaries with id and name.
    """
    projects = ctx.deps.workspace.list_projects()
    return [
        {
            "id": p.id,
            "name": p.metadata.name,
            "description": getattr(p.metadata, "description", ""),
        }
        for p in projects
    ]


async def list_experiments(
    ctx: RunContext[MolexpDeps], project_id: str
) -> list[dict[str, Any]]:
    """List all experiments in a project.

    Args:
        project_id: ID of the project to list experiments for
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}  # type: ignore[return-value]
    experiments = project.list_experiments()
    return [
        {
            "id": e.id,
            "name": e.metadata.name,
            "description": getattr(e.metadata, "description", ""),
        }
        for e in experiments
    ]


async def list_runs(
    ctx: RunContext[MolexpDeps], project_id: str, experiment_id: str
) -> list[dict[str, Any]]:
    """List all runs in an experiment.

    Args:
        project_id: ID of the project
        experiment_id: ID of the experiment
    """
    project = ctx.deps.workspace.get_project(project_id)
    if project is None:
        return {"error": f"Project '{project_id}' not found"}  # type: ignore[return-value]
    experiment = project.get_experiment(experiment_id)
    if experiment is None:
        return {"error": f"Experiment '{experiment_id}' not found"}  # type: ignore[return-value]
    runs = experiment.list_runs()
    return [
        {
            "id": r.id,
            "status": str(r.status),
            "parameters": getattr(r, "parameters", {}),
        }
        for r in runs
    ]


async def get_run_summary(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Get a detailed summary of a specific run.

    Args:
        project_id: ID of the project
        experiment_id: ID of the experiment
        run_id: ID of the run
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
        "id": run.id,
        "status": str(run.status),
        "parameters": getattr(run, "parameters", {}),
        "metadata": {
            "created_at": str(getattr(run.metadata, "created_at", "")),
            "finished_at": str(getattr(run.metadata, "finished_at", "")),
        },
    }


# ── Level 2: Workflow tools (Phase 3) ────────────────────────────────────────

async def workflow_execute(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    run_id: str,
    workflow_spec_json: str,
) -> dict[str, Any]:
    """Execute a workflow for a given run.

    Note: Full implementation available in Phase 3 (pydantic-graph runtime).

    Args:
        project_id: ID of the project
        experiment_id: ID of the experiment
        run_id: ID of the run to execute in
        workflow_spec_json: JSON-serialized WorkflowSpec
    """
    return {
        "error": "not_implemented",
        "message": (
            "Workflow execution will be available in Phase 3 "
            "(pydantic-graph runtime integration)."
        ),
    }


# ── Level 3: Product write tools ──────────────────────────────────────────────

async def create_run(
    ctx: RunContext[MolexpDeps],
    project_id: str,
    experiment_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create a new run in an experiment with specified parameters.

    Args:
        project_id: ID of the project
        experiment_id: ID of the experiment
        parameters: Hyperparameters and configuration for this run
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
        "parameters": parameters,
    }


def get_all_builtin_tools() -> list:
    """Return all built-in tool functions for catalog registration."""
    return [
        list_projects,
        list_experiments,
        list_runs,
        get_run_summary,
        workflow_execute,
        create_run,
    ]
