"""Tests for the workspace-structure + execution tools (the ones that
turn the agent into a complete chat-driven driver)."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.workspace_tools import (
    create_experiment,
    create_project,
    execute_run,
    get_run_results,
    list_experiments,
    list_projects,
    list_runs,
    list_workflow_templates,
    submit_run,
)


# ── Read-only listers ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_projects_empty_returns_empty_list(fake_ctx):
    assert await list_projects(fake_ctx) == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_projects_after_create(fake_ctx):
    await create_project(fake_ctx, name="demo", description="smoke test")
    rows = await list_projects(fake_ctx)
    assert any(r["name"] == "demo" for r in rows)
    assert all("id" in r for r in rows)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_experiments_unknown_project(fake_ctx):
    out = await list_experiments(fake_ctx, project_id="nope")
    assert isinstance(out, dict) and "error" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_experiments_after_create(fake_ctx):
    proj = await create_project(fake_ctx, name="demo")
    await create_experiment(
        fake_ctx, project_id=proj["project_id"], name="square", template="square"
    )
    rows = await list_experiments(fake_ctx, project_id=proj["project_id"])
    assert any(r["name"] == "square" and r["has_workflow"] is True for r in rows)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_runs_unknown_experiment(fake_ctx):
    proj = await create_project(fake_ctx, name="demo")
    out = await list_runs(fake_ctx, project_id=proj["project_id"], experiment_id="nope")
    assert isinstance(out, dict) and "error" in out


# ── Workflow templates registry ──────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_workflow_templates_includes_square(fake_ctx):
    rows = await list_workflow_templates(fake_ctx)
    names = {r["name"] for r in rows}
    assert "square" in names
    sq = next(r for r in rows if r["name"] == "square")
    assert sq["parameters"] == ["x"]
    assert "x^2" in sq["description"] or "x ** 2" in sq["description"]


# ── create_experiment + workflow attachment ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_experiment_unknown_template(fake_ctx):
    proj = await create_project(fake_ctx, name="demo")
    out = await create_experiment(
        fake_ctx,
        project_id=proj["project_id"],
        name="oops",
        template="not-a-template",
    )
    assert "error" in out
    assert "not-a-template" in out["error"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_experiment_attaches_workflow(fake_ctx, workspace):
    proj = await create_project(fake_ctx, name="demo")
    await create_experiment(
        fake_ctx,
        project_id=proj["project_id"],
        name="square",
        template="square",
    )
    project = workspace.get_project(proj["project_id"])
    experiment = project.get_experiment("square")
    assert experiment.workflow is not None


# ── End-to-end: submit + execute + read results ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_full_chat_lifecycle_square(fake_ctx, workspace):
    """The flow the AI is expected to drive in a single conversation."""
    # 1. AI creates the project.
    proj = await create_project(fake_ctx, name="demo")
    project_id = proj["project_id"]

    # 2. AI picks a template and creates the experiment.
    exp = await create_experiment(
        fake_ctx, project_id=project_id, name="square", template="square"
    )
    experiment_id = exp["experiment_id"]

    # 3. AI submits a run with the user-specified parameter.
    submitted = await submit_run(
        fake_ctx,
        project_id=project_id,
        experiment_id=experiment_id,
        parameters={"x": 7},
    )
    run_id = submitted["run_id"]

    # 4. AI actually executes the workflow against that run.
    executed = await execute_run(
        fake_ctx, project_id=project_id, experiment_id=experiment_id, run_id=run_id
    )
    assert "error" not in executed, executed
    assert executed["run_id"] == run_id
    assert executed["results"]["y"] == 49.0
    assert executed["results"]["x"] == 7.0

    # 5. AI reads the results back via the public lookup.
    result = await get_run_results(
        fake_ctx, project_id=project_id, experiment_id=experiment_id, run_id=run_id
    )
    assert result["results"]["y"] == 49.0
    assert result["status"] in {"succeeded", "completed"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_run_errors_when_no_workflow_attached(
    fake_ctx, workspace
):
    """Simulates the post-server-restart case: experiment exists on disk
    but no workflow is bound in this process."""
    project = workspace.project("demo")
    experiment = project.experiment("naked")
    # Materialize a run on disk, bypassing create_experiment so no workflow
    # is attached in-memory.
    run = experiment.run(parameters={"x": 1})
    out = await execute_run(
        fake_ctx,
        project_id=project.id,
        experiment_id=experiment.id,
        run_id=run.id,
    )
    assert "error" in out
    assert "no workflow attached" in out["error"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_run_unknown_run(fake_ctx, workspace):
    proj = await create_project(fake_ctx, name="demo")
    await create_experiment(
        fake_ctx, project_id=proj["project_id"], name="square", template="square"
    )
    out = await execute_run(
        fake_ctx,
        project_id=proj["project_id"],
        experiment_id="square",
        run_id="run-missing",
    )
    assert "error" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_full_chat_lifecycle_add_template(fake_ctx, workspace):
    """Templates with multiple parameters work too."""
    proj = await create_project(fake_ctx, name="math")
    await create_experiment(
        fake_ctx, project_id=proj["project_id"], name="adder", template="add"
    )
    submitted = await submit_run(
        fake_ctx,
        project_id=proj["project_id"],
        experiment_id="adder",
        parameters={"a": 3, "b": 4},
    )
    executed = await execute_run(
        fake_ctx,
        project_id=proj["project_id"],
        experiment_id="adder",
        run_id=submitted["run_id"],
    )
    assert executed["results"]["z"] == 7.0
