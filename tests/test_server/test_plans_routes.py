"""Tests for the plan-document routes (``/projects/{p}/experiments/{e}/plans``).

A "plan" is a run carrying a persisted ``experiment_report`` artifact (the
PlanMode result). These routes read that durable artifact so the UI's Agents hub
can list + display generated plans independently of the ephemeral plan task.
"""

from __future__ import annotations

from pathlib import Path

from molexp.harness.store.file_artifact_store import FileArtifactStore

PROJECT_ID = "test-project"
EXPERIMENT_ID = "test-exp"
BASE = f"/api/projects/{PROJECT_ID}/experiments/{EXPERIMENT_ID}/plans"


def _write_plan_artifacts(run, *, title: str, draft: str) -> None:
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    store.put_json(
        "experiment_report",
        {"title": title, "objective": "Measure something", "assumptions": ["a", "b"]},
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "user_plan",
        {"raw_text": draft, "user_id": "u", "submitted_at": "now"},
        created_by="test",
        parent_ids=[],
    )


def test_list_plans_includes_runs_with_experiment_report(client, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrun1")
    _write_plan_artifacts(run, title="A polymer melt plan", draft="build a melt")

    resp = client.get(BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    plan = body["plans"][0]
    assert plan["runId"] == "planrun1"
    assert plan["title"] == "A polymer melt plan"
    assert plan["hasWorkflow"] is False


def test_list_plans_excludes_runs_without_report(client, experiment):
    # A plain run with no experiment_report artifact must not appear as a plan.
    experiment.add_run(params={"lr": 1e-4}, id="plainrun")

    resp = client.get(BASE)
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_get_plan_returns_report_and_draft(client, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrun2")
    _write_plan_artifacts(run, title="Detailed plan", draft="build a melt")

    resp = client.get(f"{BASE}/planrun2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runId"] == "planrun2"
    assert body["title"] == "Detailed plan"
    assert body["draft"] == "build a melt"
    assert body["experimentReport"]["objective"] == "Measure something"
    assert body["experimentReport"]["assumptions"] == ["a", "b"]


def test_get_plan_404_for_run_without_report(client, experiment):
    experiment.add_run(params={"lr": 1e-4}, id="plainrun2")

    resp = client.get(f"{BASE}/plainrun2")
    assert resp.status_code == 404


def test_get_plan_404_for_unknown_run(client, experiment):
    resp = client.get(f"{BASE}/does-not-exist")
    assert resp.status_code == 404


def _write_full_plan_artifacts(run, *, title: str, draft: str) -> None:
    """Write the full set of 9-step deliverables (spec, caps, input set, dry run, report)."""
    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    _write_plan_artifacts(run, title=title, draft=draft)
    store.put_json(
        "experiment_spec",
        {
            "id": "spec-1",
            "experiment_report_id": "rep-1",
            "title": title,
            "objective": "Measure something",
            "variables": [{"name": "T", "value": {"value": 300, "source": "user_provided"}}],
        },
        created_by="test",
        parent_ids=[],
    )
    store.put_text(
        "capability_catalog",
        "## Available molcrafts capabilities\n\n- molpy.build_water",
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "workflow_ir",
        {
            "id": "wf-1",
            "name": "demo_wf",
            "objective": "x",
            "inputs": {"T": {"value": 300, "source": "user_provided"}},
            "tasks": [
                {
                    "id": "build",
                    "name": "Build",
                    "purpose": "make it",
                    "task_type": "builder",
                    "inputs": {},
                    "outputs": {"structure": "pdb"},
                },
                {
                    "id": "pack",
                    "name": "Pack",
                    "purpose": "pack it",
                    "task_type": "builder",
                    "inputs": {},
                    "outputs": {"box": "data"},
                },
            ],
            "edges": [{"source_task_id": "build", "target_task_id": "pack"}],
            "expected_outputs": [],
        },
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "plan_review",
        {"passed": True, "findings": [], "summary": "Workflow faithfully implements the spec."},
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "input_set",
        {"id": "is-1", "experiment_spec_id": "spec-1", "title": "sweep", "total_runs": 3},
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "execution_result",
        {
            "id": "er-1",
            "bound_workflow_id": "bw-1",
            "status": "succeeded",
            "exit_code": 0,
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
            "metadata": {"mode": "compile"},
        },
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        "execution_report",
        {"id": "exr-1", "bound_workflow_id": "bw-1", "target_name": "laptop", "total_runs": 3},
        created_by="test",
        parent_ids=[],
    )


def test_get_plan_surfaces_all_nine_step_deliverables(client, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "build a melt"}, id="planrunfull")
    _write_full_plan_artifacts(run, title="Full plan", draft="build a melt")

    body = client.get(f"{BASE}/planrunfull").json()
    assert body["experimentSpec"]["variables"][0]["name"] == "T"
    # Draft spec is ONE merged YAML: the experiment spec keys PLUS a
    # workflow_spec: section embedding the curated workflow IR.
    spec_yaml = body["experimentSpecYaml"]
    assert "objective: Measure something" in spec_yaml
    assert "workflow_spec:" in spec_yaml
    assert "demo_wf" in spec_yaml and "build → pack" in spec_yaml
    assert "molpy.build_water" in body["capabilities"]
    assert body["inputSet"]["total_runs"] == 3
    assert body["dryRun"]["metadata"]["mode"] == "compile"
    assert body["executionReport"]["target_name"] == "laptop"
    # Workflow IR → curated workflow-spec YAML (topology + typed I/O + per-task deps).
    assert body["workflowIr"]["name"] == "demo_wf"
    yaml_text = body["workflowIrYaml"]
    assert "tasks:" in yaml_text and "build" in yaml_text
    assert "outputs:" in yaml_text and "structure" in yaml_text
    assert "depends_on:" in yaml_text  # pack depends on build
    assert "dataflow:" in yaml_text and "build → pack" in yaml_text
    # Review deliverable surfaces the plan review.
    assert body["planReview"]["passed"] is True


def test_list_all_plans_is_workspace_wide(client, experiment):
    run = experiment.add_run(params={"mode": "plan", "draft": "x"}, id="planrun3")
    _write_plan_artifacts(run, title="Workspace-wide plan", draft="x")
    experiment.add_run(params={"lr": 1e-4}, id="plainrun3")  # no report → excluded

    resp = client.get("/api/plans")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    plan = body["plans"][0]
    assert plan["projectId"] == PROJECT_ID
    assert plan["experimentId"] == EXPERIMENT_ID
    assert plan["runId"] == "planrun3"
    assert plan["title"] == "Workspace-wide plan"
