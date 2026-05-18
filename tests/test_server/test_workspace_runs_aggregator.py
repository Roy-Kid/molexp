"""Tests for the workspace-level runs aggregator endpoint.

Exercises ``GET /api/workspace/runs`` end-to-end via FastAPI TestClient,
including filter handling, embedded execution rows, and stats counts.
"""

from datetime import datetime, timedelta

from molexp.workspace.models import ExecutionRecord


def _seed_run(
    workspace, *, project_id, experiment_id, parameters, executor_info=None, status="pending"
):
    project = workspace.add_project(project_id)
    experiment = project.add_experiment(
        experiment_id, workflow_source="train.py", params=parameters
    )
    run = experiment.add_run(parameters=parameters)
    updates = {"status": status}
    if executor_info is not None:
        updates["executor_info"] = executor_info
    run._update_metadata(**updates)
    return run


def _append_execution(
    run, *, execution_id, status, scheduler_job_id=None, started_at=None, finished_at=None
):
    """Append a synthetic ExecutionRecord to a run's history."""
    started = started_at or datetime.now()
    record = ExecutionRecord(
        execution_id=execution_id,
        started_at=started,
        finished_at=finished_at,
        status=status,
        scheduler_job_id=scheduler_job_id,
    )
    history = list(run.metadata.execution_history)
    history.append(record)
    run._update_metadata(execution_history=history)


class TestWorkspaceRunsAggregator:
    def test_lists_runs_across_projects_and_experiments(self, client, workspace):
        _seed_run(
            workspace,
            project_id="proj-a",
            experiment_id="exp-a",
            parameters={"lr": 1e-4},
            executor_info={"backend": "local"},
            status="succeeded",
        )
        _seed_run(
            workspace,
            project_id="proj-b",
            experiment_id="exp-b",
            parameters={"lr": 1e-3},
            executor_info={
                "backend": "molq",
                "scheduler": "slurm",
                "cluster_name": "dardel",
            },
            status="running",
        )

        resp = client.get("/api/workspace/runs")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total"] == 2
        assert data["truncated"] is False
        backends = {row["backend"] for row in data["runs"]}
        assert backends == {"local", "molq"}

    def test_embeds_execution_rows_with_backend_metadata(self, client, workspace):
        run = _seed_run(
            workspace,
            project_id="proj-a",
            experiment_id="exp-a",
            parameters={},
            executor_info={
                "backend": "molq",
                "scheduler": "slurm",
                "cluster_name": "dardel",
            },
            status="running",
        )
        started = datetime.now() - timedelta(minutes=10)
        finished = datetime.now() - timedelta(minutes=2)
        _append_execution(
            run,
            execution_id="exec-1",
            status="succeeded",
            scheduler_job_id="48201",
            started_at=started,
            finished_at=finished,
        )

        resp = client.get("/api/workspace/runs")
        rows = resp.json()["runs"]
        molq_row = next(r for r in rows if r["backend"] == "molq")
        assert molq_row["cluster"] == "dardel"
        assert molq_row["executionCount"] == 1
        assert molq_row["latestSchedulerJobId"] == "48201"

        execution = molq_row["executions"][0]
        assert execution["schedulerJobId"] == "48201"
        assert execution["backend"] == "molq"
        assert execution["backendMetadata"]["cluster_name"] == "dardel"
        assert execution["backendMetadata"]["scheduler_job_id"] == "48201"
        assert execution["durationSeconds"] is not None and execution["durationSeconds"] > 0

    def test_filters_by_backend_and_status(self, client, workspace):
        _seed_run(
            workspace,
            project_id="p1",
            experiment_id="e1",
            parameters={},
            executor_info={"backend": "local"},
            status="succeeded",
        )
        _seed_run(
            workspace,
            project_id="p2",
            experiment_id="e2",
            parameters={},
            executor_info={"backend": "molq"},
            status="running",
        )

        resp = client.get("/api/workspace/runs", params={"backend": "molq"})
        assert resp.status_code == 200
        rows = resp.json()["runs"]
        assert len(rows) == 1
        assert rows[0]["backend"] == "molq"

        resp = client.get("/api/workspace/runs", params={"status": "succeeded"})
        rows = resp.json()["runs"]
        assert all(r["status"] == "succeeded" for r in rows)

    def test_stats_counts_match_run_states(self, client, workspace):
        for idx in range(3):
            _seed_run(
                workspace,
                project_id="p",
                experiment_id=f"e-{idx}",
                parameters={"i": idx},
                status="running",
            )
        _seed_run(
            workspace,
            project_id="p",
            experiment_id="e-done",
            parameters={},
            status="succeeded",
        )
        _seed_run(
            workspace,
            project_id="p",
            experiment_id="e-fail",
            parameters={},
            status="failed",
        )

        stats = client.get("/api/workspace/runs").json()["stats"]
        assert stats["total"] == 5
        assert stats["running"] == 3
        assert stats["succeeded"] == 1
        assert stats["failed"] == 1

    def test_empty_workspace_returns_zero(self, client):
        resp = client.get("/api/workspace/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["runs"] == []
        assert body["stats"]["total"] == 0
