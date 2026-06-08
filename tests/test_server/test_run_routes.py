"""Tests for run API routes."""

import pytest

from molexp.plugins.submit_molq.submit import SubmitHandler
from molexp.workspace import ComputeTarget, add_target


class TestRunRoutes:
    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_create(self, client, project, experiment):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {"lr": 1e-4}},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["parameters"] == {"lr": 1e-4}
        assert data["status"] == "pending"
        assert data["projectId"] == project.id
        assert data["experimentId"] == experiment.id

    def test_create_captures_workflow_snapshot(self, client, project, experiment):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}},
        )
        data = resp.json()
        assert data["workflow"] is not None
        assert data["workflow"]["source"] == "train.py"

    def test_list(self, client, project, experiment):
        client.post(self._prefix(project, experiment), json={"parameters": {}})
        client.post(self._prefix(project, experiment), json={"parameters": {}})
        resp = client.get(self._prefix(project, experiment))
        assert len(resp.json()) == 2

    def test_get(self, client, project, experiment, run):
        run._update_metadata(executor_info={"backend": "molq", "scheduler": "slurm"})
        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == run.id
        assert resp.json()["executorInfo"] == {"backend": "molq", "scheduler": "slurm"}

    def test_update_status(self, client, project, experiment, run):
        resp = client.patch(
            f"{self._prefix(project, experiment)}/{run.id}/status",
            json={"status": "succeeded"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "succeeded"
        assert data["finished"] is not None

    def test_get_exposes_results_and_history(self, client, project, experiment, run):
        # Drive a real execution so context.results and execution_history are populated.
        with run.start() as ctx:
            ctx.set_result("y", 9.0)

        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == {"y": 9.0}
        assert data["workflowSource"] == "train.py"
        assert len(data["executionHistory"]) == 1
        record = data["executionHistory"][0]
        assert record["executionId"].startswith("exec-")
        assert record["status"] == "succeeded"
        assert record["startedAt"] is not None
        assert record["finishedAt"] is not None

    def test_list_summary_includes_results(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_result("y", 9.0)

        resp = client.get(
            f"/api/projects/{project.id}/experiments/{experiment.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"], "experiment should report at least one run"
        summary = next(r for r in data["runs"] if r["id"] == run.id)
        assert summary["results"] == {"y": 9.0}
        assert summary["finished"] is not None


@pytest.fixture
def local_target(workspace):
    add_target(
        workspace,
        ComputeTarget(name="laptop", scratch_root=str(workspace.root / "scratch")),
    )
    return "laptop"


@pytest.fixture
def captured_submits(monkeypatch):
    """Recorder for ``SubmitHandler.__call__``; each entry is ``(handler, args, kwargs)``."""
    calls: list = []
    monkeypatch.setattr(
        SubmitHandler,
        "__call__",
        lambda self, *args, **kwargs: calls.append((self, args, kwargs)),
    )
    return calls


class TestRunSubmissionWiring:
    """The API path must dispatch to molq when a target is given."""

    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_create_with_target_invokes_submit_handler(
        self,
        client,
        project,
        experiment_with_entrypoint,
        local_target,
        captured_submits,
    ):
        resp = client.post(
            self._prefix(project, experiment_with_entrypoint),
            json={"parameters": {"lr": 1e-4}, "target": local_target},
        )
        assert resp.status_code == 201, resp.text
        assert len(captured_submits) == 1
        handler, args, _kwargs = captured_submits[0]
        assert handler._scheduler == "local"
        _script, mol_run, exp, proj = args
        assert mol_run.id == resp.json()["id"]
        assert exp.id == experiment_with_entrypoint.id
        assert proj.id == project.id

    def test_create_without_target_skips_submit(
        self, client, project, experiment, captured_submits
    ):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}},
        )
        assert resp.status_code == 201
        assert captured_submits == []

    def test_create_with_target_without_entrypoint_returns_422(
        self, client, project, experiment, local_target
    ):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}, "target": local_target},
        )
        assert resp.status_code == 422
        assert "entrypoint" in resp.json()["detail"].lower()

    def test_rerun_starts_new_execution_on_same_run_and_dispatches(
        self,
        client,
        project,
        experiment_with_entrypoint,
        local_target,
        captured_submits,
    ):
        """ac-002: rerun appends a new execution on the SAME run; no clone."""
        src_run = experiment_with_entrypoint.add_run(parameters={"lr": 1e-4}, target=local_target)
        captured_submits.clear()  # ignore implicit submit on source-run creation
        runs_before = len(experiment_with_entrypoint.list_runs())

        resp = client.post(
            f"{self._prefix(project, experiment_with_entrypoint)}/{src_run.id}/rerun"
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        # Same run id — not a fresh run.
        assert body["runId"] == src_run.id
        assert "newRunId" not in body
        assert "sourceRunId" not in body
        # A real exec id on this run.
        assert body["executionId"].startswith(f"exec-{src_run.id}")
        # No clone: the experiment's run count is unchanged.
        assert len(experiment_with_entrypoint.list_runs()) == runs_before
        # One molq dispatch carrying the chosen execution id.
        assert len(captured_submits) == 1
        _handler, _args, kwargs = captured_submits[0]
        assert kwargs.get("execution_id") == body["executionId"]

    def test_rerun_without_target_does_not_submit(
        self, client, project, experiment, run, captured_submits
    ):
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 201
        body = resp.json()
        assert body["runId"] == run.id
        assert "newRunId" not in body
        assert body["executionId"].startswith(f"exec-{run.id}")
        assert captured_submits == []

    def test_resume_reopens_last_non_succeeded_execution(
        self, client, project, experiment, run, captured_submits
    ):
        """ac-003: resume reopens the run's last non-succeeded execution.

        Drive a real execution that fails so ``execution_history`` carries
        a non-succeeded record; resume must target *that* existing id and
        append no new execution.
        """
        with pytest.raises(RuntimeError, match="boom"), run.start():
            raise RuntimeError("boom")

        history = run.metadata.execution_history
        assert history, "the failed run should have one execution record"
        last_exec_id = history[-1].execution_id
        history_len = len(history)

        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/resume")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["runId"] == run.id
        assert body["executionId"] == last_exec_id
        # No new execution appended by resume (it reopened the existing one).
        assert len(run.metadata.execution_history) == history_len

    def test_resume_without_resumable_execution_returns_409(self, client, project, experiment, run):
        """ac-004: resume on a fresh run (empty history) → 409 mentioning rerun."""
        assert run.metadata.execution_history == []

        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/resume")
        assert resp.status_code == 409, resp.text
        assert "rerun" in resp.json()["detail"].lower()
        # No execution was created as a side effect.
        assert run.metadata.execution_history == []

    def test_targeted_resume_dispatches_on_reopened_execution_id(
        self,
        client,
        project,
        experiment_with_entrypoint,
        local_target,
        captured_submits,
    ):
        """ac-005: targeted resume hits the molq seam with the reopened exec id."""
        src_run = experiment_with_entrypoint.add_run(parameters={"lr": 1e-4}, target=local_target)
        with pytest.raises(RuntimeError, match="boom"), src_run.start():
            raise RuntimeError("boom")
        captured_submits.clear()
        last_exec_id = src_run.metadata.execution_history[-1].execution_id

        resp = client.post(
            f"{self._prefix(project, experiment_with_entrypoint)}/{src_run.id}/resume"
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["executionId"] == last_exec_id
        assert len(captured_submits) == 1
        _handler, _args, kwargs = captured_submits[0]
        assert kwargs.get("execution_id") == last_exec_id

    def test_resume_with_unregistered_target_returns_422(
        self, client, project, experiment_with_entrypoint, captured_submits
    ):
        """ac-005: an inherited target not registered on the workspace → 422."""
        src_run = experiment_with_entrypoint.add_run(parameters={"lr": 1e-4}, target="ghost")
        with pytest.raises(RuntimeError, match="boom"), src_run.start():
            raise RuntimeError("boom")
        captured_submits.clear()

        resp = client.post(
            f"{self._prefix(project, experiment_with_entrypoint)}/{src_run.id}/resume"
        )
        assert resp.status_code == 422, resp.text
        assert captured_submits == []

    def test_rerun_with_target_but_no_entrypoint_returns_422(
        self, client, project, experiment, local_target, captured_submits
    ):
        """ac-005: targeted run whose experiment lacks a workflow entrypoint → 422.

        ``experiment`` (not ``experiment_with_entrypoint``) binds no spec, so
        ``_dispatch_to_molq``'s entrypoint guard fires.
        """
        src_run = experiment.add_run(parameters={"lr": 1e-4}, target=local_target)
        captured_submits.clear()

        resp = client.post(f"{self._prefix(project, experiment)}/{src_run.id}/rerun")
        assert resp.status_code == 422, resp.text
        assert "entrypoint" in resp.json()["detail"].lower()
        assert captured_submits == []


class TestRunContinuationSchemaAndOpenAPI:
    """ac-001 + ac-006: the schema swap and OpenAPI surface change."""

    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_run_continue_response_schema_shape(self):
        """ac-001: RunContinueResponse importable with the five fields."""
        from molexp.server.schemas import RunContinueResponse

        model = RunContinueResponse(
            runId="r1",
            executionId="exec-r1",
            status="pending",
            projectId="p1",
            experimentId="e1",
        )
        assert set(model.model_fields) == {
            "runId",
            "executionId",
            "status",
            "projectId",
            "experimentId",
        }

    def test_run_rerun_response_is_removed(self):
        """ac-001: the old clone-shaped response is gone."""
        import molexp.server.schemas as schemas

        assert not hasattr(schemas, "RunRerunResponse")
        with pytest.raises(ImportError):
            from molexp.server.schemas import RunRerunResponse  # noqa: F401

    def test_openapi_exposes_resume_and_rerun_paths(self, client):
        """ac-006: paths + new schema present, old shape absent."""
        spec = client.get("/api/openapi.json").json()
        paths = spec["paths"]
        resume_paths = [p for p in paths if p.endswith("/{run_id}/resume")]
        rerun_paths = [p for p in paths if p.endswith("/{run_id}/rerun")]
        assert resume_paths, "resume path missing from OpenAPI"
        assert rerun_paths, "rerun path missing from OpenAPI"

        schema_names = set(spec.get("components", {}).get("schemas", {}))
        assert "RunContinueResponse" in schema_names
        assert "RunRerunResponse" not in schema_names

        raw = client.get("/api/openapi.json").text
        assert "RunRerunResponse" not in raw
        assert "newRunId" not in raw

    def test_kill_routes_through_try_cancel(self, client, project, experiment, run, monkeypatch):
        seen: list = []

        def fake_try_cancel(r):
            seen.append(r.id)
            r.cancel()
            return

        monkeypatch.setattr("molexp.server.routes.run.try_cancel", fake_try_cancel)
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/kill")
        assert resp.status_code == 200
        assert seen == [run.id]
        assert resp.json()["message"] == "Run cancelled"
        assert resp.json()["status"] == "cancelled"

    def test_kill_falls_back_when_try_cancel_warns(
        self, client, project, experiment, run, monkeypatch
    ):
        monkeypatch.setattr(
            "molexp.server.routes.run.try_cancel",
            lambda r: "no molq job id",  # noqa: ARG005
        )
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/kill")
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "no molq job id"
        assert body["status"] == "cancelled"
