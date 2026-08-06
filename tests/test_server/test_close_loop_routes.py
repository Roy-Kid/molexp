"""Route tests for close-loop analyze-failure + metrics ingest (CLI ≡ API)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.workspace import Workspace
from molexp.workspace.run_ops import RunStatus


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    workspace = Workspace(root=tmp_path / "ws", name="lab")
    workspace.materialize()
    return workspace


@pytest.fixture()
def client(ws: Workspace) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: ws
    return TestClient(app)


def _failed_run(ws: Workspace):
    exp = ws.add_project("p").add_experiment("e")
    run = exp.add_run(params={"x": 1}, id="aabbcc01")
    run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
    err = run.run_dir / "executions" / "exec-aabbcc01" / "error.txt"
    err.parent.mkdir(parents=True, exist_ok=True)
    err.write_text("unit-test-oom\n", encoding="utf-8")
    return run


class TestAnalyzeFailureRoute:
    def test_failed_run_writes_failure_analysis(self, client: TestClient, ws: Workspace) -> None:
        run = _failed_run(ws)
        res = client.post(
            f"/api/projects/p/experiments/e/runs/{run.id}/analyze-failure",
            json={"created_by": "test"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["name"] == f"failure-analysis-{run.id}"
        assert "path" in body
        # Second call is idempotent by name.
        res2 = client.post(
            f"/api/projects/p/experiments/e/runs/{run.id}/analyze-failure",
            json={"created_by": "test", "narrative": "updated"},
        )
        assert res2.status_code == 200
        assert res2.json()["name"] == body["name"]

    def test_succeeded_run_refused(self, client: TestClient, ws: Workspace) -> None:
        run = _failed_run(ws)
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        res = client.post(
            f"/api/projects/p/experiments/e/runs/{run.id}/analyze-failure",
            json={},
        )
        assert res.status_code == 400


class TestMetricsIngestRoutes:
    def test_detect_and_ingest_empty_run_soft_ok(self, client: TestClient, ws: Workspace) -> None:
        exp = ws.add_project("p").add_experiment("e")
        run = exp.add_run(params={}, id="aabbcc02")
        det = client.get(f"/api/projects/p/experiments/e/runs/{run.id}/metrics/detect")
        assert det.status_code == 200, det.text
        assert det.json()["runId"] == run.id
        assert isinstance(det.json()["hits"], list)

        ing = client.post(f"/api/projects/p/experiments/e/runs/{run.id}/metrics/ingest")
        assert ing.status_code == 200, ing.text
        body = ing.json()
        assert body["runId"] == run.id
        assert body["records"] == 0
        assert isinstance(body["skipped"], list)
