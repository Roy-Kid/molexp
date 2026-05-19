"""Tests for catalog reverse-lookup routes."""

from __future__ import annotations

from pathlib import Path


class TestCatalogByPath:
    def test_unmatched_path_under_projects_returns_derived_scope(self, client, project, experiment):
        rel = f"projects/{project.id}/experiments/{experiment.id}"
        resp = client.get("/api/catalog/by-path", params={"path": rel})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is False
        assert data["scope"]["kind"] == "experiment"
        assert data["scope"]["projectId"] == project.id
        assert data["scope"]["experimentId"] == experiment.id

    def test_path_outside_workspace_rejected(self, client):
        resp = client.get("/api/catalog/by-path", params={"path": "/etc/passwd"})
        assert resp.status_code == 400

    def test_unknown_path_returns_unmatched(self, client):
        resp = client.get(
            "/api/catalog/by-path",
            params={"path": "no/such/dir"},
        )
        assert resp.status_code == 200
        assert resp.json()["matched"] is False

    def test_matched_artifact_returns_producer_and_scope(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_active_task("train")
            ctx.artifact.save("metrics.json", '{"loss": 0.1}')

        # Pick the relative path under the workspace root
        from molexp.workspace.assets import AssetScope

        scope = AssetScope(kind="run", ids=(project.id, experiment.id, run.id))
        assets = experiment.project.workspace.catalog.query_assets(scope=scope)
        artifact = next(a for a in assets if a.kind == "artifact")
        rel = (Path(run.run_dir) / artifact.path).relative_to(
            Path(experiment.project.workspace.root)
        )

        resp = client.get("/api/catalog/by-path", params={"path": str(rel)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["assetKind"] == "artifact"
        assert data["scope"]["kind"] == "run"
        assert data["scope"]["runId"] == run.id
        assert data["scope"]["experimentId"] == experiment.id
        assert data["scope"]["projectId"] == project.id
        assert data["producer"]["runId"] == run.id
        assert data["producer"]["taskId"] == "train"


class TestRunFilesAndActions:
    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_run_files_lists_outputs(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_active_task("train")
            ctx.artifact.save("model.bin", b"weights")

        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runId"] == run.id
        # The artifact file appears somewhere in the tree, with assetKind set
        all_nodes: list[dict] = []

        def walk(node: dict) -> None:
            all_nodes.append(node)
            for child in node.get("children", []):
                walk(child)

        for top in data["nodes"]:
            walk(top)
        artifact_nodes = [n for n in all_nodes if n.get("assetKind") == "artifact"]
        assert artifact_nodes, "expected at least one artifact node"

    def test_run_rerun_creates_new_run_with_same_params(self, client, project, experiment, run):
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 201
        data = resp.json()
        assert data["sourceRunId"] == run.id
        assert data["newRunId"] != run.id
        new_run = experiment.get_run(data["newRunId"])
        assert new_run is not None
        assert new_run.parameters == run.parameters

    def test_run_kill_marks_cancelled(self, client, project, experiment, run):
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/kill")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        refreshed = experiment.get_run(run.id)
        assert refreshed.status == "cancelled"

    def test_run_export_returns_zip(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_active_task("train")
            ctx.artifact.save("model.bin", b"weights")

        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert resp.content[:2] == b"PK"  # zip magic bytes


class TestExperimentComparison:
    def test_comparison_aggregates_runs(self, client, project, experiment):
        r1 = experiment.add_run(parameters={"lr": 1e-3})
        r2 = experiment.add_run(parameters={"lr": 1e-4})

        with r1.start() as ctx:
            ctx.set_active_task("train")
        with r2.start() as ctx:
            ctx.set_active_task("train")

        resp = client.get(f"/api/projects/{project.id}/experiments/{experiment.id}/comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experimentId"] == experiment.id
        assert {row["runId"] for row in data["runs"]} >= {r1.id, r2.id}
        assert data["paramKeys"] == ["lr"]


class TestWorkspaceFilesIncludeCatalog:
    def test_include_catalog_enriches_artifact_node(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_active_task("train")
            ctx.artifact.save("metrics.json", '{"loss": 0.1}')

        resp = client.get(
            "/api/workspace/files",
            params={"include": "catalog", "max_depth": 8},
        )
        assert resp.status_code == 200

        all_nodes: list[dict] = []

        def walk(node: dict) -> None:
            all_nodes.append(node)
            for child in node.get("children", []):
                walk(child)

        for child in resp.json()["children"]:
            walk(child)
        artifact_nodes = [n for n in all_nodes if n.get("assetKind") == "artifact"]
        assert artifact_nodes, "expected catalog-enriched artifact node"
