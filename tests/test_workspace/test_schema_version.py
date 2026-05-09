"""Tests for workspace JSON schema versioning (spec: core-versioning).

Covers acceptance criteria:
- ac-003: Every entity JSON writer emits schema_version: 1
- ac-004: Legacy v0 JSON (missing schema_version) loads + auto-upgrades
- ac-005: Future schema_version raises IncompatibleSchemaError
"""

from __future__ import annotations

import json

import pytest

from molexp.workspace import Workspace
from molexp.workspace.schema_version import (
    MOLEXP_SCHEMA_VERSION,
    IncompatibleSchemaError,
)


def _seed_workspace(root) -> Workspace:
    ws = Workspace(root=root, name="Lab")
    proj = ws.Project("p")
    exp = proj.Experiment("e", params={"lr": 1e-3})
    run = exp.Run()
    with run.start() as ctx:
        ctx.artifact.save("metrics.json", {"loss": 0.1})
    return ws


def _every_entity_json(workspace_root) -> list:
    out = []
    out.append(workspace_root / "workspace.json")
    out.append(workspace_root / "projects.json")
    for proj in (workspace_root / "projects").iterdir():
        out.append(proj / "project.json")
        out.append(proj / "experiments.json")
        for exp in (proj / "experiments").iterdir():
            out.append(exp / "experiment.json")
            out.append(exp / "runs.json")
            for run in (exp / "runs").iterdir():
                out.append(run / "run.json")
                out.append(run / "executions.json")
                exec_root = run / "executions"
                if exec_root.exists():
                    for ex in exec_root.iterdir():
                        if ex.is_dir():
                            out.append(ex / "execution.json")
    return [p for p in out if p.exists()]


class TestSchemaVersionEmitted:
    def test_every_entity_json_carries_schema_version(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        targets = _every_entity_json(ws.root)
        assert targets, "test seed produced no entity JSON files"

        for path in targets:
            with open(path) as fh:
                data = json.load(fh)
            assert "schema_version" in data, f"missing schema_version: {path}"
            assert data["schema_version"] == MOLEXP_SCHEMA_VERSION


class TestLegacyV0Load:
    def test_workspace_without_schema_version_loads(self, tmp_path):
        # Hand-craft a v0 workspace: no schema_version field anywhere.
        root = tmp_path / "ws_v0"
        root.mkdir()
        (root / "workspace.json").write_text(
            json.dumps(
                {
                    "id": "ws_v0",
                    "name": "Legacy Lab",
                    "created_at": "2024-01-01T00:00:00",
                    "targets": [],
                }
            )
        )
        proj_dir = root / "projects" / "p"
        proj_dir.mkdir(parents=True)
        (proj_dir / "project.json").write_text(
            json.dumps(
                {
                    "id": "p",
                    "name": "p",
                    "description": "",
                    "owner": "",
                    "tags": [],
                    "config": {},
                    "created_at": "2024-01-01T00:00:00",
                }
            )
        )

        ws = Workspace.load(root)
        assert ws.name == "Legacy Lab"
        projects = ws.list_projects()
        assert len(projects) == 1
        assert projects[0].id == "p"

        # Re-saving must upgrade to current schema_version.
        ws.save()
        with open(root / "workspace.json") as fh:
            saved = json.load(fh)
        assert saved["schema_version"] == MOLEXP_SCHEMA_VERSION


class TestFutureSchemaRejected:
    def test_workspace_future_schema_raises(self, tmp_path):
        root = tmp_path / "ws_future"
        root.mkdir()
        (root / "workspace.json").write_text(
            json.dumps(
                {
                    "schema_version": MOLEXP_SCHEMA_VERSION + 99,
                    "id": "ws_future",
                    "name": "From Tomorrow",
                    "created_at": "2099-01-01T00:00:00",
                    "targets": [],
                }
            )
        )
        with pytest.raises(IncompatibleSchemaError):
            Workspace.load(root)
