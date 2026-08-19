"""Tests for workspace JSON schema versioning (spec: core-versioning).

Covers acceptance criteria:
- ac-003: Every entity JSON writer emits schema_version: 1
- ac-004: JSON missing schema_version is rejected
- ac-005: Future schema_version raises IncompatibleSchemaError
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.schema_version import (
    MOLEXP_SCHEMA_VERSION,
    IncompatibleSchemaError,
)


def _seed_workspace(root) -> Workspace:
    ws = Workspace(root=root, name="Lab")
    proj = ws.add_project("p")
    exp = proj.add_experiment("e", params={"lr": 1e-3})
    run = exp.add_run()
    with run.start() as ctx:
        ctx.register_artifact({"loss": 0.1}, name="metrics.json")
    return ws


def _every_entity_json(workspace_root) -> list:
    root = Path(workspace_root)
    out = []
    out.append(root / "workspace.json")
    for proj in (root / "projects").iterdir():
        out.append(proj / "project.json")
        for exp in (proj / "experiments").iterdir():
            out.append(exp / "experiment.json")
            for run in (exp / "runs").iterdir():
                out.append(run / "run.json")
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
            with open(path) as fh:  # noqa: PTH123
                data = json.load(fh)
            assert "schema_version" in data, f"missing schema_version: {path}"
            assert data["schema_version"] == MOLEXP_SCHEMA_VERSION


class TestMissingSchemaRejected:
    def test_workspace_without_schema_version_raises(self, tmp_path):
        root = tmp_path / "ws_v0"
        root.mkdir()
        (root / "workspace.json").write_text(
            json.dumps(
                {
                    "id": "ws_v0",
                    "name": "Lab",
                    "created_at": "2024-01-01T00:00:00",
                    "targets": [],
                }
            )
        )

        with pytest.raises(IncompatibleSchemaError, match="missing schema_version"):
            Workspace.load(root)


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
