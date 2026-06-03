"""Tests for the workflow document write-back route (flowgram-workflow-canvas-02).

The route is a thin wrapper over ``WorkflowCodec``: PUT validates + normalizes
an IR document through ``ir_to_spec``/``spec_to_ir`` and persists it onto the
experiment's ``workflow_source`` metadata; GET reads it back. Invalid documents
map to a structured 4xx (workflow-layer ``WorkflowError`` or codec ``ValueError``),
never a 500.
"""

from __future__ import annotations

import json

import pytest

from molexp.workflow import WorkflowCompiler, default_codec
from molexp.workflow.registry import _Constant


def _valid_ir() -> dict:
    wf = WorkflowCompiler(name="wf")
    wf.add(_Constant(value=1), name="a", task_type="core.constant", config={"value": 1})
    wf.add(_Constant(value=2), name="b", task_type="core.constant", config={"value": 2})
    wf.add(
        _Constant(value=3),
        name="c",
        depends_on=["a", "b"],
        task_type="core.constant",
        config={"value": 3},
    )
    return dict(default_codec.spec_to_ir(wf.compile()))


def _cyclic_ir() -> dict:
    """Data cycle a->b->a — ``ir_to_spec`` raises a workflow-layer CycleError."""
    ir = _valid_ir()
    ir["links"] = [
        {"source": "a", "target": "b", "mapping": {}, "status": "pending", "kind": "data"},
        {"source": "b", "target": "a", "mapping": {}, "status": "pending", "kind": "data"},
    ]
    return ir


def _dangling_ir() -> dict:
    """Link to an unknown task_id — ``ir_to_spec`` raises ValueError."""
    ir = _valid_ir()
    ir["links"].append(
        {"source": "ghost", "target": "c", "mapping": {}, "status": "pending", "kind": "data"}
    )
    return ir


@pytest.fixture
def exp(project):
    """A bare experiment with no workflow document stored yet."""
    return project.add_experiment("wf-exp", params={})


def _url(project, exp) -> str:
    return f"/api/projects/{project.id}/experiments/{exp.id}/workflow"


class TestPutGet:
    def test_put_valid_persists_and_returns_normalized(self, client, project, exp) -> None:
        resp = client.put(_url(project, exp), json={"document": _valid_ir()})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert {tc["task_id"] for tc in body["document"]["task_configs"]} == {"a", "b", "c"}

        # workflow_source metadata now holds the normalized IR JSON.
        reloaded = project.get_experiment(exp.id)
        stored = json.loads(reloaded.metadata.workflow_source)
        assert {tc["task_id"] for tc in stored["task_configs"]} == {"a", "b", "c"}

    def test_get_round_trips_put(self, client, project, exp) -> None:
        put = client.put(_url(project, exp), json={"document": _valid_ir()})
        assert put.status_code == 200, put.text
        got = client.get(_url(project, exp))
        assert got.status_code == 200, got.text
        assert got.json()["document"] == put.json()["document"]

    def test_get_on_never_written_returns_structured_404(self, client, project, exp) -> None:
        resp = client.get(_url(project, exp))
        assert resp.status_code == 404
        assert "error" in resp.json()


class TestErrorMapping:
    def test_cyclic_document_maps_to_structured_4xx(self, client, project, exp) -> None:
        resp = client.put(_url(project, exp), json={"document": _cyclic_ir()})
        assert resp.status_code != 500
        assert 400 <= resp.status_code < 500
        assert "error" in resp.json()

    def test_dangling_link_maps_to_400(self, client, project, exp) -> None:
        resp = client.put(_url(project, exp), json={"document": _dangling_ir()})
        assert resp.status_code == 400
        assert "error" in resp.json()


class TestRegistration:
    def test_workflow_path_present_in_openapi(self, client) -> None:
        paths = client.app.openapi()["paths"]
        assert "/api/projects/{project_id}/experiments/{experiment_id}/workflow" in paths
