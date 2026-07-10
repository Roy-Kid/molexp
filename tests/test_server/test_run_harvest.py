"""POST …/runs/{id}/harvest — ordinary run → KnowledgeItem."""

from __future__ import annotations


def test_harvest_terminal_run(client, project, experiment, run):
    with run.start():
        pass
    assert run.status == "succeeded"
    resp = client.post(
        f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}/harvest",
        json={
            "kind": "Finding",
            "narrative": "Mobility looks good.",
            "created_by": "test",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "finding" in body["name"] or run.id in body["name"]


def test_harvest_empty_narrative_400(client, project, experiment, run):
    with run.start():
        pass
    resp = client.post(
        f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}/harvest",
        json={"kind": "Finding", "narrative": "  ", "created_by": "test"},
    )
    assert resp.status_code == 400
