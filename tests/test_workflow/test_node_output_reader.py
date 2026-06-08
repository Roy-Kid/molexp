"""RED tests for ``read_node_outputs`` (continue-two-verbs-01-core, verb A).

``read_node_outputs(run_dir, execution_id)`` reads
``<run_dir>/executions/<execution_id>/workflow.json`` and returns
``{task_name: outputs}`` for every task that is ``completed`` AND carries an
``"outputs"`` key. Task name is the task dict's ``"task_id"`` (fallback ``"id"``).
It is non-raising: any missing / malformed / non-dict input yields ``{}``.

Production code does not exist yet — these tests are expected to fail at import
(``ImportError`` / ``AttributeError``) until ``read_node_outputs`` ships.
"""

from __future__ import annotations

import json
from pathlib import Path

from molexp.workflow._pydantic_graph.persistence import read_node_outputs


def _write_workflow_json(run_dir: Path, execution_id: str, document: object) -> None:
    """Write a raw ``workflow.json`` (object may be malformed on purpose)."""
    exec_dir = run_dir / "executions" / execution_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    wf_path = exec_dir / "workflow.json"
    if isinstance(document, str):
        wf_path.write_text(document)
    else:
        wf_path.write_text(json.dumps(document))


def _standard_document() -> dict:
    return {
        "schema_version": 1,
        "execution_id": "exec-x",
        "status": "failed",
        "task_configs": [
            {"task_id": "good", "status": "completed", "outputs": "good-out"},
            {"task_id": "warm", "status": "completed", "outputs": {"k": 1}},
            {"task_id": "boom", "status": "failed", "error": "RuntimeError: x"},
            {"task_id": "sidefx", "status": "completed"},  # completed, no outputs
        ],
        "links": [],
    }


# ── Basics: happy path ──────────────────────────────────────────────────────


def test_happy_path_returns_only_completed_with_outputs(tmp_path: Path) -> None:
    """Exactly the two completed-with-outputs tasks come back, name->outputs."""
    _write_workflow_json(tmp_path, "exec-x", _standard_document())
    result = read_node_outputs(tmp_path, "exec-x")
    assert result == {"good": "good-out", "warm": {"k": 1}}


def test_happy_path_dict_value_preserved(tmp_path: Path) -> None:
    """A dict-valued ``outputs`` is returned by equality, not stringified."""
    _write_workflow_json(tmp_path, "exec-x", _standard_document())
    result = read_node_outputs(tmp_path, "exec-x")
    assert result["warm"] == {"k": 1}


# ── Edge cases: status filtering ────────────────────────────────────────────


def test_failed_task_excluded(tmp_path: Path) -> None:
    _write_workflow_json(tmp_path, "exec-x", _standard_document())
    assert "boom" not in read_node_outputs(tmp_path, "exec-x")


def test_running_and_skipped_tasks_excluded(tmp_path: Path) -> None:
    doc = {
        "task_configs": [
            {"task_id": "alive", "status": "running", "outputs": "partial"},
            {"task_id": "passed", "status": "skipped", "outputs": "ignored"},
            {"task_id": "done", "status": "completed", "outputs": "kept"},
        ],
        "links": [],
    }
    _write_workflow_json(tmp_path, "exec-x", doc)
    assert read_node_outputs(tmp_path, "exec-x") == {"done": "kept"}


def test_completed_without_outputs_omitted(tmp_path: Path) -> None:
    """A completed task that never wrote ``outputs`` is omitted entirely."""
    _write_workflow_json(tmp_path, "exec-x", _standard_document())
    assert "sidefx" not in read_node_outputs(tmp_path, "exec-x")


def test_task_id_fallback_to_id_key(tmp_path: Path) -> None:
    """When ``task_id`` is absent the ``id`` key names the task."""
    doc = {
        "task_configs": [
            {"id": "legacy", "status": "completed", "outputs": "via-id"},
        ],
        "links": [],
    }
    _write_workflow_json(tmp_path, "exec-x", doc)
    assert read_node_outputs(tmp_path, "exec-x") == {"legacy": "via-id"}


# ── Edge cases: non-raising on bad / missing input ──────────────────────────


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_node_outputs(tmp_path, "exec-never-written") == {}


def test_malformed_json_returns_empty(tmp_path: Path) -> None:
    _write_workflow_json(tmp_path, "exec-x", "{not json")
    assert read_node_outputs(tmp_path, "exec-x") == {}


def test_non_dict_top_level_returns_empty(tmp_path: Path) -> None:
    _write_workflow_json(tmp_path, "exec-x", [])
    assert read_node_outputs(tmp_path, "exec-x") == {}


def test_none_run_dir_returns_empty() -> None:
    assert read_node_outputs(None, "exec-x") == {}


def test_none_execution_id_returns_empty(tmp_path: Path) -> None:
    assert read_node_outputs(tmp_path, None) == {}
