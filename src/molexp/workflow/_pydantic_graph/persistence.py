"""``workflow.json`` writer for a workflow execution.

Each workflow execution writes one ``workflow.json`` under::

    <run_dir>/executions/<execution_id>/workflow.json

The file is observability state, not resume state. It carries the execution
status plus a copy of the compiled workflow IR with per-node/per-link statuses
so the UI can render a live workflow graph while tasks are running.

Atomic writes route through workspace's
:func:`molexp.workspace.atomic_write_json` so the atomicity guarantee is
workspace's, not a workflow-layer reinvention (workspace <- workflow direction).
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.workspace import atomic_write_json

from ..._typing import JSONValue

if TYPE_CHECKING:
    from ..compiled import CompiledWorkflow
    from ..protocols import TaskOutput

_LOCK = threading.Lock()


def _iter_dicts(value: JSONValue) -> Iterator[dict[str, JSONValue]]:
    """Yield only the ``dict`` items from a JSONValue expected to be a list."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def _workflow_json_path(run_dir: Path, execution_id: str) -> Path:
    return run_dir / "executions" / execution_id / "workflow.json"


def _now() -> str:
    return datetime.now().isoformat()


def _jsonable(value: Any) -> JSONValue:  # noqa: ANN401
    """Return a compact JSON-safe representation for workflow observability."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in list(value)[:20]]
    return str(value)


def _task_id(task: dict[str, JSONValue]) -> str | None:
    value = task.get("task_id", task.get("id"))
    return value if isinstance(value, str) else None


def _link_source(link: dict[str, JSONValue]) -> str | None:
    value = link.get("source", link.get("from"))
    return value if isinstance(value, str) else None


def _link_target(link: dict[str, JSONValue]) -> str | None:
    value = link.get("target", link.get("to"))
    return value if isinstance(value, str) else None


def read_node_outputs(
    run_dir: str | os.PathLike[str] | None, execution_id: str | None
) -> dict[str, TaskOutput]:
    """Return completed-task outputs persisted in an execution's ``workflow.json``.

    Reads ``<run_dir>/executions/<execution_id>/workflow.json`` and returns the
    ``{task_name: output}`` map for every task whose ``status`` is
    ``"completed"`` and that recorded an ``outputs`` value. The result is
    suitable as a ``seed_outputs=`` argument to
    :meth:`molexp.workflow.WorkflowRuntime.execute`, letting a resumed run skip
    already-finished nodes and recompute only the remainder.

    Resume seeding is **JSON-fidelity only**: outputs were persisted through
    :func:`_jsonable` (a JSON-lossy round-trip in :func:`mark_task_status`), so
    the values returned here are JSON-normalized rather than the original Python
    objects. Tasks whose outputs are not JSON-roundtrippable cannot be resumed
    at node granularity and must be continued via ``rerun`` (a fresh execution
    from the top of the graph, where the content-addressed cache may opportunistically hit).

    Non-raising: returns an empty mapping when *run_dir* or *execution_id* is
    ``None``, the file is missing, the JSON is malformed, or its top-level shape
    is not a JSON object.
    """
    if run_dir is None or execution_id is None:
        return {}
    wf_path = _workflow_json_path(Path(run_dir), execution_id)
    if not wf_path.exists():
        return {}
    try:
        data = json.loads(wf_path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    outputs: dict[str, TaskOutput] = {}
    for task in data.get("task_configs", []):
        if not isinstance(task, dict) or task.get("status") != "completed":
            continue
        if "outputs" not in task:
            continue
        name = _task_id(task)
        if name is not None:
            outputs[name] = task["outputs"]
    return outputs


def _initial_document(execution_id: str, compiled: CompiledWorkflow | None) -> dict[str, JSONValue]:
    if compiled is None:
        return {
            "schema_version": 1,
            "execution_id": execution_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "task_configs": [],
            "links": [],
        }

    # Observability serialization: tolerate slug-less tasks (decorator /
    # bare ``Task`` subclasses) — workflow.json renders the live graph and is
    # not round-tripped, so a missing ``task_type`` must not crash the run.
    ir = copy.deepcopy(compiled.to_ir(strict=False))
    raw_tasks = ir.get("task_configs", [])
    raw_links = ir.get("links", [])
    tasks: list[JSONValue] = (
        [t for t in raw_tasks if isinstance(t, dict)] if isinstance(raw_tasks, list) else []
    )
    links: list[JSONValue] = (
        [ln for ln in raw_links if isinstance(ln, dict)] if isinstance(raw_links, list) else []
    )
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task["status"] = "pending"
    for link in links:
        if not isinstance(link, dict):
            continue
        link["status"] = "pending"

    document: dict[str, JSONValue] = {
        **ir,
        "schema_version": 1,
        "execution_id": execution_id,
        "workflow_id": compiled.workflow_id,
        "workflow_name": compiled.name,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
    }
    document["task_configs"] = tasks
    document["links"] = links
    return document


def write_initial_workflow_json(
    run_dir: Path,
    execution_id: str,
    *,
    compiled: CompiledWorkflow | None = None,
) -> None:
    """Create ``executions/<execution_id>/`` and write initial ``workflow.json``."""
    exec_dir = run_dir / "executions" / execution_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(exec_dir / "workflow.json", _initial_document(execution_id, compiled))


def _mutate_document(
    run_dir: Path | None,
    execution_id: str | None,
    mutate,  # noqa: ANN001
) -> None:
    if run_dir is None or execution_id is None:
        return
    wf_path = _workflow_json_path(run_dir, execution_id)
    if not wf_path.exists():
        return
    with _LOCK:
        try:
            data = json.loads(wf_path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        mutate(data)
        atomic_write_json(wf_path, data)


def mark_task_status(
    run_dir: Path | None,
    execution_id: str | None,
    task_name: str,
    status: str,
    *,
    output: Any = None,  # noqa: ANN401
    error: str | None = None,
) -> None:
    """Update one task and adjacent links in ``workflow.json``."""

    def _apply(data: dict[str, JSONValue]) -> None:
        now = _now()
        for task in _iter_dicts(data.get("task_configs", [])):
            if _task_id(task) == task_name:
                task["status"] = status
                if status == "running":
                    task["started_at"] = task.get("started_at") or now
                elif status in {"completed", "failed", "skipped"}:
                    task["finished_at"] = now
                if output is not None:
                    task["outputs"] = _jsonable(output)
                if error:
                    task["error"] = error
        for link in _iter_dicts(data.get("links", [])):
            if _link_target(link) == task_name and status == "running":
                link["status"] = "running"
            if _link_target(link) == task_name and status in {"completed", "skipped"}:
                link["status"] = "completed"
            if _link_source(link) == task_name and status == "completed":
                link["status"] = "running"
            if (
                _link_source(link) == task_name or _link_target(link) == task_name
            ) and status == "failed":
                link["status"] = "failed"

    _mutate_document(run_dir, execution_id, _apply)


def mark_workflow_finished(
    run_dir: Path,
    execution_id: str,
    *,
    status: str,
    outputs: Any = None,  # noqa: ANN401
    error: str | None = None,
) -> None:
    """Mark the execution document terminal."""

    def _apply(data: dict[str, JSONValue]) -> None:
        data["status"] = status
        data["finished_at"] = _now()
        data["outputs"] = _jsonable(outputs or {})
        if error:
            data["error"] = error
        if status == "completed":
            for link in _iter_dicts(data.get("links", [])):
                if link.get("status") == "running":
                    link["status"] = "completed"

    _mutate_document(run_dir, execution_id, _apply)
