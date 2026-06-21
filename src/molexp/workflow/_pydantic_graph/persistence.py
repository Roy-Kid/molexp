"""``workflow.json`` writer for a workflow execution.

Each workflow execution writes one ``workflow.json`` under::

    <run_dir>/executions/<execution_id>/workflow.json

The file is observability state, not resume state. It carries the execution
status plus a copy of the compiled workflow IR with per-node/per-link statuses
so the UI can render a live workflow graph while tasks are running.

Writes are **coalesced** during a live execution: the runtime opens the
document via :func:`open_execution_document`, after which the authoritative
copy lives in memory — per-task status transitions mutate it and merely mark
it dirty, and a bounded-staleness flusher (:data:`WORKFLOW_JSON_MAX_STALENESS_S`)
writes the full document. Without this, every transition rewrote the whole
file: O(N²) bytes for an N-element ``wf.parallel``. Task failures, the
execution terminal state (:func:`mark_workflow_finished`) and the runtime's
``finally``-path :func:`close_execution_document` flush synchronously, so the
crash window can only lose recent NON-terminal node records (resume recomputes
those by design) — never the terminal state. Callers that never open the
document (standalone tooling/tests) keep the legacy synchronous
read-modify-write semantics.

Atomic writes route through the cross-layer
:func:`molexp.atomicio.atomic_write_json` primitive (the source that
``workspace.atomic_write_json`` itself re-exports), so the atomicity guarantee
is shared infra, not a workflow-layer reinvention — and the workflow layer no
longer needs a runtime import of ``workspace`` for it.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.atomicio import atomic_write_json

from ..._typing import JSONValue

if TYPE_CHECKING:
    from collections.abc import Mapping

    from molexp.workspace.run import Run

    from ..compiled import CompiledWorkflow
    from ..protocols import TaskOutput
    from ..snapshot import TaskSnapshot

logger = get_logger(__name__)

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


def _is_json_safe(value: Any) -> bool:  # noqa: ANN401
    """True iff *value* round-trips through ``json.dumps`` without coercion."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _jsonable(value: Any) -> JSONValue:  # noqa: ANN401
    """Return a compact JSON-safe representation for workflow observability.

    LOSSY for non-JSON-safe values (truncated to 20 keys/items, remainder
    str-ified). Callers persisting task outputs must record the fidelity
    flag (see :func:`mark_task_status` ``outputs_lossy``) so the resume
    seeding path never trusts a truncated value as a real task output.
    """
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
    objects. Tasks whose persisted output is flagged ``outputs_lossy`` (the
    original was not JSON-safe, so the stored value is truncated/str-ified) are
    SKIPPED with a warning — seeding a truncated value would silently corrupt
    downstream tasks; those nodes are recomputed instead (the content-addressed
    cache may opportunistically hit).

    The returned seeds are name+value only. Code-identity verification
    (was this output produced by the same task code/config?) happens at
    seed time in :func:`filter_resume_seeds`, driven by the engine.

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
        if name is None:
            continue
        if task.get("outputs_lossy"):
            logger.warning(
                f"resume: persisted output of task {name!r} in execution "
                f"{execution_id!r} is lossy (the original value was not "
                f"JSON-safe and was truncated for observability); the node "
                f"will be recomputed instead of seeded"
            )
            continue
        outputs[name] = task["outputs"]
    return outputs


def last_resumable_execution_id(run: Run) -> str | None:
    """Return the execution_id of the most recent non-succeeded execution.

    ``resume`` reopens this execution and seeds it with the node outputs already
    persisted there. Returns ``None`` when the run has no execution to reopen —
    the caller errors (no fallback to a fresh execution).
    """
    for record in reversed(run.metadata.execution_history):
        if record.status != "succeeded":
            return record.execution_id
    return None


def seed_from_execution(run: Run) -> tuple[str | None, dict[str, TaskOutput] | None]:
    """Build ``resume`` seeds from *run*'s last resumable execution.

    Reopens the most recent non-succeeded execution (see
    :func:`last_resumable_execution_id`) and reads its persisted completed-node
    outputs via :func:`read_node_outputs`. Returns
    ``(execution_id, seed_outputs)`` for
    :meth:`molexp.workflow.WorkflowRuntime.execute`.

    A pending run (no execution yet) has nothing to reopen — its first
    execution runs fresh (``(None, None)``). That is not a fallback: there is
    no prior attempt to fall back from. An execution that crashed before any
    node finished yields ``(execution_id, None)`` — reopen and recompute all
    nodes within the same execution; not a fallback either. Code-identity
    verification of the seeds happens later, at the engine's
    :func:`filter_resume_seeds` gate.
    """
    execution_id = last_resumable_execution_id(run)
    if execution_id is None:
        return None, None
    return execution_id, read_node_outputs(run.run_dir, execution_id) or None


def filter_resume_seeds(
    run_dir: str | os.PathLike[str],
    execution_id: str,
    seeds: Mapping[str, TaskOutput],
    snapshots: Mapping[str, TaskSnapshot],
) -> dict[str, TaskOutput]:
    """Drop resume seeds the persisted execution document cannot vouch for.

    Called by the engine (``WorkflowRuntime.execute``) BEFORE the execution's
    ``workflow.json`` is rewritten. For every seed it checks the prior
    document's per-task record:

    * persisted ``snapshot_key`` differs from the live task's recomputed
      :class:`~molexp.workflow.snapshot.TaskSnapshot` key → the task's code or
      config changed between attempts; seeding the old output would resurrect
      a stale result. DROP — the node is recomputed.
    * ``outputs_lossy`` is set → the stored value is a truncated
      observability rendering, not the real output. DROP — recomputed.
    * no persisted ``snapshot_key`` (pre-upgrade ``workflow.json``) or no
      per-task record at all → cannot verify. DROP — recomputed (backward
      compatible: old documents resume by recomputation, never by trusting
      an unverifiable value).

    Every drop logs one warning naming the node and the reason; dropping is
    never an error (the node simply recomputes; the content-addressed cache
    may still hit). When the prior document is missing or malformed the seeds
    did not come from it — they pass through unchanged (e.g. programmatic
    ``seed_outputs`` into a fresh execution id). Unknown seed *names* are not
    this function's concern: the runtime fail-fast-validates them against the
    compiled spec before any IO.
    """
    wf_path = _workflow_json_path(Path(run_dir), execution_id)
    if not wf_path.exists():
        return dict(seeds)
    try:
        data = json.loads(wf_path.read_text())
    except (OSError, ValueError):
        return dict(seeds)
    if not isinstance(data, dict):
        return dict(seeds)

    tasks_by_name: dict[str, dict[str, JSONValue]] = {}
    for task in _iter_dicts(data.get("task_configs", [])):
        name = _task_id(task)
        if name is not None:
            tasks_by_name[name] = task

    def _drop(name: str, why: str) -> None:
        logger.warning(
            f"resume: dropping seed for node {name!r} in execution "
            f"{execution_id!r} — {why}; the node will be recomputed"
        )

    kept: dict[str, TaskOutput] = {}
    for name, value in seeds.items():
        record = tasks_by_name.get(name)
        if record is None:
            _drop(name, "the persisted execution document has no record for it")
            continue
        if record.get("outputs_lossy"):
            _drop(name, "its persisted output is lossy (original was not JSON-safe)")
            continue
        persisted_key = record.get("snapshot_key")
        if not isinstance(persisted_key, str) or not persisted_key:
            _drop(
                name,
                "the persisted record carries no snapshot key (pre-upgrade "
                "workflow.json) so the output cannot be verified against the "
                "current task code",
            )
            continue
        live = snapshots.get(name)
        if live is None:
            _drop(name, "the current workflow has no snapshot to verify it against")
            continue
        if persisted_key != live.key:
            _drop(
                name,
                "the task's code or config changed since the output was "
                "persisted (snapshot key mismatch)",
            )
            continue
        kept[name] = value
    return kept


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


# ── Coalescing execution-document writer ─────────────────────────────────────

#: Maximum staleness (seconds) of the on-disk ``workflow.json`` relative to
#: the in-memory authoritative document while an execution is live.
#:
#: This is a PERFORMANCE knob, NOT a correctness gate. Per-task status
#: transitions mutate the in-memory document and only schedule a flush;
#: coalescing them turns the per-transition full-document rewrite (O(N²)
#: bytes for an N-element fan-out) into O(N). Nothing in engine coordination
#: ever waits on this value — the engine's no-timing-constants pin
#: (``test_pg_lowering.py::test_no_timing_constants_for_coordination``) stays
#: intact: the flusher is a daemon ``threading.Timer`` on the observability
#: write path only, never a coroutine the scheduler blocks on. Correctness is
#: carried entirely by the MANDATORY synchronous flushes: task failure
#: (``mark_task_status(status="failed")``), execution terminal states
#: (:func:`mark_workflow_finished`), and the runtime's ``finally``-path
#: :func:`close_execution_document`. Crash-window semantics: a hard crash may
#: lose up to this much of the most recent non-terminal node state (resume
#: recomputes unverifiable nodes by design); terminal states are never
#: deferred.
WORKFLOW_JSON_MAX_STALENESS_S: float = 0.2


class _ExecutionDocumentWriter:
    """One live execution's authoritative in-memory document + flush state.

    All mutation and serialization happen under ``self._lock``: marks arrive
    on the event-loop thread, the staleness timer fires on its own daemon
    thread. Disk writes go through workspace's :func:`atomic_write_json`
    (tmp + rename), so readers never observe a torn document.
    """

    def __init__(self, path: Path, document: dict[str, JSONValue]) -> None:
        self._path = path
        self._document = document
        self._lock = threading.Lock()
        self._dirty = False
        self._closed = False
        self._timer: threading.Timer | None = None

    def mutate(self, apply: Callable[[dict[str, JSONValue]], None], *, flush: bool) -> None:
        """Apply *apply* in memory; flush synchronously or within the staleness bound."""
        with self._lock:
            if self._closed:
                return
            apply(self._document)
            self._dirty = True
            if flush or WORKFLOW_JSON_MAX_STALENESS_S <= 0:
                self._flush_locked()
            elif self._timer is None:
                timer = threading.Timer(WORKFLOW_JSON_MAX_STALENESS_S, self._flush_on_timer)
                timer.daemon = True
                self._timer = timer
                timer.start()

    def _flush_on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if self._closed or not self._dirty:
                return
            try:
                self._flush_locked()
            except Exception as exc:
                logger.warning(
                    f"coalesced workflow.json flush failed for {self._path}: "
                    f"{type(exc).__name__}: {exc}"
                )

    def _flush_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        atomic_write_json(self._path, self._document)
        self._dirty = False

    def close(self) -> None:
        """Final flush (if dirty) + stop the timer; the writer is dead after.

        Swallows (logs) write errors — ``close`` runs on the runtime's
        ``finally`` path and must never mask the engine's own exception.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._dirty:
                    self._flush_locked()
            except Exception as exc:
                logger.warning(
                    f"final workflow.json flush failed for {self._path}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None

    def discard(self) -> None:
        """Drop pending state WITHOUT writing — the document was superseded."""
        with self._lock:
            self._closed = True
            self._dirty = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


_WRITERS: dict[Path, _ExecutionDocumentWriter] = {}
_REGISTRY_LOCK = threading.Lock()


def _writer_for(run_dir: Path, execution_id: str) -> _ExecutionDocumentWriter | None:
    path = _workflow_json_path(run_dir, execution_id)
    with _REGISTRY_LOCK:
        return _WRITERS.get(path)


def open_execution_document(
    run_dir: Path,
    execution_id: str,
    *,
    compiled: CompiledWorkflow | None = None,
) -> None:
    """Begin a coalesced-writer lifecycle for one execution.

    Writes the initial document synchronously (the execution directory always
    exists post-open) and registers the in-memory copy as authoritative:
    subsequent :func:`mark_task_status` / :func:`mark_workflow_finished` calls
    mutate it in memory and flush at bounded staleness instead of rewriting
    the file per transition. Callers MUST pair this with
    :func:`close_execution_document` (the runtime does so in a ``finally``)
    so the last document state always lands on disk.

    Reopening a path that already has a live writer (in-process resume of the
    same execution id) discards the superseded writer without flushing it —
    the fresh initial document is the new truth.
    """
    exec_dir = run_dir / "executions" / execution_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    path = exec_dir / "workflow.json"
    document = _initial_document(execution_id, compiled)
    with _REGISTRY_LOCK:
        prior = _WRITERS.pop(path, None)
    if prior is not None:
        prior.discard()
    atomic_write_json(path, document)
    with _REGISTRY_LOCK:
        _WRITERS[path] = _ExecutionDocumentWriter(path, document)


def close_execution_document(run_dir: Path | None, execution_id: str | None) -> None:
    """End a writer lifecycle: flush pending state and unregister.

    Idempotent and ``None``-tolerant so the runtime can call it from a
    ``finally`` regardless of how the execution ended (including engine
    raises the except-arms never see); a no-persist (SubWorkflow inner)
    execution never opened a writer, so this is a no-op there.
    """
    if run_dir is None or execution_id is None:
        return
    path = _workflow_json_path(run_dir, execution_id)
    with _REGISTRY_LOCK:
        writer = _WRITERS.pop(path, None)
    if writer is not None:
        writer.close()


def _mutate_document(
    run_dir: Path | None,
    execution_id: str | None,
    mutate: Callable[[dict[str, JSONValue]], None],
    *,
    flush: bool = False,
) -> None:
    """Apply *mutate* to the execution document.

    Routed through the registered in-memory writer when the execution was
    opened via :func:`open_execution_document` (coalesced flush; ``flush=True``
    forces the mandatory synchronous write — terminal/failure paths).
    Otherwise falls back to the legacy synchronous read-modify-write so
    standalone callers and pre-existing documents keep their semantics.
    """
    if run_dir is None or execution_id is None:
        return
    writer = _writer_for(run_dir, execution_id)
    if writer is not None:
        writer.mutate(mutate, flush=flush)
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
    snapshot_key: str | None = None,
) -> None:
    """Update one task and adjacent links in ``workflow.json``.

    ``snapshot_key`` (the task's :class:`~molexp.workflow.snapshot.TaskSnapshot`
    content key) is persisted alongside a completed task's outputs so resume
    seeding can verify the persisted value was produced by the SAME code +
    config (see :func:`filter_resume_seeds`). When the output is not
    JSON-safe, the stored value went through the lossy :func:`_jsonable`
    path and ``outputs_lossy: true`` is recorded — such values are
    observability-only and never eligible as resume seeds.

    Non-terminal transitions are coalesced (see
    :data:`WORKFLOW_JSON_MAX_STALENESS_S`); a ``"failed"`` status flushes
    synchronously — a failure record must land on disk immediately.
    """

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
                    lossy = not _is_json_safe(output)
                    task["outputs"] = _jsonable(output)
                    if lossy:
                        task["outputs_lossy"] = True
                    else:
                        task.pop("outputs_lossy", None)
                if snapshot_key is not None:
                    task["snapshot_key"] = snapshot_key
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

    _mutate_document(run_dir, execution_id, _apply, flush=status == "failed")


def mark_workflow_finished(
    run_dir: Path,
    execution_id: str,
    *,
    status: str,
    outputs: Any = None,  # noqa: ANN401
    error: str | None = None,
) -> None:
    """Mark the execution document terminal.

    MANDATORY synchronous flush — the terminal state is the one record the
    crash-window semantics never allow to be lost — and the end of the
    coalesced-writer lifecycle (the writer is closed/unregistered, so the
    runtime's ``finally``-path :func:`close_execution_document` is a no-op
    on the normal paths).
    """

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

    _mutate_document(run_dir, execution_id, _apply, flush=True)
    close_execution_document(run_dir, execution_id)
