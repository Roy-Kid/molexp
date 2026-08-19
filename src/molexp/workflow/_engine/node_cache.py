"""Content-addressed result caching for per-task node bodies.

Wraps :func:`molexp.workflow._engine.node.run_task_body` with the
cache get/put dance: collect + JSON-safe the inputs, look up by
``(snapshot, inputs)``, re-register cached artifacts on a hit, and store the
result + produced-artifact manifest on a miss. Kept apart from the dispatch
core in :mod:`.node`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mollog import get_logger

from ..protocols import JSONValue, TaskInput, TaskOutput
from ..types import UnknownTaskError
from .node import NO_OUTPUT, _collect_upstream_outputs, run_task_body
from .state import WorkflowDeps, WorkflowState

logger = get_logger(__name__)

# Once-per-(execution, task) dedup for promoted cache-put failure warnings.
# Graceful degradation stays (the run continues uncached), but a permanently
# failing cache backend (permissions, full disk) must surface at WARNING
# instead of vanishing at debug level. Keyed on ``(id(deps), task_name)`` —
# ``WorkflowDeps`` is built fresh per execution, so each run warns at most
# once per task. Bounded so a long-lived process never grows it unboundedly.
_PUT_FAILURE_WARNED: set[tuple[int, str]] = set()
_PUT_FAILURE_WARNED_MAX = 4096


def _is_json_safe(value: object) -> bool:
    """Return True iff *value* round-trips through ``json.dumps`` cleanly."""
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _canonical_root_inputs(value: TaskOutput) -> TaskOutput:
    """Canonicalize one engine-injected root-inputs value for cache identity.

    The content-addressed ``workdir`` Path is excluded: it varies per
    workspace / execution but never changes task semantics (the body's
    behavior is a function of params + upstream data, not of *where* it
    scratches). Run params — and any SubWorkflow-forwarded keys merged into
    the root entry — MUST stay in, so a parameter sweep never collides on
    one cache entry.
    """
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "workdir"}
    return value


def _cache_inputs(
    name: str,
    state: WorkflowState,
    upstream: TaskInput,
    delivered: TaskInput = NO_OUTPUT,
) -> dict[str, JSONValue]:
    """Build the cache ``inputs`` mapping — the task's FULL runtime-input identity.

    ``{"inputs": <upstream>}`` is the shipped key shape for plain tasks
    (unchanged, so existing cache entries stay valid). When the engine
    injected root inputs for *name* (sweep params + workdir for a workspace
    run, possibly merged with a SubWorkflow-forwarded value), they are folded
    in under a separate ``"root_inputs"`` key — because the body binds its
    named parameters from ``state.root_inputs[name]``, those values are part
    of the task's cache identity. The workdir Path is canonicalized OUT (see
    :func:`_canonical_root_inputs`). A trigger-*delivered* value (branch-routed
    / loop-back input for a dep-less task) likewise joins the identity under a
    ``"delivered"`` key — two different routed values must never share a cache
    entry. Determinism: the downstream ``Caching._compute_input_hash``
    serializes with ``sort_keys=True``, so key insertion order never moves the
    hash.
    """
    payload: dict[str, JSONValue] = {"inputs": cast("JSONValue", upstream)}
    if name in state.root_inputs:
        root = _canonical_root_inputs(state.root_inputs[name])
        payload["root_inputs"] = cast("JSONValue", root)
    if delivered is not NO_OUTPUT:
        payload["delivered"] = cast("JSONValue", delivered)
    return payload


def _artifact_manifest(deps: WorkflowDeps, name: str) -> list[dict[str, JSONValue]]:
    """Build the JSON artifact manifest for task *name* in the current run.

    Queries the current run's catalog view for artifacts whose producer
    task is *name* and snapshots each as a JSON dict
    ``{name, kind, content_hash, asset_id}``. Returns ``[]`` when no
    workspace asset view is reachable.
    """
    run_context = deps.run_context
    if run_context is None:
        return []
    # The scope-filtered asset view lives on the Run (``run_context.run``);
    # fall back to a direct ``.assets`` on the context for duck-typed stubs.
    run = getattr(run_context, "run", None)
    assets_view = getattr(run, "assets", None) or getattr(run_context, "assets", None)
    query = getattr(assets_view, "query", None)
    if not callable(query):
        return []
    try:
        found = query(producer_task=name, kind="artifact")
    except Exception:
        return []
    manifest: list[dict[str, JSONValue]] = []
    for asset in found or []:
        content_hash = getattr(asset, "content_hash", None)
        if not content_hash:
            continue
        path = getattr(asset, "path", None)
        manifest.append(
            {
                "name": getattr(asset, "name", None),
                "kind": getattr(asset, "kind", "artifact"),
                "content_hash": content_hash,
                "asset_id": getattr(asset, "asset_id", None),
                "mime": getattr(asset, "mime", None),
                "tags": getattr(asset, "tags", None) or {},
                "path": str(path) if path is not None else None,
            }
        )
    return manifest


def _upstream_asset_ids(deps: WorkflowDeps, registration: object) -> tuple[str, ...]:
    """Project the workflow DAG into asset lineage: upstream artifact ids.

    For each declared upstream task of *registration* (the same
    ``depends_on`` set :func:`_collect_upstream_outputs` reads), collect the
    ``asset_id``s of the artifacts that upstream registered **in this run's
    manifests** (the existing ``producer_task=`` query — no new attribution
    mechanism). Deduped, stable order.

    Semantics (vision-loop-09): edges reference per-run manifest entries —
    resume-seeded tasks whose artifacts were registered by the prior attempt
    of the same run still resolve (same manifest); an upstream that persisted
    nothing (non-JSON-safe output, no run context) contributes no edge,
    honestly reflecting that nothing was persisted for it. Root tasks (no
    upstreams) return ``()`` — a root artifact's param identity stays
    ``config_hash``'s job (no fourth id layer).
    """
    ids: list[str] = []
    for upstream in getattr(registration, "depends_on", ()) or ():
        try:
            entries = _artifact_manifest(deps, upstream)
        except Exception:  # fail-soft: lineage rides the persistence bonus channel
            logger.debug(f"lineage: upstream query for {upstream!r} skipped")
            continue
        for entry in entries:
            asset_id = entry.get("asset_id")
            if isinstance(asset_id, str) and asset_id and asset_id not in ids:
                ids.append(asset_id)
    return tuple(ids)


def _put_file_blobs(deps: WorkflowDeps, manifest: list[dict]) -> None:
    """Copy registered file bytes into the cache blob store (cache axis)."""
    store = getattr(getattr(deps, "cache", None), "store", None)
    put_blob = getattr(store, "put_blob", None)
    if not callable(put_blob) or not manifest:
        return
    run_dir = getattr(deps.run_context, "run_dir", None) or getattr(deps, "run_dir", None)
    if run_dir is None:
        return
    root = Path(run_dir)
    for entry in manifest:
        content_hash = entry.get("content_hash")
        rel = entry.get("path")
        if not content_hash or not rel:
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            put_blob(str(content_hash), path.read_bytes())
        except Exception:
            logger.debug(f"cache: blob put for {entry.get('name')!r} skipped")


def _restore_cached_files(
    deps: WorkflowDeps,
    manifest: list[dict],
    *,
    consumed: tuple[str, ...] = (),
) -> None:
    """Publish cached file blobs onto the current run via register_artifact."""
    run_context = deps.run_context
    register = getattr(run_context, "register_artifact", None)
    store = getattr(getattr(deps, "cache", None), "store", None)
    get_blob = getattr(store, "get_blob", None)
    if not callable(register) or not callable(get_blob) or not manifest:
        return
    for entry in manifest:
        content_hash = entry.get("content_hash")
        name = entry.get("name")
        if not content_hash or not name:
            continue
        try:
            blob = get_blob(str(content_hash))
        except Exception:
            blob = None
        if blob is None:
            continue
        tags = entry.get("tags")
        try:
            register(
                blob,
                name=str(name),
                mime=entry.get("mime"),
                tags=tags if isinstance(tags, dict) else None,
                consumed=list(consumed) or None,
            )
        except Exception:
            logger.debug(f"cache: restore of {name!r} skipped")


async def run_task_body_cached(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
    *,
    delivered: TaskInput = NO_OUTPUT,
) -> TaskOutput:
    """Run task *name*'s body with content-addressed result caching.

    Gating (caller pre-checks ``deps.cache is not None``, non-actor task,
    ``name in deps.snapshots``):

    * collect the upstream inputs once and wrap them — together with any
      engine-injected root inputs for this task (sweep params; the workdir
      Path is canonicalized out) — as the cache ``inputs`` payload;
    * ``cache.get`` → on HIT, re-register the cached artifact manifest into
      the current run and return the recorded ``result`` WITHOUT running the
      body (the per-task body counter must not increment);
    * on MISS, run the body, assemble the produced-artifact manifest, and
      ``cache.put({"result": raw, "artifacts": manifest})``. Non-JSON-safe
      inputs / results degrade gracefully — the body still runs and the put
      is skipped.

    The returned raw value is routed by the caller through the SAME
    ``_classify_return`` path as a plain return, so branch / End semantics
    hold identically on hits and misses.
    """
    registration = deps.registration_by_name.get(name)
    if registration is None:
        raise UnknownTaskError(f"run_task_body_cached: unknown task {name!r}")

    cache = deps.cache
    snapshot = deps.snapshots.get(name)
    assert cache is not None and snapshot is not None  # caller-gated

    inputs = _collect_upstream_outputs(registration, state)
    cache_inputs = _cache_inputs(name, state, inputs, delivered)
    cacheable = _is_json_safe(cache_inputs)

    # ``bypass_cache`` (the --fresh escape hatch) skips the READ only: the
    # body always runs, and the fresh result still lands in the cache below.
    if cacheable and not deps.bypass_cache:
        try:
            payload = cache.get(snapshot, cache_inputs)
        except Exception:
            payload = None
        if payload is not None:
            files = payload.get("files", payload.get("artifacts", []))
            if isinstance(files, list):
                set_active = getattr(deps.run_context, "set_active_task", None)
                if callable(set_active):
                    set_active(name)
                try:
                    consumed = _upstream_asset_ids(deps, registration)
                except Exception:
                    consumed = ()
                _restore_cached_files(
                    deps,
                    [a for a in files if isinstance(a, dict)],
                    consumed=consumed,
                )
            return payload.get("result")

    raw = await run_task_body(name, deps, state, delivered=delivered)

    if cacheable and _is_json_safe(raw):
        manifest = _artifact_manifest(deps, name)
        _put_file_blobs(deps, manifest)
        result_payload = cast("dict[str, JSONValue]", {"result": raw, "files": manifest})
        try:
            cache.put(snapshot, cache_inputs, result_payload)
        except Exception as exc:
            # JSON-safety is pre-checked above, so an exception here is a real
            # store failure (permissions, full disk, …) — promote the FIRST
            # one per (execution, task) to WARNING so a permanently failing
            # cache is visible; repeats stay at debug. The run continues
            # uncached either way (graceful degradation).
            warn_key = (id(deps), name)
            if warn_key not in _PUT_FAILURE_WARNED:
                if len(_PUT_FAILURE_WARNED) >= _PUT_FAILURE_WARNED_MAX:
                    _PUT_FAILURE_WARNED.clear()
                _PUT_FAILURE_WARNED.add(warn_key)
                logger.warning(
                    f"cache: put for task {name!r} failed "
                    f"({type(exc).__name__}: {exc}); result caching is degraded "
                    f"for this task — the workflow continues uncached"
                )
            else:
                logger.debug(f"cache: put for task {name!r} failed again (suppressed)")
    return raw
