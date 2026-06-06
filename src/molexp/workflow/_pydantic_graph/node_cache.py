"""Content-addressed result caching for per-task node bodies.

Wraps :func:`molexp.workflow._pydantic_graph.node.run_task_body` with the
cache get/put dance: collect + JSON-safe the inputs, look up by
``(snapshot, inputs)``, re-register cached artifacts on a hit, and store the
result + produced-artifact manifest on a miss. Kept apart from the dispatch
core in :mod:`.node`.
"""

from __future__ import annotations

from typing import cast

from mollog import get_logger

from ..protocols import JSONValue, TaskInput, TaskOutput
from ..types import UnknownTaskError
from .node import _collect_upstream_outputs, run_task_body
from .state import WorkflowDeps, WorkflowState

logger = get_logger(__name__)


def _is_json_safe(value: object) -> bool:
    """Return True iff *value* round-trips through ``json.dumps`` cleanly."""
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _cache_inputs(inputs: TaskInput) -> dict[str, JSONValue]:
    """Wrap the collected upstream inputs as the cache ``inputs`` mapping."""
    return {"inputs": inputs}


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
        manifest.append(
            {
                "name": getattr(asset, "name", None),
                "kind": getattr(asset, "kind", "artifact"),
                "content_hash": content_hash,
                "asset_id": getattr(asset, "asset_id", None),
            }
        )
    return manifest


def _reregister_artifacts(deps: WorkflowDeps, name: str, manifest: list[dict]) -> None:
    """Re-register cached artifacts into the current run by content-hash.

    Idempotent catalog upsert keyed on ``(kind, content_hash)`` pointing at
    bytes already present in the content-addressed store — no recompute, no
    byte recopy. Entries whose bytes are absent (fresh workspace) are
    skipped gracefully. Reached purely through duck-typed ``run_context``
    surface so the workflow layer keeps its decoupling from a concrete
    workspace import.
    """
    run_context = deps.run_context
    if run_context is None or not manifest:
        return
    run = getattr(run_context, "run", None)
    scope = getattr(run, "scope", None)
    # Reach the workspace catalog through the run's ancestry (duck-typed).
    experiment = getattr(run, "experiment", None)
    project = getattr(experiment, "project", None)
    workspace = getattr(project, "workspace", None)
    catalog = getattr(workspace, "catalog", None)
    reregister = getattr(catalog, "reregister_artifact", None)
    if scope is None or not callable(reregister):
        return
    for entry in manifest:
        content_hash = entry.get("content_hash")
        if not content_hash:
            continue
        try:
            reregister(
                name=entry.get("name"),
                content_hash=content_hash,
                target_scope=scope,
                producer_task=name,
            )
        except Exception:
            logger.debug(f"cache: re-register of artifact {entry.get('name')!r} skipped")


async def run_task_body_cached(
    name: str,
    deps: WorkflowDeps,
    state: WorkflowState,
) -> TaskOutput:
    """Run task *name*'s body with content-addressed result caching.

    Gating (caller pre-checks ``deps.cache is not None``, non-actor task,
    ``name in deps.snapshots``):

    * collect the upstream inputs once and wrap them JSON-safely as the
      cache ``inputs`` payload;
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
    cacheable = _is_json_safe(inputs)
    cache_inputs = _cache_inputs(inputs)

    if cacheable:
        try:
            payload = cache.get(snapshot, cache_inputs)
        except Exception:
            payload = None
        if payload is not None:
            artifacts = payload.get("artifacts", [])
            if isinstance(artifacts, list):
                _reregister_artifacts(deps, name, [a for a in artifacts if isinstance(a, dict)])
            return payload.get("result")

    raw = await run_task_body(name, deps, state)

    if cacheable and _is_json_safe(raw):
        manifest = _artifact_manifest(deps, name)
        result_payload = cast("dict[str, JSONValue]", {"result": raw, "artifacts": manifest})
        try:
            cache.put(snapshot, cache_inputs, result_payload)
        except Exception:
            logger.debug(f"cache: put for task {name!r} skipped (non-serializable)")
    return raw
