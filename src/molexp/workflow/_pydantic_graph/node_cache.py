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
    in under a separate ``"root_inputs"`` key — because the body consumes
    ``state.root_inputs[name]`` as its ``ctx.inputs``, those values are part
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

    raw = await run_task_body(name, deps, state, delivered=delivered)

    # Engine materialization: persist the task's return value as a content-hashed
    # artifact (fail-soft) — the live caller of the materialization layer. Runs
    # before the cache put so the artifact manifest captures it too.
    materialization = getattr(deps, "materialization", None)
    if materialization is not None:
        try:
            materialization.persist_result(name, raw, run_context=deps.run_context)
        except Exception:
            logger.debug(f"materialize: persist for task {name!r} skipped")

    if cacheable and _is_json_safe(raw):
        manifest = _artifact_manifest(deps, name)
        result_payload = cast("dict[str, JSONValue]", {"result": raw, "artifacts": manifest})
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
