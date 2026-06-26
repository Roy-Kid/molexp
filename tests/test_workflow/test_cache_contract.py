"""Cache-identity contract: code_hash + config_hash + inputs_hash.

Phase 01 of the pure-task-context chain pinned the ``Caching``-seam half of
this contract; sweep-param injection has since landed, so the engine half is
now LIVE and pinned here too: the engine-injected root inputs (run params —
plus any SubWorkflow-forwarded keys merged into the root entry) are folded
into the cache identity by ``node_cache._cache_inputs``, while the
content-addressed workdir ``Path`` is canonicalized OUT (it varies per
workspace/execution without changing task semantics).

Contract, stated as six pins:

1. ``inputs`` participate in ``cache_key`` — differing inputs ⇒ different key.
2. Identical code + config + inputs collide on one ``cache_key`` (reuse).
3. A ``pathlib.Path`` carried through ``inputs`` hashes stably (no
   memory-address nondeterminism), via the ``_robust_json_default`` Path branch.
4. ``TaskSnapshot.key`` stays ``f"{code_hash}:{config_hash}"`` — ``inputs`` are
   NOT folded into the snapshot identity; the cache, not the snapshot, owns the
   inputs term.
5. Engine-injected root inputs (sweep params) participate in the cache key —
   two runs with different params NEVER share a root-task cache entry.
6. The injected workdir Path does NOT participate — same params with a
   different workdir/execution still HIT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import Task, TaskContext, WorkflowCompiler, WorkflowRuntime
from molexp.workflow.cache import Caching
from molexp.workflow.snapshot import TaskSnapshot
from molexp.workspace import Workspace


class _Body(Task):
    """A trivial task whose ``__init__`` arg is its build-time config."""

    def __init__(self, k: str = "v") -> None:
        self.k = k

    async def execute(self, ctx: TaskContext) -> dict[str, int]:
        return {"x": 1}


def _snapshot(task_id: str = "t", *, k: str = "v") -> TaskSnapshot:
    # Config is the instance's captured __init__ args — not a registration dict.
    return TaskSnapshot.from_task_body(task_id, _Body(k))


# ── (a) inputs participate in cache_key ────────────────────────────────────────


def test_differing_inputs_yield_different_cache_key() -> None:
    snap = _snapshot()
    key_a = Caching._compute_cache_key(snap.key, Caching._compute_input_hash({"n": 1}))
    key_b = Caching._compute_cache_key(snap.key, Caching._compute_input_hash({"n": 2}))
    assert key_a != key_b


def test_put_then_get_with_different_inputs_misses(tmp_path: Path) -> None:
    cache = Caching(store_dir=tmp_path)
    snap = _snapshot()
    cache.put(snap, {"n": 1}, {"result": "A"})
    assert cache.get(snap, {"n": 1}) == {"result": "A"}  # hit on same inputs
    assert cache.get(snap, {"n": 2}) is None  # different inputs ⇒ MISS


# ── (b) reuse on identical code+config+inputs ──────────────────────────────────


def test_identical_inputs_hit_and_reuse(tmp_path: Path) -> None:
    cache = Caching(store_dir=tmp_path)
    snap = _snapshot()
    cache.put(snap, {"n": 1}, {"result": "A"})
    # A fresh snapshot of the same body+config has the same key; identical
    # inputs must therefore collide on the same cache_key and hit.
    same = _snapshot()
    assert same.key == snap.key
    assert cache.get(same, {"n": 1}) == {"result": "A"}


def test_input_hash_deterministic_for_same_dict() -> None:
    a = Caching._compute_input_hash({"n": 1, "m": 2})
    b = Caching._compute_input_hash({"n": 1, "m": 2})
    assert a == b


def test_input_hash_insensitive_to_key_order() -> None:
    # sort_keys=True ⇒ insertion order is irrelevant.
    a = Caching._compute_input_hash({"n": 1, "m": 2})
    b = Caching._compute_input_hash({"m": 2, "n": 1})
    assert a == b


# ── (c) Path-valued inputs hash stably ─────────────────────────────────────────


def test_path_input_hashes_stably_across_instances() -> None:
    # Two independently constructed Path objects for the same path string must
    # hash identically — proving _robust_json_default uses str(path), not an
    # address-bearing object repr.
    h1 = Caching._compute_input_hash({"workdir": Path("/scratch/abc")})
    h2 = Caching._compute_input_hash({"workdir": Path("/scratch") / "abc"})
    assert h1 == h2


def test_different_paths_hash_differently() -> None:
    h1 = Caching._compute_input_hash({"workdir": Path("/scratch/abc")})
    h2 = Caching._compute_input_hash({"workdir": Path("/scratch/xyz")})
    assert h1 != h2


def test_path_input_does_not_collide_with_equivalent_str() -> None:
    # The {"__type__": "Path"} wrapper keeps Path("x") distinct from str "x".
    h_path = Caching._compute_input_hash({"v": Path("x")})
    h_str = Caching._compute_input_hash({"v": "x"})
    assert h_path != h_str


# ── (d) snapshot identity excludes inputs ──────────────────────────────────────


def test_snapshot_key_is_code_and_config_only() -> None:
    snap = _snapshot()
    assert snap.key == f"{snap.code_hash}:{snap.config_hash}"


def test_snapshot_identity_independent_of_inputs() -> None:
    # from_task_body takes no inputs argument: the snapshot cannot know inputs.
    # Same body + same __init__ config ⇒ identical key regardless of runtime inputs.
    s1 = _snapshot(k="v")
    s2 = _snapshot(k="v")
    assert s1.key == s2.key
    # The instance's __init__ config DOES move the key (it is part of identity);
    # inputs never reach here.
    s3 = _snapshot(k="other")
    assert s3.key != s1.key


# ── (e)+(f) engine-injected root inputs: params in, workdir out ────────────────


def _root_inputs_payload(root: dict) -> dict:
    """The cache ``inputs`` payload for a root task carrying injected *root*."""
    from molexp.workflow._pydantic_graph.node_cache import _cache_inputs
    from molexp.workflow._pydantic_graph.state import WorkflowState

    state = WorkflowState()
    state.root_inputs["t"] = root
    return _cache_inputs("t", state, None)


def test_root_input_params_move_the_input_hash() -> None:
    h1 = Caching._compute_input_hash(
        _root_inputs_payload({"params": {"ratio": "r1"}, "workdir": Path("/m/a")})
    )
    h2 = Caching._compute_input_hash(
        _root_inputs_payload({"params": {"ratio": "r2"}, "workdir": Path("/m/a")})
    )
    assert h1 != h2


def test_root_input_workdir_is_canonicalized_out() -> None:
    # Same params, different content-addressed workdir ⇒ SAME hash (the workdir
    # is execution location, not task identity).
    h1 = Caching._compute_input_hash(
        _root_inputs_payload({"params": {"ratio": "r1"}, "workdir": Path("/m/a")})
    )
    h2 = Caching._compute_input_hash(
        _root_inputs_payload({"params": {"ratio": "r1"}, "workdir": Path("/m/b")})
    )
    assert h1 == h2


def test_plain_task_cache_payload_shape_unchanged() -> None:
    # A task with NO injected root inputs keeps the shipped {"inputs": …}
    # payload — existing cache entries stay valid.
    from molexp.workflow._pydantic_graph.node_cache import _cache_inputs
    from molexp.workflow._pydantic_graph.state import WorkflowState

    assert _cache_inputs("t", WorkflowState(), {"up": 1}) == {"inputs": {"up": 1}}


def _workspace_run(root: Path, name: str, params: dict):
    ws = Workspace(root / f"lab-{name}")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params=params)


@pytest.mark.asyncio
async def test_sweep_runs_with_different_params_never_share_root_cache(
    tmp_path: Path,
) -> None:
    """Regression — the first sweep cell's root result must NOT be served to
    every other cell. Different run params ⇒ root-task cache MISS ⇒ body runs."""
    counters = {"root": 0}
    wf = WorkflowCompiler(name="sweep")

    # Root task binds the run param ``ratio`` by name (engine-injected).
    @wf.task
    async def root(ratio: str) -> str:
        counters["root"] += 1
        return ratio

    compiled = wf.compile()
    cache = Caching(store_dir=tmp_path / "shared-cache")

    run1 = _workspace_run(tmp_path, "a", {"ratio": "r1"})
    with run1.start() as ctx1:
        r1 = await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache)
    run2 = _workspace_run(tmp_path, "b", {"ratio": "r2"})
    with run2.start() as ctx2:
        r2 = await WorkflowRuntime().execute(compiled, run_context=ctx2, cache=cache)

    assert counters["root"] == 2  # both cells computed — no cross-param hit
    assert r1.outputs["root"] == "r1"
    assert r2.outputs["root"] == "r2"  # NOT the first cell's value


@pytest.mark.asyncio
async def test_same_params_different_workdir_and_exec_hits(tmp_path: Path) -> None:
    """Same params in two different workspaces (⇒ different content-addressed
    workdir Paths and execution ids) share one cache entry — the workdir must
    not poison the key."""
    counters = {"root": 0}
    wf = WorkflowCompiler(name="sweep-hit")

    # Root task binds the run param ``ratio`` by name (engine-injected).
    @wf.task
    async def root(ratio: str) -> str:
        counters["root"] += 1
        return ratio

    compiled = wf.compile()
    cache = Caching(store_dir=tmp_path / "shared-cache")

    run1 = _workspace_run(tmp_path, "ws1", {"ratio": "r1"})
    with run1.start() as ctx1:
        r1 = await WorkflowRuntime().execute(compiled, run_context=ctx1, cache=cache)
    run2 = _workspace_run(tmp_path, "ws2", {"ratio": "r1"})
    with run2.start() as ctx2:
        r2 = await WorkflowRuntime().execute(compiled, run_context=ctx2, cache=cache)

    assert counters["root"] == 1  # second run served from cache
    assert r1.outputs["root"] == r2.outputs["root"] == "r1"
