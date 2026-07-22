"""Cache-identity contract: code_hash + config_hash + inputs_hash.

Architectural lock for ``molexp.workflow.cache`` — the cache key is
``f(snapshot.key, inputs_hash)`` and nothing else. Six pins:

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


def _snapshot(*, k: str = "v") -> TaskSnapshot:
    # Config is the instance's captured __init__ args — not a registration dict.
    return TaskSnapshot.from_task_body("t", _Body(k))


def _workspace_run(root: Path, name: str, params: dict):
    ws = Workspace(root / f"lab-{name}")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params=params)


class TestCachingInputHash:
    """``Caching`` folds ``inputs`` (and only inputs) into the cache term."""

    def test_differing_inputs_produce_a_miss(self, tmp_path: Path) -> None:
        """Pin 1 — inputs participate: same snapshot, different inputs ⇒ miss."""
        cache = Caching(store_dir=tmp_path)
        snap = _snapshot()
        cache.put(snap, {"n": 1}, {"result": "A"})
        assert cache.get(snap, {"n": 1}) == {"result": "A"}
        assert cache.get(snap, {"n": 2}) is None

    def test_identical_code_config_inputs_reuse_one_entry(self, tmp_path: Path) -> None:
        """Pin 2 — a fresh snapshot of the same body+config collides and hits."""
        cache = Caching(store_dir=tmp_path)
        snap = _snapshot()
        cache.put(snap, {"n": 1}, {"result": "A"})
        same = _snapshot()
        assert same.key == snap.key
        assert cache.get(same, {"n": 1}) == {"result": "A"}

    def test_input_hash_ignores_key_order(self) -> None:
        """Canonical (sorted) input hashing — insertion order is irrelevant."""
        a = Caching._compute_input_hash({"n": 1, "m": 2})
        b = Caching._compute_input_hash({"m": 2, "n": 1})
        assert a == b

    def test_path_input_hashes_stably_across_instances(self) -> None:
        """Pin 3 — two Path objects for the same path string hash identically
        (``_robust_json_default`` serializes ``str(path)``, not an object repr)."""
        h1 = Caching._compute_input_hash({"workdir": Path("/scratch/abc")})
        h2 = Caching._compute_input_hash({"workdir": Path("/scratch") / "abc"})
        assert h1 == h2

    def test_path_input_distinct_from_equivalent_string(self) -> None:
        """The ``{"__type__": "Path"}`` wrapper keeps ``Path("x")`` distinct from ``"x"``."""
        h_path = Caching._compute_input_hash({"v": Path("x")})
        h_str = Caching._compute_input_hash({"v": "x"})
        assert h_path != h_str


class TestSnapshotExcludesInputs:
    """Pin 4 — ``TaskSnapshot.key`` never folds in runtime inputs."""

    def test_key_moves_with_config_but_never_with_inputs(self) -> None:
        # from_task_body takes no inputs argument: the snapshot cannot know inputs.
        s1 = _snapshot(k="v")
        s2 = _snapshot(k="v")
        assert s1.key == s2.key
        # Build-time config IS part of identity; inputs never reach here.
        s3 = _snapshot(k="other")
        assert s3.key != s1.key


class TestEngineInjectedCacheIdentity:
    """Pins 5 + 6 — engine-injected root inputs: sweep params in, workdir out."""

    @pytest.mark.asyncio
    async def test_differing_run_params_never_share_root_cache(self, tmp_path: Path) -> None:
        """Regression — the first sweep cell's root result must NOT be served to
        every other cell. Different run params ⇒ root-task cache MISS ⇒ body runs."""
        counters = {"root": 0}
        wf = WorkflowCompiler(name="sweep")

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
    async def test_same_params_different_workdir_still_hits(self, tmp_path: Path) -> None:
        """Same params in two workspaces (⇒ different content-addressed workdir
        Paths and execution ids) share one cache entry — workdir never poisons
        the key."""
        counters = {"root": 0}
        wf = WorkflowCompiler(name="sweep-hit")

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
