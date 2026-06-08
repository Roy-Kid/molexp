"""Cache-identity contract: code_hash + config_hash + inputs_hash.

Phase 01 of the pure-task-context chain. These tests *pin* shipped behavior
(they are green on first run — there is no new code) so that later phases can
rely on a proven forward guarantee: any value delivered to a task via the
``inputs`` channel — including, in later phases, sweep params injected at root
nodes and a content-addressed workdir ``Path`` — flows through
``Caching._compute_input_hash`` into the final ``cache_key``.

Contract, stated as four pins:

1. ``inputs`` participate in ``cache_key`` — differing inputs ⇒ different key.
2. Identical code + config + inputs collide on one ``cache_key`` (reuse).
3. A ``pathlib.Path`` carried through ``inputs`` hashes stably (no
   memory-address nondeterminism), via the ``_robust_json_default`` Path branch.
4. ``TaskSnapshot.key`` stays ``f"{code_hash}:{config_hash}"`` — ``inputs`` are
   NOT folded into the snapshot identity; the cache, not the snapshot, owns the
   inputs term.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow.cache import Caching
from molexp.workflow.snapshot import TaskSnapshot


async def _body() -> dict[str, int]:
    """A trivial task body to snapshot (its source is AST-hashed)."""
    return {"x": 1}


def _snapshot(task_id: str = "t", config: dict | None = None) -> TaskSnapshot:
    return TaskSnapshot.from_task_body(task_id, _body, config_data=config)


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
    # Same body + same config ⇒ identical key regardless of any runtime inputs.
    s1 = _snapshot(config={"k": "v"})
    s2 = _snapshot(config={"k": "v"})
    assert s1.key == s2.key
    # Config DOES move the key (it is part of identity); inputs never reach here.
    s3 = _snapshot(config={"k": "other"})
    assert s3.key != s1.key
