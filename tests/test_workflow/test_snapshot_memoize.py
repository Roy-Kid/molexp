"""TaskSnapshot AST-hash memoization (workflow-workspace-hardening P2-5).

``TaskSnapshot.from_task_body`` AST-parsed + normalized + hashed the body's
source on *every* call. ``codec.ir_to_spec`` runs ``compile_registrations`` on
every IR deserialization, re-hashing all tasks each time. The source hash
depends only on the body's code object, so it is now memoized by code object —
the same function is AST-parsed once, not once per snapshot.
"""

from __future__ import annotations

import molexp.workflow.snapshot as snap_mod
from molexp.workflow.snapshot import TaskSnapshot


def test_source_hash_memoized_by_code_object(monkeypatch):
    snap_mod._normalized_source_hash.cache_clear()

    calls = {"n": 0}
    orig = snap_mod._normalize_ast

    def counting(src):
        calls["n"] += 1
        return orig(src)

    monkeypatch.setattr(snap_mod, "_normalize_ast", counting)

    async def body(ctx):
        return 1

    first = TaskSnapshot.from_task_body("t0", body).code_hash
    assert calls["n"] == 1  # parsed once

    for i in range(20):
        again = TaskSnapshot.from_task_body(f"t{i}", body).code_hash
        assert again == first  # identical hash for the same body
    assert calls["n"] == 1, "AST re-parsed despite identical code object"


def test_distinct_bodies_hash_distinctly():
    snap_mod._normalized_source_hash.cache_clear()

    async def alpha(ctx):
        return 1

    async def beta(ctx):
        return 2

    assert (
        TaskSnapshot.from_task_body("a", alpha).code_hash
        != TaskSnapshot.from_task_body("b", beta).code_hash
    )


def test_compiled_derived_maps_cached():
    """CompiledWorkflow's static topology maps are derived once and reused
    (identity-stable across accesses), so _build_deps does not rebuild them
    per execution."""
    from molexp.workflow import WorkflowCompiler

    wf = WorkflowCompiler(name="m")

    @wf.task
    async def a(ctx) -> int:
        return 1

    @wf.task(depends_on=["a"])
    async def b(ctx) -> int:
        return 2

    compiled = wf.compile()
    assert compiled.registration_by_name is compiled.registration_by_name
    assert compiled.parallel_decls_by_body is compiled.parallel_decls_by_body
    assert compiled.loop_max_iters is compiled.loop_max_iters
    assert set(compiled.registration_by_name) == {"a", "b"}
