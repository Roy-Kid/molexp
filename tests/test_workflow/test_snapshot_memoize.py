"""TaskSnapshot AST-hash memoization (workflow-workspace-hardening P2-5).

``TaskSnapshot.from_task_body`` AST-parses + normalizes + hashes the body's
source. ``codec.ir_to_spec`` re-snapshots every task on every IR round-trip, so
the source hash — which depends only on the body's code object — is memoized by
code object: the same function is AST-parsed once, not once per snapshot.
"""

from __future__ import annotations

import molexp.workflow.snapshot as snap_mod
from molexp.workflow.snapshot import TaskSnapshot


class TestSourceHashMemoization:
    def test_ast_parsed_once_per_code_object(self, monkeypatch):
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
