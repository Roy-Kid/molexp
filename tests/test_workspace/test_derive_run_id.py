"""Tests for ``derive_run_id`` — the content-addressed run-id helper.

``derive_run_id`` (``molexp.workspace.utils``) canonicalizes a params dict to a
deterministic, order-invariant, ``length``-char lowercase-hex id; identical
params always map to the same id (run materialization is idempotent).
"""

from __future__ import annotations

from molexp.workspace.utils import derive_run_id


class TestDeriveRunId:
    def test_id_is_order_invariant_over_key_insertion(self) -> None:
        assert derive_run_id({"a": 1, "b": 2}) == derive_run_id({"b": 2, "a": 1})

    def test_distinct_params_produce_distinct_ids(self) -> None:
        assert derive_run_id({"a": 1}) != derive_run_id({"a": 2})

    def test_output_is_16_char_lowercase_hex_by_default(self) -> None:
        run_id = derive_run_id({"lr": 1e-4, "batch": 32})
        assert len(run_id) == 16
        assert all(c in "0123456789abcdef" for c in run_id)

    def test_length_kwarg_controls_truncation_width(self) -> None:
        assert len(derive_run_id({"a": 1}, length=8)) == 8
        assert len(derive_run_id({"a": 1}, length=32)) == 32
