"""Tests for ``derive_run_id`` — the lifted content-addressed run-id helper.

Covers acceptance criterion ac-004: ``derive_run_id`` is deterministic,
order-invariant over dict keys, distinct-on-distinct-params, and emits a
16-char lowercase-hex string.
"""

from __future__ import annotations

import string

from molexp.workspace.utils import derive_run_id


def test_basics_returns_str() -> None:
    """Basics — a simple param dict produces a string id."""
    assert isinstance(derive_run_id({"lr": 1e-4}), str)


def test_order_invariant_same_id_regardless_of_key_insertion_order() -> None:
    """ac-004 — key insertion order does not change the derived id."""
    assert derive_run_id({"a": 1, "b": 2}) == derive_run_id({"b": 2, "a": 1})


def test_distinct_dicts_produce_distinct_ids() -> None:
    """ac-004 — different param dicts derive different ids."""
    assert derive_run_id({"a": 1}) != derive_run_id({"a": 2})


def test_output_is_16_char_hex() -> None:
    """ac-004 — output is exactly 16 lowercase-hex characters."""
    run_id = derive_run_id({"lr": 1e-4, "batch": 32})
    assert len(run_id) == 16
    assert all(c in string.hexdigits.lower() for c in run_id)
    assert all(c in "0123456789abcdef" for c in run_id)


def test_deterministic_repeated_calls() -> None:
    """ac-004 — same dict yields the same id across repeated calls."""
    params = {"lr": 5e-4, "batch": 64}
    assert derive_run_id(params) == derive_run_id(dict(params))


def test_length_parameter_controls_output_length() -> None:
    """Edge — the ``length`` keyword controls the truncation width."""
    assert len(derive_run_id({"a": 1}, length=8)) == 8
    assert len(derive_run_id({"a": 1}, length=32)) == 32
