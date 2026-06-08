"""Regression guard for the ``deterministic_run_id`` lift (ac-005).

The CLI's ``deterministic_run_id`` is re-pointed to delegate to
``molexp.workspace.utils.derive_run_id``. This test pins the public
output to the pre-lift formula so the re-point introduces no behavior
change for current callers.
"""

from __future__ import annotations

import hashlib

import pytest

from molexp.cli._common import deterministic_run_id


def _pre_lift(params: dict) -> str:
    """The original sha256-over-sorted-``k=repr(v)`` formula (pre-lift)."""
    raw = "|".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    "params",
    [
        {"lr": 1e-4, "batch": 32},
        {"name": "alpha", "n": 3, "flag": True},
        {"b": 2, "a": 1},
        {},
    ],
)
def test_cli_deterministic_run_id_matches_pre_lift_formula(params) -> None:
    """ac-005 — CLI output equals the pre-lift formula for representative dicts."""
    assert deterministic_run_id(params) == _pre_lift(params)


def test_cli_deterministic_run_id_delegates_to_derive_run_id() -> None:
    """ac-005 — CLI helper delegates to the lifted workspace helper."""
    from molexp.workspace.utils import derive_run_id

    params = {"lr": 5e-4, "batch": 64}
    assert deterministic_run_id(params) == derive_run_id(params)
