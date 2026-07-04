"""Regression guard for the ``deterministic_run_id`` lift (ac-005).

The CLI's ``deterministic_run_id`` is re-pointed to delegate to
``molexp.workspace.utils.derive_run_id``; this pins the delegation so the
CLI and Python paths share one id formula (the formula itself is owned by
``tests/test_workspace/test_derive_run_id.py``).
"""

from __future__ import annotations

from molexp.cli._common import deterministic_run_id


def test_cli_deterministic_run_id_delegates_to_derive_run_id() -> None:
    """ac-005 — CLI helper delegates to the lifted workspace helper."""
    from molexp.workspace.utils import derive_run_id

    params = {"lr": 5e-4, "batch": 64}
    assert deterministic_run_id(params) == derive_run_id(params)
