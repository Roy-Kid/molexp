"""execution-id derivation (workflow-workspace-hardening P1-5 / ac-010).

``exec-{run_id}`` is the first attempt; retries become ``exec-{run_id}-2``,
``-3``, … The id must be derived from ``max(suffix) + 1`` (never ``len``) so a
deleted *middle* attempt does not collide with a still-present higher id, and
attempt matching must be by *exact* name (base or ``base-<int>``) so a run_id
that is a prefix of another run's id is not miscounted. The three historical
copies of this logic (workspace ExecutionStore, workflow runtime, harness mode)
collapse onto this single helper.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace.utils import derive_execution_id


def _mk(root: Path, *names: str) -> Path:
    """Create execution-attempt dirs under ``<root>/executions``."""
    exec_root = root / "executions"
    exec_root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (exec_root / n).mkdir()
    return exec_root


def test_first_attempt_is_bare_base(tmp_path):
    assert derive_execution_id("r1", tmp_path / "executions") == "exec-r1"


def test_sequential_attempts_increment(tmp_path):
    _mk(tmp_path, "exec-r1")
    assert derive_execution_id("r1", tmp_path / "executions") == "exec-r1-2"
    _mk(tmp_path, "exec-r1-2")
    assert derive_execution_id("r1", tmp_path / "executions") == "exec-r1-3"


def test_middle_delete_does_not_reuse_existing_id(tmp_path):
    """Deleting a *middle* attempt must not regenerate a still-present id.

    With the old ``len(existing) + 1`` scheme {exec-r1, exec-r1-3} has len 2,
    so it would emit ``exec-r1-3`` — colliding with the live attempt. ``max+1``
    derives ``exec-r1-4`` instead.
    """
    exec_root = _mk(tmp_path, "exec-r1", "exec-r1-2", "exec-r1-3")
    (exec_root / "exec-r1-2").rmdir()  # remove the middle attempt
    assert derive_execution_id("r1", tmp_path / "executions") == "exec-r1-4"


def test_exact_prefix_does_not_count_other_runs(tmp_path):
    """A run whose id is a prefix of another's must not absorb its attempts.

    ``exec-abcdef-2`` belongs to run ``abcdef``; deriving for run ``ab`` must
    ignore it (old ``startswith('exec-ab')`` would miscount it).
    """
    _mk(tmp_path, "exec-ab", "exec-abcdef-2", "exec-abc")
    assert derive_execution_id("ab", tmp_path / "executions") == "exec-ab-2"


def test_non_numeric_suffix_ignored(tmp_path):
    _mk(tmp_path, "exec-r1", "exec-r1-notanumber")
    assert derive_execution_id("r1", tmp_path / "executions") == "exec-r1-2"


def test_missing_exec_root_returns_base(tmp_path):
    assert derive_execution_id("r1", tmp_path / "nope" / "executions") == "exec-r1"


def test_callers_delegate_to_helper():
    """The workflow runtime + workspace ExecutionStore route through the
    single helper (no second copy of the derivation)."""
    import inspect

    from molexp.workflow._pydantic_graph import runtime as rt
    from molexp.workspace import run_execution

    assert "derive_execution_id" in inspect.getsource(rt.make_execution_id)
    assert "derive_execution_id" in inspect.getsource(
        run_execution.ExecutionStore.next_execution_id
    )
