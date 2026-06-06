"""RunContext decomposition: facade + 4 layered collaborators.

workspace-slim-03: `RunContext` keeps its public API unchanged but pushes
its internals down into four focused collaborators, each in its own
module:

- `run_lifecycle.RunLifecycle`  — enter/exit state machine
- `run_execution.ExecutionStore` — executions/<id>/ + execution_history
- `run_context.ContextStore`     — run.json context blob
- `run_assets.RunAssets`         — scope/manifest/catalog + accessors

These tests pin the structural contract (ac-002/003/004) and the hard
backward-compat gate on the facade's public surface (ac-005).
"""

from __future__ import annotations

import pytest

from molexp.workspace.run import RunContext

# Private methods that must relocate off RunContext onto a collaborator.
_RELOCATED_METHODS = [
    "_claim_ownership",
    "_next_execution_id",
    "_save_context",
    "_save_error_details",
    "_write_execution_metadata",
    "_close_execution_record",
    "_apply_profile_metadata",
]

# The frozen public surface of the facade — must survive the refactor.
_PUBLIC_SURFACE = [
    "artifact",
    "log",
    "checkpoint",
    "metrics",
    "context",
    "params",
    "config",
    "folder",
    "set_result",
    "get_result",
    "set_active_task",
    "bind_workflow_version",
    "set_workflow",
    "register_asset",
    "get_asset",
    "find_asset",
    "get_data_dir",
    "mark_failed",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "open",
]


def _collaborator_types():
    from molexp.workspace.run_assets import RunAssets
    from molexp.workspace.run_context import ContextStore
    from molexp.workspace.run_execution import ExecutionStore
    from molexp.workspace.run_lifecycle import RunLifecycle

    return RunLifecycle, ExecutionStore, ContextStore, RunAssets


@pytest.mark.unit
def test_four_collaborators_each_live_in_own_module():
    """ac-002: each collaborator is a class in its dedicated module."""
    for cls in _collaborator_types():
        assert isinstance(cls, type), cls


@pytest.mark.unit
def test_runcontext_constructs_the_four_collaborators(run):
    """ac-003: RunContext holds exactly one instance of each collaborator."""
    ctx = run.start()
    held = list(vars(ctx).values())
    for cls in _collaborator_types():
        count = sum(isinstance(v, cls) for v in held)
        assert count == 1, f"expected exactly one {cls.__name__}, found {count}"


@pytest.mark.unit
@pytest.mark.parametrize("method", _RELOCATED_METHODS)
def test_relocated_private_methods_gone_from_runcontext(method: str):
    """ac-004: former private methods no longer live on RunContext."""
    assert not hasattr(RunContext, method), f"RunContext still owns {method}"


@pytest.mark.unit
@pytest.mark.parametrize("name", _PUBLIC_SURFACE)
def test_public_surface_preserved(run, name: str):
    """ac-005: every public attribute/method still resolves on the facade."""
    ctx = run.start()
    assert hasattr(ctx, name), f"RunContext public surface lost {name}"
