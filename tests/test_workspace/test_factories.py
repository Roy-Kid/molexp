"""Typed semantic-sugar CRUD + workspace error hierarchy.

Covers the post-rectification ``unify-folder-abstraction-03`` spec
acceptance — workspace entities all subclass ``Folder`` and expose a
snake_case verb-noun CRUD (``add_* / get_* / has_* / list_*s / remove_*``)
that is one-line sugar over the generic ``Folder.add_folder / get_folder
/ has_folder / list_folders / remove_folder``. All ``add_*`` verbs are
**idempotent** on slugified name (return the cached / on-disk instance
on collision); strict ``get_*`` raises typed ``*NotFoundError``.

Idempotency across all three levels (in-memory + on-disk) is locked by
``test_crud_convergence.py``; this module keeps the exception-hierarchy
contract and the strict-getter refusal.
"""

from __future__ import annotations

import pytest

from molexp.workspace import (
    ExperimentExistsError,
    ExperimentNotFoundError,
    ProjectExistsError,
    ProjectNotFoundError,
    RunExistsError,
    RunNotFoundError,
    Workspace,
)

# ── Exception module shape ─────────────────────────────────────────────────


def test_not_found_errors_are_lookup_errors():
    assert issubclass(ProjectNotFoundError, LookupError)
    assert issubclass(ExperimentNotFoundError, LookupError)
    assert issubclass(RunNotFoundError, LookupError)


def test_exists_errors_are_value_errors():
    assert issubclass(ProjectExistsError, ValueError)
    assert issubclass(ExperimentExistsError, ValueError)
    assert issubclass(RunExistsError, ValueError)


# ── strict getter refuses with the typed error ─────────────────────────────


def test_workspace_get_project_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    with pytest.raises(ProjectNotFoundError) as exc:
        ws.get_project("never-created")
    assert "never-created" in str(exc.value)
