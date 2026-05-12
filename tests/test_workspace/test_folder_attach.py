"""Tests for ``Folder.attach() / create_child() / get_child()`` on entity parents.

Covers sub-spec ``unify-folder-abstraction-02-workspace-subclassing``:

- ac-001 — ``Workspace`` inherits ``Folder`` and exposes ``WORKSPACE_ROOT_KIND``
- ac-002 — ``Project`` / ``Experiment`` / ``Run`` inherit ``Folder`` with correct
  ``kind`` and parent pointers
- ac-003 — generic ``attach`` / ``create_child`` / ``get_child`` / ``children(kind=...)``
  API works on entity parents

The matching triplet-preservation criterion (ac-004) is covered by the
existing ``test_workspace.py`` / ``test_project.py`` / ``test_experiment.py``
/ ``test_run.py`` modules — they intentionally are not modified by this
sub-spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import (
    Experiment,
    ExperimentExistsError,
    ExperimentNotFoundError,
    Folder,
    Project,
    ProjectExistsError,
    ProjectNotFoundError,
    Workspace,
)
from molexp.workspace.folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    WORKSPACE_RUN_KIND,
)


# ── ac-001 ─────────────────────────────────────────────────────────────────
def test_workspace_inherits_folder_with_root_kind(tmp_path: Path) -> None:
    """``Workspace`` is a ``Folder`` and reports ``WORKSPACE_ROOT_KIND``.

    Covers ac-001. The workspace is the unique root: ``parent`` is ``None``,
    ``kind`` is ``"workspace.root"``.
    """
    ws = Workspace(tmp_path)

    assert isinstance(ws, Folder)
    assert ws.kind == WORKSPACE_ROOT_KIND
    assert ws.parent is None


# ── ac-002 ─────────────────────────────────────────────────────────────────
def test_entity_chain_inherits_folder_with_correct_kinds_and_parents(
    tmp_path: Path,
) -> None:
    """Project / Experiment / Run inherit ``Folder`` and chain parents correctly.

    Covers ac-002. Each entity reports the canonical ``workspace.*`` kind
    and its parent is the *typed* upstream entity (so ``isinstance(_, Folder)``
    holds AND the entity-specific upstream alias — ``workspace`` / ``project``
    / ``experiment`` — also works).
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")
    exp = proj.Experiment("e")
    run = exp.Run({"x": 1})

    for entity in (proj, exp, run):
        assert isinstance(entity, Folder)

    assert proj.kind == WORKSPACE_PROJECT_KIND
    assert exp.kind == WORKSPACE_EXPERIMENT_KIND
    assert run.kind == WORKSPACE_RUN_KIND

    # Folder.parent walk
    assert proj.parent is ws
    assert exp.parent is proj
    assert run.parent is exp


# ── ac-003 ─────────────────────────────────────────────────────────────────
def test_attach_is_idempotent_for_project(tmp_path: Path) -> None:
    """``ws.attach("p", kind=workspace.project, child_cls=Project)`` is idempotent.

    Covers ac-003 part 1. Repeat call returns the cached instance — same
    semantics as ``ws.Project("p")``.
    """
    ws = Workspace(tmp_path)

    proj_a = ws.attach("p", kind=WORKSPACE_PROJECT_KIND, child_cls=Project)
    proj_b = ws.attach("p", kind=WORKSPACE_PROJECT_KIND, child_cls=Project)

    assert isinstance(proj_a, Project)
    assert proj_a is proj_b


def test_attach_matches_pascal_case_factory_identity(tmp_path: Path) -> None:
    """``ws.attach(...)`` and ``ws.Project(name)`` resolve to the same instance.

    Covers ac-003 + ac-004 cross-check: the typed factory is a thin wrapper
    over ``attach()``, so both return the same cached object.
    """
    ws = Workspace(tmp_path)

    a = ws.Project("p")
    b = ws.attach("p", kind=WORKSPACE_PROJECT_KIND, child_cls=Project)

    assert a is b


def test_create_child_raises_typed_exists_error_on_collision(tmp_path: Path) -> None:
    """``create_child`` raises the entity-typed ``*ExistsError`` on duplicate.

    Covers ac-003 part 2. The exception class is determined by
    ``child_cls``, NOT by a generic ``FolderExistsError``.
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")

    proj.create_child("e", kind=WORKSPACE_EXPERIMENT_KIND, child_cls=Experiment)
    with pytest.raises(ExperimentExistsError):
        proj.create_child("e", kind=WORKSPACE_EXPERIMENT_KIND, child_cls=Experiment)

    # Same behaviour at the Workspace → Project level.
    with pytest.raises(ProjectExistsError):
        ws.create_child("p", kind=WORKSPACE_PROJECT_KIND, child_cls=Project)


def test_get_child_raises_typed_not_found_error_on_miss(tmp_path: Path) -> None:
    """``get_child`` raises the entity-typed ``*NotFoundError`` on miss.

    Covers ac-003 part 3. Like ``create_child``, the exception class
    follows ``child_cls``.
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")

    with pytest.raises(ExperimentNotFoundError):
        proj.get_child("nonexistent", kind=WORKSPACE_EXPERIMENT_KIND, child_cls=Experiment)

    with pytest.raises(ProjectNotFoundError):
        ws.get_child("nonexistent", kind=WORKSPACE_PROJECT_KIND, child_cls=Project)


def test_get_child_kind_none_finds_existing_entity(tmp_path: Path) -> None:
    """``get_child`` with ``kind=None`` resolves regardless of kind.

    Covers ac-003 part 3 (the ``kind=None`` clause). Useful when callers
    just have an id and want it back if it exists at any kind under the
    parent.
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")
    proj.Experiment("e")

    # kind=None forces a broad scan; child_cls still tells loader how to
    # reconstruct the on-disk row.
    looked_up = proj.get_child("e", kind=None, child_cls=Experiment)
    assert isinstance(looked_up, Experiment)


def test_children_filters_by_kind_at_project_level(tmp_path: Path) -> None:
    """``project.children(kind=workspace.experiment)`` returns only experiments.

    Covers ac-003 part 4 + ac-005-adjacent: a Run-kind filter at the Project
    level returns ``[]`` because Runs live one level deeper (under
    ``experiments/<id>/runs/``); the kind-filter must not leak across
    levels.
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")
    exp1 = proj.Experiment("e1")
    proj.Experiment("e2")
    # A run lives under the experiment, NOT under the project.
    exp1.Run({"x": 1})

    exps = proj.children(kind=WORKSPACE_EXPERIMENT_KIND)
    assert len(exps) == 2
    assert all(isinstance(e, Folder) for e in exps)
    assert {e.id for e in exps} == {"e1", "e2"}

    # Workflow.run kind never leaks across the project boundary.
    runs_at_project_level = proj.children(kind=WORKSPACE_RUN_KIND)
    assert runs_at_project_level == []


# ── ac-007 (FolderMetadata derived in-memory) ──────────────────────────────
def test_folder_metadata_derived_in_memory_from_entity_metadata(
    tmp_path: Path,
) -> None:
    """``proj.folder_metadata`` derives from ``proj.metadata`` at construction.

    Covers ac-007. The Folder view exposes id / name / kind / created_at
    via the entity's own metadata model — no separate on-disk file.
    """
    ws = Workspace(tmp_path)
    proj = ws.Project("p")

    fm = proj.folder_metadata
    assert fm.id == proj.metadata.id
    assert fm.name == proj.metadata.name
    assert fm.kind == WORKSPACE_PROJECT_KIND
    assert fm.created_at == proj.metadata.created_at
    # No independent on-disk source, so updated_at == created_at after fresh
    # materialization — see spec § Metadata duality.
    assert fm.updated_at == proj.metadata.created_at
