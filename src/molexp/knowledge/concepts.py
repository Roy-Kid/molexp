"""The OKF storage hierarchy as typed Concept subclasses.

``Workspace → Project → Experiment → Run`` mirror the
``molexp.workspace.Folder`` family: generic five-verb CRUD (inherited from
:class:`Folder`) plus typed semantic sugar per level (``add_project`` /
``add_experiment`` / ``add_run`` …). Each subclass declares a ``DEFAULT_TYPE``
and registers itself in the concept-type registry (via ``@concept_type``), so
a bundle reconstructs the right class from its ``meta.yaml`` ``type``.

The sugar is **convenience, not enforcement** — structure is conventional, so
any Concept can be mounted anywhere via the generic ``add_folder`` (consistent
with notes/references being "mountable anywhere"). Run is deliberately thin:
its hot operational state (status / heartbeat / executions) lives in the
``_ops/`` sidecar, not in the knowledge layer.
"""

from __future__ import annotations

import os
from typing import ClassVar, cast

from .folder import Folder
from .types import concept_type


@concept_type("run")
class Run(Folder):
    """A run Concept — a reproducible unit; operational state lives in ``_ops/``."""

    DEFAULT_TYPE: ClassVar[str] = "run"

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
        )


@concept_type("experiment")
class Experiment(Folder):
    """An experiment Concept — owns runs."""

    DEFAULT_TYPE: ClassVar[str] = "experiment"

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
        )

    def add_run(self, name: str) -> Run:
        """Create (or return existing) child run; idempotent on slug."""
        return cast(Run, self.add_folder(name, concept_type=Run.DEFAULT_TYPE))

    def get_run(self, name: str) -> Run:
        """Return the child run named *name* (raw or slug)."""
        return cast(Run, self.get_folder(name))

    def has_run(self, name: str) -> bool:
        """Whether a child run named *name* exists."""
        return self.has_folder(name)

    def list_runs(self) -> list[Run]:
        """All child runs (typed view of :meth:`list_folders`)."""
        return [f for f in self.list_folders() if isinstance(f, Run)]

    def remove_run(self, name: str) -> None:
        """Delete a child run (and its subtree)."""
        self.remove_folder(name)


@concept_type("project")
class Project(Folder):
    """A project Concept — owns experiments."""

    DEFAULT_TYPE: ClassVar[str] = "project"

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
        )

    def add_experiment(self, name: str) -> Experiment:
        """Create (or return existing) child experiment; idempotent on slug."""
        return cast(Experiment, self.add_folder(name, concept_type=Experiment.DEFAULT_TYPE))

    def get_experiment(self, name: str) -> Experiment:
        """Return the child experiment named *name* (raw or slug)."""
        return cast(Experiment, self.get_folder(name))

    def has_experiment(self, name: str) -> bool:
        """Whether a child experiment named *name* exists."""
        return self.has_folder(name)

    def list_experiments(self) -> list[Experiment]:
        """All child experiments (typed view of :meth:`list_folders`)."""
        return [f for f in self.list_folders() if isinstance(f, Experiment)]

    def remove_experiment(self, name: str) -> None:
        """Delete a child experiment (and its subtree)."""
        self.remove_folder(name)


@concept_type("workspace")
class Workspace(Folder):
    """A workspace Concept — the top of the storage hierarchy; owns projects."""

    DEFAULT_TYPE: ClassVar[str] = "workspace"

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            parent=parent,
            root=root,
            concept_type=concept_type or self.DEFAULT_TYPE,
        )

    def add_project(self, name: str) -> Project:
        """Create (or return existing) child project; idempotent on slug."""
        return cast(Project, self.add_folder(name, concept_type=Project.DEFAULT_TYPE))

    def get_project(self, name: str) -> Project:
        """Return the child project named *name* (raw or slug)."""
        return cast(Project, self.get_folder(name))

    def has_project(self, name: str) -> bool:
        """Whether a child project named *name* exists."""
        return self.has_folder(name)

    def list_projects(self) -> list[Project]:
        """All child projects (typed view of :meth:`list_folders`)."""
        return [f for f in self.list_folders() if isinstance(f, Project)]

    def remove_project(self, name: str) -> None:
        """Delete a child project (and its subtree)."""
        self.remove_folder(name)


__all__ = ["Experiment", "Project", "Run", "Workspace"]
