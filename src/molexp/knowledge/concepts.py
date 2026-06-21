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
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, cast

from .folder import Folder
from .ops import (
    RUN_OPS_NAME,
    TERMINAL_STATUSES,
    ExecutionRecord,
    RunOpsState,
    RunStatus,
    _utcnow,
)
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

    # ── operational state (_ops/run.json — hot, never in meta.yaml) ───────

    def read_ops(self) -> RunOpsState:
        """Load the typed Run ops state, or the default if not yet written."""
        return RunOpsState.model_validate(self.read_ops_json(RUN_OPS_NAME) or {})

    def write_ops(self, state: RunOpsState) -> None:
        """Persist the Run ops state to ``_ops/run.json`` (atomic)."""
        self.write_ops_json(RUN_OPS_NAME, state.model_dump(mode="json"))

    def update_ops(self, fn: Callable[[RunOpsState], RunOpsState]) -> RunOpsState:
        """Read-modify-write the typed Run ops state under an advisory lock."""

        def apply(raw: dict) -> dict:
            return fn(RunOpsState.model_validate(raw or {})).model_dump(mode="json")

        return RunOpsState.model_validate(self.update_ops_json(RUN_OPS_NAME, apply))

    def set_status(self, status: RunStatus, *, now: datetime | None = None) -> RunOpsState:
        """Transition the run status; stamp ``finished_at`` / ``started_at``.

        Terminal statuses set ``finished_at``; a non-terminal status clears it.
        Entering ``running`` stamps ``started_at`` if not already set. This
        records state — it does not police transition legality (okf-05).
        """
        ts = now or _utcnow()

        def transition(state: RunOpsState) -> RunOpsState:
            update: dict[str, object] = {"status": status}
            update["finished_at"] = ts if status in TERMINAL_STATUSES else None
            if status == RunStatus.RUNNING and state.started_at is None:
                update["started_at"] = ts
            return state.model_copy(update=update)

        return self.update_ops(transition)

    def beat(self, *, now: datetime | None = None) -> RunOpsState:
        """Refresh the ownership heartbeat timestamp (aware-UTC)."""
        ts = now or _utcnow()
        return self.update_ops(lambda s: s.model_copy(update={"heartbeat_at": ts}))

    def record_execution(self, record: ExecutionRecord) -> RunOpsState:
        """Append an execution attempt and make it the current execution."""
        return self.update_ops(
            lambda s: s.model_copy(
                update={
                    "executions": (*s.executions, record),
                    "current_execution_id": record.execution_id,
                }
            )
        )

    def is_retryable(self) -> bool:
        """Whether ``resume`` / ``rerun`` apply to this run's current status."""
        return self.read_ops().is_retryable


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
