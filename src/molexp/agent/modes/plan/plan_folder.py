"""On-disk layout for a single PlanMode read-only plan.

:class:`PlanFolder` is the agent-layer :class:`molexp.workspace.Folder`
subclass owned by PlanMode (``kind = "agent.plan"``). It mounts on any
workspace ``Folder`` via the generic ``add_folder`` API; the plan id
becomes the folder name. The agent layer owns the layout entirely —
workspace stays unaware of the ``agent.plan`` kind.

PlanMode is a *read-only typed planner*: the folder persists only typed
plan artefacts (the :class:`IntentSpec`, the :class:`CapabilityGraph`,
the candidate / selected :class:`PlanGraph`\\ s, the preflight report).
It writes **no** ``src/`` / ``tests/`` / ``ir/`` directory and no
executable code — codegen is AuthorMode's job.

Mount points::

    ws = Workspace("./lab")
    pf = ws.add_folder(PlanFolder(name="my-plan"))  # workspace-level
    exp = ws.add_project("proj").add_experiment("exp")
    pf = exp.add_folder(PlanFolder(name="my-plan"))  # or under an experiment

Subtree layout::

    <parent>/plans/<plan_id>/
    ├── plan_folder.json    # PlanFolder own metadata + lifecycle state
    ├── intent.json         # the typed IntentSpec
    ├── capability_graph.json
    ├── candidates/<label>.json   # one candidate PlanGraph per label
    ├── selected_plan.json
    └── preflight_report.json

Construction is side-effect free; ``parent.add_folder(plan)`` /
:meth:`Folder.path` create directories lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from molexp._typing import JSONValue
from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
    PlanState,
    assert_legal_transition,
)
from molexp.agent.modes.plan.plan_graph_preflight import PlanGraphPreflightReport
from molexp.path import Path as MolexpPath
from molexp.workspace import Folder, FolderMetadata, atomic_write_text
from molexp.workspace.base import _load_metadata, _reconstruct, _save_metadata

if TYPE_CHECKING:
    from molexp.workspace.fs import PathArg

__all__ = ["AGENT_PLAN_KIND", "PlanFolder", "PlanFolderMetadata"]

AGENT_PLAN_KIND = "agent.plan"
"""Folder ``kind`` for a single PlanMode plan workspace."""

PLAN_METADATA_FILENAME = "plan_folder.json"
"""Per-:class:`PlanFolder` metadata file (auto-derived from class name)."""


class PlanFolderMetadata(FolderMetadata, frozen=True):
    """Frozen lifecycle metadata for a :class:`PlanFolder`.

    Extends :class:`FolderMetadata` (so the workspace layer treats it as
    opaque ``Folder`` metadata) with the PlanMode lifecycle field.

    Attributes:
        plan_state: The plan's current
            :class:`~molexp.agent.modes._planning.PlanState`. Defaults
            to :data:`PlanState.intake`.
    """

    plan_state: PlanState = PlanState.intake


def _new_plan_id() -> str:
    """Generate a fresh human-readable plan id."""
    from molexp.workflow import generate_name

    return generate_name()


class PlanFolder(Folder):
    """One PlanMode plan workspace — ``kind = "agent.plan"``.

    Construct with an explicit ``name`` (the plan id, slug-safe) or leave
    it to default to a fresh generated name. Mount via the generic
    :meth:`Folder.add_folder` on any workspace folder::

        plan = ws.add_folder(PlanFolder(name="my-plan"))
        plan.write_intent(intent_spec)
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str | None = None,
        kind: str = AGENT_PLAN_KIND,
        _entity_metadata: PlanFolderMetadata | None = None,
    ) -> None:
        plan_id = name if name is not None else _new_plan_id()
        super().__init__(parent=parent, name=plan_id, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else PlanFolderMetadata(id=self._name, name=plan_id, kind=kind)
        )
        self._entity_metadata: PlanFolderMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> MolexpPath:
        if self._parent is None:
            raise RuntimeError(
                f"PlanFolder {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self).child_dir(self._parent, self._name)

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> MolexpPath:
        """Plans live under ``<parent>/plans/<plan_id>/``."""
        return MolexpPath(parent._fs.join(parent.path(), "plans", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> PlanFolder:
        meta_path = parent._fs.join(child_dir, PLAN_METADATA_FILENAME)
        meta = _load_metadata(PlanFolderMetadata, meta_path, fs=parent._fs)
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=AGENT_PLAN_KIND,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {"_entity_metadata": meta}
        return _reconstruct(cls, attrs)

    def materialize(self) -> None:
        self._fs.mkdir(self.path(), parents=True, exist_ok=True)
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), PLAN_METADATA_FILENAME),
            fs=self._fs,
        )

    def save(self) -> None:
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), PLAN_METADATA_FILENAME),
            fs=self._fs,
        )

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    @property
    def metadata(self) -> PlanFolderMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def plan_id(self) -> str:
        """Alias for :attr:`Folder.name`."""
        return self._name

    # ── Lifecycle ────────────────────────────────────────────────────────

    @property
    def plan_state(self) -> PlanState:
        """The plan's current :class:`PlanState`."""
        return self._entity_metadata.plan_state

    def transition_to(self, dst: PlanState) -> None:
        """Move the plan to ``dst``, enforcing the legal-transition table.

        Raises:
            IllegalPlanTransitionError: if the move is not in
                ``LEGAL_TRANSITIONS``.
        """
        assert_legal_transition(self._entity_metadata.plan_state, dst)
        self._entity_metadata = self._entity_metadata.model_copy(update={"plan_state": dst})

    # ── Directory helper ─────────────────────────────────────────────────

    def _root(self) -> Path:
        """Resolve + mkdir the plan root, returning a local :class:`Path`."""
        path = Path(self.path())
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Typed-plan artefact writers ──────────────────────────────────────

    def write_intent(self, intent: IntentSpec) -> Path:
        """Persist the typed :class:`IntentSpec` to ``intent.json``."""
        path = self._root() / "intent.json"
        atomic_write_text(path, intent.model_dump_json(indent=2))
        return path

    def write_capability_graph(self, graph: CapabilityGraph) -> Path:
        """Persist the typed :class:`CapabilityGraph` to ``capability_graph.json``."""
        path = self._root() / "capability_graph.json"
        atomic_write_text(path, graph.model_dump_json(indent=2))
        return path

    def write_candidate(self, label: str, plan_graph: PlanGraph) -> Path:
        """Persist one candidate :class:`PlanGraph` to ``candidates/<label>.json``."""
        candidates_dir = self._root() / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        path = candidates_dir / f"{label}.json"
        atomic_write_text(path, plan_graph.model_dump_json(indent=2))
        return path

    def write_selected_plan(self, plan_graph: PlanGraph) -> Path:
        """Persist the selected :class:`PlanGraph` to ``selected_plan.json``."""
        path = self._root() / "selected_plan.json"
        atomic_write_text(path, plan_graph.model_dump_json(indent=2))
        return path

    def write_preflight_report(self, report: PlanGraphPreflightReport) -> Path:
        """Persist the :class:`PlanGraphPreflightReport` to ``preflight_report.json``."""
        path = self._root() / "preflight_report.json"
        atomic_write_text(path, report.model_dump_json(indent=2))
        return path
