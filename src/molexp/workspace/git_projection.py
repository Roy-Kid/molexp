"""workspace → git projection — entities mapped onto real git objects.

A **derived, rebuildable, single-direction** view of the authoritative
workspace (spec: workspace-git-projection-03-map). The authoritative truth is
always the on-disk ``*.json`` entities + per-scope ``assets.json`` manifests;
git is a projection *target*, never a second truth. This module maps
``Folder`` / ``Asset`` / ``RunMetadata`` / ``ExecutionRecord`` onto the
content-agnostic git-object primitives from :mod:`molexp.git.objects` and
writes refs under ``refs/molexp/*``.

The load-bearing guarantee is **deterministic rebuild**: :meth:`GitProjection.rebuild`
erases the object database + ``refs/molexp/*`` and re-derives everything from
the authoritative files, producing byte-identical OIDs. This is only possible
because every commit's author/committer date is taken from the **recorded**
``ExecutionRecord`` timestamps — never ``now()``.

What is projected, per run:

* ``run.json`` (entity identity: params / config / config_hash / source
  snapshot / script) and ``assets.json`` (the manifest) as blobs;
* the ``source/`` snapshot directory as a subtree;
* settled ``artifacts/`` as a subtree — small files inline as blobs, files
  over ``blob_threshold_bytes`` as a small **pointer** blob (the bytes stay in
  the molexp CAS / remote FS, never bloating the object DB).

What is **excluded**: ``_ops/`` (hot machine state), ``cache/``, the
per-attempt ``executions/`` churn, and every derived children-index ``*.json``
(those live at container level and are rebuildable, not truth).

Each ``Execution`` becomes a commit over the run tree, chained in recorded
order (``refs/molexp/runs/<run_id>`` points at the tip), so
``git log refs/molexp/runs/<id>`` walks the run's execution history and
``git diff <treeA> <treeB>`` surfaces param/script deltas between two runs.

``molexp.ids`` is untouched: ``compute_content_hash`` / ``config_hash`` stay
the orthogonal molexp-native identity and never receive a git OID. The git OID
is a *structural projection*, computed separately, never written back into any
authoritative ``*.json``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.git import (
    ObjectDb,
    Oid,
    Signature,
    TreeEntry,
    build_commit,
    build_tree,
    ensure_object_db,
    set_ref,
    write_blob,
)
from molexp.ids import compute_content_hash

if TYPE_CHECKING:
    from molexp.workspace.run import Run
    from molexp.workspace.workspace import Workspace

__all__ = [
    "ARTIFACT_POINTER_MARKER",
    "DEFAULT_BLOB_THRESHOLD_BYTES",
    "GitProjection",
    "ProjectedRun",
    "ProjectionResult",
]

# Files larger than this are projected as a pointer, not inline bytes.
DEFAULT_BLOB_THRESHOLD_BYTES = 10 * 1024 * 1024

# First line of an artifact pointer blob (bytes kept in CAS / remote FS).
ARTIFACT_POINTER_MARKER = "molexp-artifact-pointer"

# Deterministic projection identity for commits (never the operator's git config).
_PROJECTION_NAME = "molexp"
_PROJECTION_EMAIL = "projection@molexp"

# git tree modes.
_MODE_FILE = "100644"
_MODE_DIR = "40000"

# Run-dir entries projected as top-level blobs (authoritative identity + manifest).
_RUN_BLOB_FILES = ("run.json", "assets.json")


@dataclass(frozen=True)
class ProjectedRun:
    """The git objects a single run projects to."""

    run_id: str
    tree: Oid
    commits: tuple[Oid, ...]
    ref: str | None  # refs/molexp/runs/<run_id>, or None when the run never ran


@dataclass(frozen=True)
class ProjectionResult:
    """The outcome of projecting a whole workspace."""

    workspace_tree: Oid
    runs: tuple[ProjectedRun, ...]

    def run(self, run_id: str) -> ProjectedRun | None:
        """Return the projection for ``run_id``, or ``None`` if absent."""
        return next((r for r in self.runs if r.run_id == run_id), None)


def _git_date(dt: datetime) -> str:
    """Render ``dt`` as a deterministic git ``<epoch> +0000`` date string.

    A naive timestamp is interpreted as UTC so the projected OID is stable and
    host-tz-independent. Aware timestamps are converted to their epoch.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return f"{int(dt.timestamp())} +0000"


class GitProjection:
    """Project a :class:`~molexp.workspace.workspace.Workspace` onto git objects.

    The projection is single-direction (molexp authoritative → git derived) and
    reads exclusively through the authoritative files, so :meth:`rebuild` is
    deterministic.
    """

    def __init__(
        self,
        workspace: Workspace,
        db: ObjectDb,
        *,
        blob_threshold_bytes: int = DEFAULT_BLOB_THRESHOLD_BYTES,
    ) -> None:
        self._ws = workspace
        self._db = db
        self._threshold = blob_threshold_bytes

    async def project(self) -> ProjectionResult:
        """Derive the full git projection from the current authoritative state."""
        projected: list[ProjectedRun] = []
        project_entries: list[TreeEntry] = []

        for project in self._ws.list_projects():
            experiment_entries: list[TreeEntry] = []
            for experiment in project.list_experiments():
                run_entries: list[TreeEntry] = []
                for run in experiment.list_runs():
                    pr = await self._project_run(run)
                    projected.append(pr)
                    run_entries.append(TreeEntry(_MODE_DIR, f"run-{run.id}", pr.tree))
                exp_tree = await build_tree(self._db, run_entries)
                experiment_entries.append(TreeEntry(_MODE_DIR, experiment.id, exp_tree))
            proj_tree = await build_tree(self._db, experiment_entries)
            project_entries.append(TreeEntry(_MODE_DIR, project.id, proj_tree))

        workspace_tree = await build_tree(self._db, project_entries)
        return ProjectionResult(workspace_tree=workspace_tree, runs=tuple(projected))

    async def rebuild(self) -> ProjectionResult:
        """Erase the object DB + ``refs/molexp/*`` and re-derive (deterministic).

        Reproduces byte-identical OIDs from the authoritative files — the proof
        that the projection is a derived view, not a second source of truth.
        """
        shutil.rmtree(self._db.path, ignore_errors=True)
        self._db = await ensure_object_db(self._db.path)
        return await self.project()

    async def _project_run(self, run: Run) -> ProjectedRun:
        tree = await self._project_run_tree(run)
        commits = await self._project_run_commits(run, tree)
        ref: str | None = None
        if commits:
            ref = f"refs/molexp/runs/{run.id}"
            await set_ref(self._db, ref, commits[-1])
        return ProjectedRun(run_id=run.id, tree=tree, commits=commits, ref=ref)

    async def _project_run_tree(self, run: Run) -> Oid:
        run_dir = Path(str(run.run_dir))
        entries: list[TreeEntry] = []
        for name in _RUN_BLOB_FILES:
            f = run_dir / name
            if f.is_file():
                oid = await write_blob(self._db, f.read_bytes())
                entries.append(TreeEntry(_MODE_FILE, name, oid))
        source_tree = await self._project_plain_subtree(run_dir / "source")
        if source_tree is not None:
            entries.append(TreeEntry(_MODE_DIR, "source", source_tree))
        artifacts_tree = await self._project_artifacts(run_dir / "artifacts")
        if artifacts_tree is not None:
            entries.append(TreeEntry(_MODE_DIR, "artifacts", artifacts_tree))
        return await build_tree(self._db, entries)

    async def _project_plain_subtree(self, directory: Path) -> Oid | None:
        """Project ``directory`` recursively as blobs (used for ``source/``)."""
        if not directory.is_dir():
            return None
        entries: list[TreeEntry] = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                sub = await self._project_plain_subtree(child)
                if sub is not None:
                    entries.append(TreeEntry(_MODE_DIR, child.name, sub))
            elif child.is_file():
                oid = await write_blob(self._db, child.read_bytes())
                entries.append(TreeEntry(_MODE_FILE, child.name, oid))
        return await build_tree(self._db, entries)

    async def _project_artifacts(self, directory: Path) -> Oid | None:
        """Project ``artifacts/`` — small files inline, large files as pointers."""
        if not directory.is_dir():
            return None
        entries: list[TreeEntry] = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            if not child.is_file():
                continue
            size = child.stat().st_size
            if size > self._threshold:
                oid = await write_blob(self._db, _artifact_pointer(child, size))
            else:
                oid = await write_blob(self._db, child.read_bytes())
            entries.append(TreeEntry(_MODE_FILE, child.name, oid))
        return await build_tree(self._db, entries)

    async def _project_run_commits(self, run: Run, tree: Oid) -> tuple[Oid, ...]:
        """One commit per recorded ``ExecutionRecord``, chained in order.

        Author date = ``started_at``; committer date = ``finished_at`` (or
        ``started_at`` while still running). Both come from the recorded
        timestamps so the projection rebuilds byte-identically.
        """
        commits: list[Oid] = []
        parent: Oid | None = None
        for record in run.read_ops().executions:
            author = Signature(
                name=_PROJECTION_NAME,
                email=_PROJECTION_EMAIL,
                date=_git_date(record.started_at),
            )
            committer = Signature(
                name=_PROJECTION_NAME,
                email=_PROJECTION_EMAIL,
                date=_git_date(record.finished_at or record.started_at),
            )
            oid = await build_commit(
                self._db,
                tree=tree,
                parents=[parent] if parent is not None else [],
                message=_execution_message(record),
                author=author,
                committer=committer,
            )
            commits.append(oid)
            parent = oid
        return tuple(commits)


def _artifact_pointer(path: Path, size: int) -> bytes:
    """A small pointer blob for an over-threshold artifact (bytes stay in CAS)."""
    content_hash = compute_content_hash(path)
    return (
        f"{ARTIFACT_POINTER_MARKER}\n"
        f"name: {path.name}\n"
        f"size: {size}\n"
        f"content_hash: {content_hash}\n"
    ).encode()


def _execution_message(record) -> str:  # noqa: ANN001 — ExecutionRecord (workspace-internal)
    """Deterministic commit message summarising one execution attempt."""
    finished = record.finished_at.isoformat() if record.finished_at else "-"
    return (
        f"{record.status} {record.execution_id}\n\n"
        f"started_at: {record.started_at.isoformat()}\n"
        f"finished_at: {finished}\n"
    )
