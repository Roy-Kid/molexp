"""Filesystem implementation of :class:`ArtifactStore`.

Layout under ``root``::

    <kind>/<id>.json        # put_json content
    <kind>/<id>.txt         # put_text content
    <kind>/<id>-<original>  # put_file content (original filename preserved)
    _refs/<id>.json         # full ArtifactRef as JSON
    _index/<kind>.json      # ordered list of ids per kind

Content writes go through :func:`molexp.workspace.atomic_write_json` /
:func:`molexp.workspace.atomic_write_text`, so a crash mid-write leaves the
original file (if any) intact. Content hash comes from
:func:`molexp.workspace.utils.compute_content_hash` with the ``sha256:``
prefix stripped before populating :attr:`ArtifactRef.sha256` (which stores
bare hex per harness-goal.md §4.1).

Idempotency: ``put_*`` is keyed on ``(kind, content_hash)``. The
artifact id is the first 16 hex characters of
``sha256(f"{kind}:{content_sha}")``, so two ``put_*`` calls with
identical content under the same kind return the same
:class:`ArtifactRef`, while identical content put under two *different*
kinds yields two distinct ids — preventing the previous overwrite where
``put_text(kind=A, …)`` and ``put_text(kind=B, …)`` of the same bytes
clobbered each other's :class:`ArtifactRef`.

Provenance merging: an idempotent hit (same kind + content) returns a
ref with the *union* of historical ``parent_ids`` and the
newly-supplied ones, so lineage doesn't lose edges when the same
artifact is re-derived via a new path. The ref's existing
``created_at`` / ``created_by`` are preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from molexp.harness.errors import ArtifactNotFoundError
from molexp.harness.schemas import ArtifactKind, ArtifactRef
from molexp.workspace import atomic_write_json, atomic_write_text
from molexp.workspace.utils import compute_content_hash

__all__ = ["FileArtifactStore"]


_ID_LEN = 16  # 64 bits of sha256 — collision-free at harness scale.


def _hash_path(path: Path) -> str:
    """Return bare-hex sha256 of an on-disk file/dir via the workspace helper."""
    return compute_content_hash(path).removeprefix("sha256:")


def _derive_id(kind: ArtifactKind, sha: str) -> str:
    """Derive an artifact id from ``(kind, content_sha)``.

    Including ``kind`` in the digest prevents two distinct artifacts —
    same bytes, different kind — from sharing the same id and
    overwriting each other's ref in the ``_refs/`` directory.
    """
    return hashlib.sha256(f"{kind}:{sha}".encode()).hexdigest()[:_ID_LEN]


class FileArtifactStore:
    """Content-addressed filesystem artifact store."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._refs_dir = self._root / "_refs"
        self._index_dir = self._root / "_index"

    # ------------------------------------------------------------------ put

    def put_json(
        self,
        kind: ArtifactKind,
        obj: object,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef:
        return self._put_via_staging(
            kind=kind,
            suffix=".json",
            write_staged=lambda staged: atomic_write_json(staged, obj),
            created_by=created_by,
            parent_ids=parent_ids,
        )

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef:
        return self._put_via_staging(
            kind=kind,
            suffix=".txt",
            write_staged=lambda staged: atomic_write_text(staged, text),
            created_by=created_by,
            parent_ids=parent_ids,
        )

    def put_file(
        self,
        kind: ArtifactKind,
        path: Path,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef:
        # Hash the source directly — no need to copy first.
        sha = _hash_path(path)
        existing = self._find_existing(kind, sha)
        if existing is not None:
            return self._merge_parent_ids(existing, parent_ids)

        artifact_id = _derive_id(kind, sha)
        dest_dir = self._root / kind
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{artifact_id}-{path.name}"
        shutil.copy2(path, dest)
        return self._finalize(
            artifact_id=artifact_id,
            kind=kind,
            content_path=dest,
            sha=sha,
            created_by=created_by,
            parent_ids=parent_ids,
        )

    def _put_via_staging(
        self,
        *,
        kind: ArtifactKind,
        suffix: str,
        write_staged: Callable[[Path], None],
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef:
        """Write to a temp path, hash from disk, then move into place.

        Hashing post-write (via :func:`compute_content_hash`) keeps
        :attr:`ArtifactRef.sha256` byte-identical to the on-disk file, which
        ``atomic_write_json``'s ``indent=2`` formatting would otherwise
        diverge from an in-memory canonical hash.
        """
        kind_dir = self._root / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        # Stage in the destination directory so the final replace is atomic
        # (same filesystem).
        fd, staged_str = tempfile.mkstemp(prefix=".staging_", suffix=suffix, dir=kind_dir)
        os.close(fd)
        staged = Path(staged_str)
        staged.unlink()  # let write_staged create the file fresh
        try:
            write_staged(staged)
            sha = _hash_path(staged)
            existing = self._find_existing(kind, sha)
            if existing is not None:
                staged.unlink(missing_ok=True)
                return self._merge_parent_ids(existing, parent_ids)

            artifact_id = _derive_id(kind, sha)
            final = kind_dir / f"{artifact_id}{suffix}"
            staged.replace(final)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

        return self._finalize(
            artifact_id=artifact_id,
            kind=kind,
            content_path=final,
            sha=sha,
            created_by=created_by,
            parent_ids=parent_ids,
        )

    # ------------------------------------------------------------------ get

    def get(self, artifact_id: str) -> bytes:
        ref = self.get_ref(artifact_id)
        content_path = Path(ref.uri.removeprefix("file://"))
        if not content_path.exists():
            raise ArtifactNotFoundError(
                f"artifact {artifact_id!r} ref exists but content missing at {content_path}"
            )
        return content_path.read_bytes()

    def get_ref(self, artifact_id: str) -> ArtifactRef:
        ref_path = self._refs_dir / f"{artifact_id}.json"
        if not ref_path.exists():
            raise ArtifactNotFoundError(f"artifact {artifact_id!r} not found")
        return ArtifactRef.model_validate_json(ref_path.read_text(encoding="utf-8"))

    def list_by_kind(self, kind: ArtifactKind) -> list[ArtifactRef]:
        index = self._read_index(kind)
        return [self.get_ref(aid) for aid in index]

    def latest_by_kind(self, kind: ArtifactKind) -> ArtifactRef | None:
        index = self._read_index(kind)
        if not index:
            return None
        return self.get_ref(index[-1])

    # ----------------------------------------------------------- internals

    def _find_existing(self, kind: ArtifactKind, sha: str) -> ArtifactRef | None:
        artifact_id = _derive_id(kind, sha)
        ref_path = self._refs_dir / f"{artifact_id}.json"
        if not ref_path.exists():
            return None
        ref = ArtifactRef.model_validate_json(ref_path.read_text(encoding="utf-8"))
        if ref.kind != kind or ref.sha256 != sha:
            # Hash collision at 64 bits is essentially impossible; if it
            # ever happens, fall through and treat as a new artifact.
            return None
        return ref

    def _merge_parent_ids(self, existing: ArtifactRef, new_parent_ids: list[str]) -> ArtifactRef:
        """On idempotent hit, union the new parent_ids with the stored ones.

        Without this, ``put_X`` returning an existing ref on a second
        derivation path would drop the lineage edge for that second
        derivation — provenance would be incomplete. We preserve the
        original ``created_at`` / ``created_by`` so the audit trail
        still points to the first producer.
        """
        merged: list[str] = list(existing.parent_ids)
        added = False
        for pid in new_parent_ids:
            if pid not in merged:
                merged.append(pid)
                added = True
        if not added:
            return existing
        updated = existing.model_copy(update={"parent_ids": merged})
        atomic_write_json(
            self._refs_dir / f"{existing.id}.json",
            json.loads(updated.model_dump_json()),
        )
        return updated

    def _finalize(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        content_path: Path,
        sha: str,
        created_by: str,
        parent_ids: list[str],
    ) -> ArtifactRef:
        ref = ArtifactRef(
            id=artifact_id,
            kind=kind,
            uri=f"file://{content_path.resolve()}",
            sha256=sha,
            created_at=datetime.now(tz=UTC),
            created_by=created_by,
            parent_ids=list(parent_ids),
            metadata={},
        )
        atomic_write_json(self._refs_dir / f"{artifact_id}.json", json.loads(ref.model_dump_json()))
        self._append_to_index(kind, artifact_id)
        return ref

    def _read_index(self, kind: ArtifactKind) -> list[str]:
        index_path = self._index_dir / f"{kind}.json"
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _append_to_index(self, kind: ArtifactKind, artifact_id: str) -> None:
        index = self._read_index(kind)
        if artifact_id in index:
            return
        index.append(artifact_id)
        atomic_write_json(self._index_dir / f"{kind}.json", index)
