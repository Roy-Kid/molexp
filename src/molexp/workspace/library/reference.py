"""Reference — a molexp-native bibliographic record + its JSON-backed store.

A reference is a *record*, not a file: a bib entry (arXiv id / DOI / authors)
has no payload bytes of its own, so it is **not** an :class:`Asset`.  It lives
as a row in the scope's ``library/references.json``.  When a reference does
have an attached PDF, that PDF is imported as a :class:`DataAsset` and linked
back here via :attr:`Reference.pdf_asset_id` — keeping the "Asset == file"
contract intact.

``ReferenceStore`` is the read/write surface over ``references.json``; writes
go through the workspace atomic-JSON helper.
"""

from __future__ import annotations

from datetime import datetime
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
REFERENCES_FILENAME = "references.json"


class Reference(BaseModel):
    """One bibliographic entry in a scope's library.

    ``key`` is a stable, human-meaningful citation key (e.g. ``so3krates2026``)
    used both as the dedup identity and as the token a :class:`NoteAsset`
    cites in its ``refs``.  Identifier fields (``arxiv`` / ``doi`` / ``url``)
    are all optional; at least one of them — or a ``title`` — is expected.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    arxiv: str | None = None
    doi: str | None = None
    url: str | None = None
    tags: tuple[str, ...] = ()
    note: str = ""
    pdf_asset_id: str | None = None
    pdf_path: str | None = None
    """Filesystem path to an attached PDF we *point at* (e.g. a Zotero
    ``storage/`` file) — never copied into the workspace. Distinct from
    ``pdf_asset_id``, which is a PDF imported as a workspace ``DataAsset``."""
    source: str | None = None
    """Where this record came from, e.g. ``"zotero"`` — ``None`` for hand-entered."""
    source_key: str | None = None
    """Stable id in the originating source (e.g. the Zotero item key), so a
    re-sync updates the same record."""
    added_at: datetime = Field(default_factory=datetime.now)

    @property
    def best_url(self) -> str | None:
        """A clickable URL, preferring an explicit ``url`` then DOI then arXiv."""
        if self.url:
            return self.url
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv:
            return f"https://arxiv.org/abs/{self.arxiv}"
        return None


class ReferenceStore:
    """JSON-backed list of :class:`Reference`, keyed by ``key``.

    Layout (``library/references.json``)::

        {"schema_version": 1, "references": [ { ...Reference... }, ... ]}

    ``path`` is coerced to :class:`pathlib.Path` — this is genuine local I/O.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)

    # ── Read ────────────────────────────────────────────────────────────────

    def list(self) -> list[Reference]:
        """Return all references in insertion order."""
        if not self.path.exists():
            return []
        import json

        with open(self.path) as fh:  # noqa: PTH123
            data = json.load(fh)
        return [Reference.model_validate(r) for r in data.get("references", [])]

    def get(self, key: str) -> Reference | None:
        for ref in self.list():
            if ref.key == key:
                return ref
        return None

    # ── Write ─────────────────────────────────────────────────────────────

    def add(self, ref: Reference) -> Reference:
        """Insert or replace a reference (idempotent on ``key``)."""
        refs = [r for r in self.list() if r.key != ref.key]
        refs.append(ref)
        self._save(refs)
        return ref

    def remove(self, key: str) -> bool:
        """Drop the reference with ``key``; return whether it existed."""
        refs = self.list()
        kept = [r for r in refs if r.key != key]
        if len(kept) == len(refs):
            return False
        self._save(kept)
        return True

    # ── Internal ──────────────────────────────────────────────────────────

    def _save(self, refs: list[Reference]) -> None:
        from ..base import _atomic_write_json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "references": [r.model_dump(mode="json") for r in refs],
        }
        _atomic_write_json(self.path, payload)
