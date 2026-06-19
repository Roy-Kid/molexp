"""Read a local Zotero library (``zotero.sqlite``) into molexp ``Reference``s.

The link is a *pointer*, never a copy: bibliographic fields are read from the
Zotero SQLite DB read-only, and any attached PDF is recorded as a filesystem
``pdf_path`` into Zotero's own ``storage/`` tree — molexp imports no bytes.

Zotero locks its DB while running; we open it ``mode=ro`` (and fall back to a
temp copy if that fails) so a running Zotero never blocks the import.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .reference import Reference

# Item types that are not standalone references.
_NON_REFERENCE_TYPES = frozenset({"attachment", "note", "annotation"})

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


class ZoteroImportError(RuntimeError):
    """The given path is not a readable Zotero library."""


def resolve_zotero_db(path: str | Path) -> Path:
    """Resolve a user-supplied path to the ``zotero.sqlite`` file.

    Accepts either the SQLite file itself or the Zotero data directory that
    contains it.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "zotero.sqlite"
    if not p.is_file():
        raise ZoteroImportError(f"no zotero.sqlite at {path!r}")
    return p


def _connect_readonly(db_path: Path) -> tuple[sqlite3.Connection, Path | None]:
    """Open the DB read-only; copy to a temp file if Zotero holds a lock."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        conn.execute("SELECT 1 FROM items LIMIT 1")
        return conn, None
    except sqlite3.Error:
        tmp = Path(tempfile.mkstemp(suffix=".sqlite")[1])
        shutil.copy2(db_path, tmp)
        return sqlite3.connect(tmp), tmp


def read_zotero_references(path: str | Path) -> list[Reference]:
    """Read every standalone item in a Zotero library as a :class:`Reference`.

    Args:
        path: The ``zotero.sqlite`` file or its containing data directory.

    Returns:
        One :class:`Reference` per non-attachment/non-note item, with
        ``source="zotero"``, ``source_key`` set to the Zotero item key, and
        ``pdf_path`` pointing at the first attached PDF on disk (if any).

    Raises:
        ZoteroImportError: The path holds no readable ``zotero.sqlite``.
    """
    db_path = resolve_zotero_db(path)
    storage_root = db_path.parent / "storage"
    conn, tmp = _connect_readonly(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return _read(conn, storage_root)
    finally:
        conn.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _read(conn: sqlite3.Connection, storage_root: Path) -> list[Reference]:
    deleted = {r[0] for r in conn.execute("SELECT itemID FROM deletedItems")}

    # itemID -> {fieldName: value}
    fields: dict[int, dict[str, str]] = {}
    for row in conn.execute(
        """
        SELECT id.itemID AS itemID, f.fieldName AS name, idv.value AS value
        FROM itemData id
        JOIN fields f ON f.fieldID = id.fieldID
        JOIN itemDataValues idv ON idv.valueID = id.valueID
        """
    ):
        fields.setdefault(row["itemID"], {})[row["name"]] = row["value"]

    # itemID -> ["First Last", ...] (authors, in order)
    authors: dict[int, list[str]] = {}
    for row in conn.execute(
        """
        SELECT ic.itemID AS itemID, c.firstName AS first, c.lastName AS last
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
        WHERE ct.creatorType = 'author'
        ORDER BY ic.itemID, ic.orderIndex
        """
    ):
        name = " ".join(part for part in (row["first"], row["last"]) if part)
        if name:
            authors.setdefault(row["itemID"], []).append(name)

    # itemID -> [tag, ...]
    tags: dict[int, list[str]] = {}
    for row in conn.execute(
        """
        SELECT it.itemID AS itemID, t.name AS name
        FROM itemTags it JOIN tags t ON t.tagID = it.tagID
        """
    ):
        tags.setdefault(row["itemID"], []).append(row["name"])

    # parentItemID -> first PDF attachment path on disk
    pdfs: dict[int, str] = {}
    for row in conn.execute(
        """
        SELECT ia.parentItemID AS parent, i.key AS att_key,
               ia.path AS path, ia.contentType AS ctype
        FROM itemAttachments ia
        JOIN items i ON i.itemID = ia.itemID
        WHERE ia.parentItemID IS NOT NULL
        """
    ):
        if row["parent"] in pdfs or not row["path"]:
            continue
        if row["ctype"] and row["ctype"] != "application/pdf":
            continue
        raw = row["path"]
        if raw.startswith("storage:"):
            resolved = storage_root / row["att_key"] / raw[len("storage:") :]
        else:
            resolved = Path(raw.replace("attachments:", "", 1))
        pdfs[row["parent"]] = str(resolved)

    references: list[Reference] = []
    for row in conn.execute(
        """
        SELECT i.itemID AS itemID, i.key AS key, it.typeName AS type
        FROM items i JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        """
    ):
        item_id = row["itemID"]
        if item_id in deleted or row["type"] in _NON_REFERENCE_TYPES:
            continue
        fld = fields.get(item_id, {})
        title = fld.get("title")
        if not title:
            continue
        extra = fld.get("extra", "")
        url = fld.get("url")
        arxiv = _extract_arxiv(extra, url)
        references.append(
            Reference(
                key=row["key"],
                title=title,
                authors=tuple(authors.get(item_id, ())),
                year=_extract_year(fld.get("date")),
                venue=fld.get("publicationTitle") or fld.get("proceedingsTitle"),
                arxiv=arxiv,
                doi=fld.get("DOI"),
                url=url,
                tags=tuple(tags.get(item_id, ())),
                note=fld.get("abstractNote", "") or "",
                pdf_path=pdfs.get(item_id),
                source="zotero",
                source_key=row["key"],
            )
        )
    references.sort(key=lambda r: (r.year or 0, r.title.lower()))
    return references


def _extract_year(date: str | None) -> int | None:
    if not date:
        return None
    match = re.search(r"\d{4}", date)
    return int(match.group()) if match else None


def _extract_arxiv(extra: str, url: str | None) -> str | None:
    for text in (extra, url or ""):
        if "arxiv" in text.lower():
            match = _ARXIV_RE.search(text)
            if match:
                return match.group(1)
    return None
