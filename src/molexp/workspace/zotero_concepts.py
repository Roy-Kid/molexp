"""Read-only Zotero ingestion for the OKF workspace concepts (wsokf-05).

Reads a local ``zotero.sqlite`` (opened read-only) and yields :class:`ZoteroItem`
records. Each item's first PDF is *resolved to a path* under Zotero's own
``storage/`` tree — no bytes are read or copied. :meth:`Bundle.import_zotero`
turns these into :class:`molexp.workspace.concepts.ReferenceConcept` Concepts
whose ``ReferenceMeta.pdf_path`` points at (never copies) the original file.

This is a port of ``molexp.knowledge.zotero`` onto the workspace surface; it is
the *concept-producing* importer and is distinct from the legacy record-path
``molexp.workspace.library.zotero`` (which yields bib-record ``Reference`` rows).
The two coexist until the post-migration cleanup.
"""

from __future__ import annotations

import re
import sqlite3
from os import PathLike
from pathlib import Path

from pydantic import BaseModel

_STORAGE_PREFIX = "storage:"
_YEAR = re.compile(r"\d{4}")


class ZoteroItem(BaseModel, frozen=True):
    """A parsed Zotero library item (bibliographic record + PDF pointer)."""

    key: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    pdf_path: str | None = None


def _parse_year(date_value: str | None) -> int | None:
    if not date_value:
        return None
    match = _YEAR.search(date_value)
    return int(match.group()) if match else None


def _item_fields(conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
    rows = conn.execute(
        "SELECT f.fieldName, idv.value FROM itemData d "
        "JOIN fields f ON d.fieldID = f.fieldID "
        "JOIN itemDataValues idv ON d.valueID = idv.valueID "
        "WHERE d.itemID = ?",
        (item_id,),
    ).fetchall()
    return dict(rows)


def _item_authors(conn: sqlite3.Connection, item_id: int) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT c.firstName, c.lastName FROM itemCreators ic "
        "JOIN creators c ON ic.creatorID = c.creatorID "
        "WHERE ic.itemID = ? ORDER BY ic.orderIndex",
        (item_id,),
    ).fetchall()
    names = [" ".join(p for p in (first, last) if p) for first, last in rows]
    return tuple(n for n in names if n)


def _item_pdf(conn: sqlite3.Connection, item_id: int, storage_root: Path) -> str | None:
    row = conn.execute(
        "SELECT a.path, i.key FROM itemAttachments a "
        "JOIN items i ON a.itemID = i.itemID "
        "WHERE a.parentItemID = ? AND a.path LIKE 'storage:%' "
        "AND a.contentType = 'application/pdf' LIMIT 1",
        (item_id,),
    ).fetchone()
    if row is None:
        return None
    path, attachment_key = row
    filename = path[len(_STORAGE_PREFIX) :]
    return str(storage_root / attachment_key / filename)


def read_zotero_items(path: str | PathLike[str]) -> list[ZoteroItem]:
    """Parse a ``zotero.sqlite`` (read-only) into :class:`ZoteroItem` records.

    Attachments and notes are excluded; each regular item's PDF (if any) is
    resolved to ``<db_dir>/storage/<attachment_key>/<filename>`` without
    touching the file.

    Args:
        path: Path to the ``zotero.sqlite`` database.

    Returns:
        One :class:`ZoteroItem` per non-attachment, non-note library item.
    """
    db = Path(path)
    storage_root = db.parent / "storage"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT i.itemID, i.key FROM items i "
            "JOIN itemTypes it ON i.itemTypeID = it.itemTypeID "
            "WHERE it.typeName NOT IN ('attachment', 'note') "
            "ORDER BY i.itemID"
        ).fetchall()
        items: list[ZoteroItem] = []
        for item_id, key in rows:
            fields = _item_fields(conn, item_id)
            items.append(
                ZoteroItem(
                    key=key,
                    title=fields.get("title"),
                    authors=_item_authors(conn, item_id),
                    year=_parse_year(fields.get("date")),
                    doi=fields.get("DOI"),
                    url=fields.get("url"),
                    pdf_path=_item_pdf(conn, item_id, storage_root),
                )
            )
        return items
    finally:
        conn.close()


__all__ = ["ZoteroItem", "read_zotero_items"]
