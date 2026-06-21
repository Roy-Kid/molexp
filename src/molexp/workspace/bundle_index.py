"""Derived bundle index models for ``molexp.workspace`` (OKF rollup).

:meth:`molexp.workspace.Bundle.build_index` walks the whole Concept tree and
rolls each Concept's identity (path / type / id / tags / title / out-edges) into
a :class:`BundleIndex`, then writes two **derived** siblings at the bundle root:
``index.json`` (machine-readable) and ``INDEX.md`` (human/agent-readable, same
spirit as ``.claude/specs/INDEX.md``). Both are rebuilt on demand from the
authoritative ``meta.yaml`` + ``index.md`` graph — never a source of truth (the
workspace "one source of truth" law).

This is a NEW model distinct from the per-scope ``workspace.LibraryIndex`` (an
index *of notes and references inside one scope*); ``BundleIndex`` is an index
*of every Concept in a directory subtree*.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel

INDEX_JSON_FILENAME = "index.json"
INDEX_MD_FILENAME = "INDEX.md"

# First markdown H1 (``# Title``) line, if any.
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract_title(index_md: str) -> str | None:
    """Return the first markdown H1 in *index_md*, or ``None`` if absent.

    Args:
        index_md: The raw ``index.md`` narrative text.

    Returns:
        The H1 title (without the leading ``#``), or ``None`` when there is no
        H1 heading.
    """
    match = _H1.search(index_md)
    return match.group(1).strip() if match else None


class ConceptIndexEntry(BaseModel, frozen=True):
    """One Concept's row in the derived bundle index.

    Attributes:
        path: Bundle-relative POSIX path — the Concept's identity.
        type: The Concept's ``meta.yaml`` type.
        id: Optional stable id from ``meta.yaml``.
        title: The ``index.md`` H1 if present, else the Concept name.
        tags: Categorical labels from ``meta.yaml``.
        links: Out-edges (in-tree Concept targets) as bundle-relative paths.
    """

    path: str
    type: str
    id: str | None = None
    title: str = ""
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()


class BundleIndex(BaseModel, frozen=True):
    """The derived rollup of a bundle's Concept tree.

    Attributes:
        generated_at: Aware-UTC build timestamp, or ``None`` if unset.
        entries: One :class:`ConceptIndexEntry` per Concept, in walk order.
    """

    generated_at: datetime | None = None
    entries: tuple[ConceptIndexEntry, ...] = ()

    def to_markdown(self) -> str:
        """Render the human/agent-readable ``INDEX.md``."""
        generated = self.generated_at.isoformat() if self.generated_at else "unknown"
        lines = [
            "# Knowledge Index",
            "",
            f"_Derived — rebuilt by `Bundle.build_index()`. "
            f"Machine-readable: `{INDEX_JSON_FILENAME}`. Generated {generated}._",
            "",
        ]
        for entry in self.entries:
            title = entry.title or entry.path
            tags = f" · tags: {', '.join(entry.tags)}" if entry.tags else ""
            lines.append(f"- [{title}]({entry.path}) — `{entry.type}`{tags}")
        return "\n".join(lines) + "\n"


__all__ = [
    "INDEX_JSON_FILENAME",
    "INDEX_MD_FILENAME",
    "BundleIndex",
    "ConceptIndexEntry",
    "extract_title",
]
