"""Knowledge-native tools for the emergent :class:`InteractiveLoop` loop.

Two plain-Python callables — :func:`search_knowledge`, :func:`read_knowledge` —
handed to pydantic-ai alongside the generic file tools. They wrap the
workspace's OKF ``Bundle`` verbs (body-aware :meth:`Bundle.search`,
path-as-identity :meth:`Bundle.get`, reverse :meth:`Bundle.backlinks`) so the
model reads *knowledge* — typed items, sources, edges — instead of grepping
raw markdown by luck.

Same discipline as :mod:`.tools`: read-only, confined to the workspace root
(``_safe_path`` rejects escapes before any I/O), hard caps on every list, no
``pydantic_ai`` import (bare callables introspected by signature + docstring).
Import direction: agent→workspace (legal per the layer DAG).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .tools import _MAX_FILE_BYTES, _safe_path

if TYPE_CHECKING:
    from molexp.workspace import Bundle

__all__ = ["knowledge_tools"]

_MAX_SEARCH_ROWS = 20
"""Hits rendered per ``search_knowledge`` call."""

_MAX_EDGE_ROWS = 20
"""Out-edges / backlinks / sources rendered per ``read_knowledge`` call."""


def _bundle(root: Path) -> Bundle:
    from molexp.workspace import Bundle

    return Bundle(root)


def _meta_head(meta: Mapping[str, object]) -> list[str]:
    """Render the identity lines of a Concept's ``meta.yaml`` head."""
    rows = [f"type: {meta.get('type', 'unknown')}"]
    for field in ("kind", "status", "created_by"):
        if meta.get(field):
            rows.append(f"{field}: {meta[field]}")
    sources = meta.get("sources")
    if isinstance(sources, list) and sources:
        rows.append("sources:")
        for source in sources[:_MAX_EDGE_ROWS]:
            if isinstance(source, dict):
                rows.append(f"  - {source.get('kind', '?')}: {source.get('ref', '?')}")
        if len(sources) > _MAX_EDGE_ROWS:
            rows.append(f"  (+{len(sources) - _MAX_EDGE_ROWS} more sources)")
    return rows


def knowledge_tools(workspace_root: Path) -> tuple[Callable[..., str], ...]:
    """Build the knowledge tool callables confined to ``workspace_root``.

    Args:
        workspace_root: Directory (the OKF bundle root) both tools read from.

    Returns:
        ``(search_knowledge, read_knowledge)`` in a stable order.
    """
    root = Path(workspace_root)

    def search_knowledge(query: str, concept_type: str | None = None) -> str:
        """Search the workspace's knowledge base by content.

        Use this BEFORE planning or answering questions about prior
        experiments: it searches the workspace's knowledge notes, findings,
        failure analyses, decisions, and literature references — matching
        titles, tags, paths, and the notes' body text. Read a promising hit
        in full with ``read_knowledge``.

        Args:
            query: Case-insensitive text to look for (e.g. a parameter name,
                a phenomenon, a material).
            concept_type: Optional exact Concept type filter, e.g.
                ``"knowledge.item"`` or ``"knowledge.note"``.
        """
        result = _bundle(root).search(query, concept_type=concept_type, limit=_MAX_SEARCH_ROWS)
        if not result.hits:
            return f"no knowledge matches for {query!r}"
        rows = []
        for hit in result.hits:
            entry = hit.entry
            row = f"{entry.path} · {entry.type} · {entry.title or entry.path}"
            if hit.snippet:
                row += f" · {hit.snippet}"
            rows.append(row)
        if result.truncated:
            rows.append(f"(truncated at {_MAX_SEARCH_ROWS} hits — refine the query)")
        return "\n".join(rows)

    def read_knowledge(path: str) -> str:
        """Read one knowledge Concept in full: metadata, body, and edges.

        Use after ``search_knowledge`` surfaces a promising path. Returns
        the Concept's typed head (kind / status / sources for a knowledge
        item), its narrative body, what it cites (out-edges), and what cites
        it (backlinks).

        Args:
            path: The Concept's workspace-relative path exactly as returned
                by ``search_knowledge`` (e.g.
                ``"projects/p/experiments/e/finding-abc"``).
        """
        from molexp.workspace.errors import ConceptNotFoundError

        _safe_path(root, path)  # confinement first — before any bundle I/O
        bundle = _bundle(root)
        try:
            concept = bundle.get(path)
        except (ConceptNotFoundError, FileNotFoundError) as exc:
            raise ValueError(
                f"no knowledge Concept at {path!r} — use search_knowledge to find valid paths"
            ) from exc

        rows = [f"# {path}", *_meta_head(concept.read_meta())]

        body = concept.read_index() or ""
        raw = body.encode("utf-8")
        if len(raw) > _MAX_FILE_BYTES:
            body = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="ignore") + "\n... (body truncated)"
        rows += ["", "## body", body.strip() or "(empty body)"]

        edges = concept.typed_out_edges()[:_MAX_EDGE_ROWS]
        rows += ["", "## cites (out-edges)"]
        rows += [f"- [{edge.role}] {edge.target}" for edge in edges] or ["(none)"]

        backlinks = bundle.backlinks(concept)[:_MAX_EDGE_ROWS]
        rows += ["", "## cited by (backlinks)"]
        rows += [f"- [{link.role}] {bundle.rel_path(link.source)}" for link in backlinks] or [
            "(none)"
        ]
        return "\n".join(rows)

    return (search_knowledge, read_knowledge)
