"""``write_knowledge_item`` — the single sourced-KnowledgeItem writer.

Workspace-layer core shared by :func:`harvest_run`, plan-record writers
(``services.plan_runtime.record``), and (later) agent session harvest. Placement
is forced by the layer DAG: ``agent → workspace`` and ``services → workspace``
are legal; the reverse is not — so the only host all three can import is
workspace.

**Lifecycle order is load-bearing** (do not reorder):

1. Evaluate ``newly_created`` via ``has_folder`` **before** ``add_folder``.
2. ``add_folder`` → ``write_knowledge_meta`` → ``set_body``.
3. Emit ``knowledge.created`` only when ``newly_created and emit`` (after meta+body).
4. ``cite`` each source non-fatally (warn, never raise after disk is written).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from molexp.path import Path

from .edges import EdgeRole
from .events import emit_workspace_event
from .folder import Folder
from .knowledge_item import (
    KNOWLEDGE_ITEM_KIND,
    KnowledgeItem,
    KnowledgeKind,
    KnowledgeMeta,
    SourceRef,
)

if TYPE_CHECKING:
    pass

__all__ = ["write_knowledge_item"]

_LOG = logging.getLogger(__name__)


def _workspace_root(host: Folder) -> Path | None:
    """Walk ``parent`` to the top Folder and return its resolved path, or None."""
    node: Folder | None = host
    try:
        while node is not None and node.parent is not None:
            node = node.parent
        if node is None:
            return None
        return node.resolve()
    except Exception:
        return None


def write_knowledge_item(
    host: Folder,
    *,
    name: str,
    kind: KnowledgeKind,
    sources: list[SourceRef],
    created_by: str,
    body: str,
    cite: Sequence[tuple[Folder, EdgeRole]] = (),
    title: str = "",
    actor: str | None = None,
    emit: bool = True,
) -> KnowledgeItem:
    """Write a sourced :class:`KnowledgeItem` under *host* (idempotent on *name*).

    Args:
        host: Parent folder (experiment, workspace root, …).
        name: Concept directory name (slug).
        kind: Knowledge category.
        sources: Non-empty source list (enforced by :class:`KnowledgeMeta`).
        created_by: Author string (person or ``agent:…`` / ``PlanMode/…``).
        body: Markdown body for ``index.md``.
        cite: Optional ``(Folder, EdgeRole)`` pairs for typed out-edges.
        title: Optional title for the event payload (defaults to *name*).
        actor: Event actor; defaults to *created_by* when omitted.
        emit: When False, skip ``knowledge.created`` even on first create.

    Returns:
        The written :class:`KnowledgeItem`.

    Raises:
        ValidationError: Empty *sources* (meta validator).
        Exception: Host not mounted / ``add_folder`` failures propagate.
    """
    newly_created = not host.has_folder(name, cls=KnowledgeItem)
    item = host.add_folder(KnowledgeItem(name=name))
    item.write_knowledge_meta(
        KnowledgeMeta(
            kind=kind,
            sources=sources,
            created_by=created_by,
            timestamp=datetime.now(UTC),
        )
    )
    item.set_body(body)

    if newly_created and emit:
        root = _workspace_root(host)
        if root is not None:
            try:
                rel = item.resolve().relative_to(root).as_posix()
            except Exception:
                _LOG.debug(
                    "write_knowledge_item: skip knowledge.created (path relative_to failed)",
                    exc_info=True,
                )
            else:
                try:
                    emit_workspace_event(
                        root,
                        "knowledge.created",
                        actor if actor is not None else created_by,
                        payload={"type": KNOWLEDGE_ITEM_KIND, "title": title or name},
                        refs=[rel],
                    )
                except Exception:
                    # emit_workspace_event is already non-fatal; belt-and-braces
                    # if a caller patches the emit site to raise.
                    _LOG.debug(
                        "write_knowledge_item: knowledge.created emit raised",
                        exc_info=True,
                    )
        else:
            _LOG.debug("write_knowledge_item: skip knowledge.created (workspace root unresolvable)")

    for source, role in cite:
        try:
            item.cite(source, role=role)
        except Exception:
            _LOG.warning(
                "write_knowledge_item: cite failed for %s role=%s",
                getattr(source, "name", source),
                role,
                exc_info=True,
            )

    return item
