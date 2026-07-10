"""Session harvest + export — agent-record-export-05.

``harvest_session`` turns an on-disk :class:`~molexp.agent.folders.AgentSession`
into a sourced :class:`~molexp.workspace.knowledge_item.KnowledgeItem` via
:func:`~molexp.workspace.knowledge_write.write_knowledge_item`.

``export_session_zip`` archives the session folder via
:func:`~molexp.workspace.archive.archive_folder_zip` — the single zip writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.ids import slugify
from molexp.workspace.archive import archive_folder_zip
from molexp.workspace.knowledge_item import KnowledgeItem, KnowledgeKind, SourceRef
from molexp.workspace.knowledge_write import write_knowledge_item

if TYPE_CHECKING:
    from molexp.agent.folders import AgentSession
    from molexp.workspace.folder import Folder

__all__ = ["export_session_zip", "harvest_session"]


def harvest_session(
    session: AgentSession,
    *,
    kind: KnowledgeKind,
    narrative: str,
    created_by: str,
    host: Folder | None = None,
    name: str | None = None,
) -> KnowledgeItem:
    """Harvest a finished agent session into a typed KnowledgeItem.

    Args:
        session: On-disk agent session folder (has messages / meta).
        kind: Knowledge category.
        narrative: Non-empty interpretation of the session.
        created_by: Author string.
        host: Parent for the KnowledgeItem; defaults to the session's parent
            folder (usually the Agent or workspace mount).
        name: Optional Concept name; default ``session-harvest-{slug}-{id}``.

    Returns:
        The written KnowledgeItem.

    Raises:
        ValueError: Empty narrative.
    """
    if not narrative.strip():
        raise ValueError(
            "harvest_session requires a non-empty narrative — knowledge is interpretation"
        )
    parent = host if host is not None else session.parent
    if parent is None:
        raise ValueError("harvest_session needs a host Folder (session has no parent)")

    item_name = name or f"session-{slugify(kind)}-{session.name}"
    body_lines = [
        f"# [{kind}] agent session {session.name}",
        "",
        narrative.strip(),
        "",
        "## Session",
        "",
        f"- status: {session.status}",
        f"- goal: {session.goal_summary or '(none)'}",
    ]
    # Optional lossless model history size (feedstock for export).
    try:
        n_msgs = len(session.read_messages())
        body_lines.append(f"- model_messages: {n_msgs}")
    except Exception:
        pass
    body = "\n".join(body_lines).rstrip() + "\n"

    return write_knowledge_item(
        parent,
        name=item_name,
        kind=kind,
        sources=[
            SourceRef(kind="agent_action", ref=session.name),
        ],
        created_by=created_by,
        body=body,
        cite=[(session, "derived_from")],
        title=f"Session {session.name}",
        actor="agent-harvest",
    )


def export_session_zip(session: AgentSession) -> bytes:
    """Zip the session directory via the workspace archive core.

    Returns:
        Zip bytes of the session folder (meta, entries, messages, …).
    """
    return archive_folder_zip(session)
