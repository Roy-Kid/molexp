"""``AssembleKnowledgeContext`` — the plan pipeline's prior-knowledge digest.

Persists the workspace's accumulated knowledge (typed ``KnowledgeItem``s,
free-form ``Note``s, literature ``ReferenceConcept``s) as a first-class
``knowledge_context`` artifact the proposal/spec writers consume — so "which
prior knowledge shaped this plan" is an auditable lineage edge through the
gateway's ``parent_ids`` path, never a prompt-side side-channel.

Selection is **deterministic, no LLM**: only ``status == "active"``
KnowledgeItems contribute bodies, ordered by kind priority
(``FailureAnalysis`` first — pitfalls outrank results) then recency;
Note/Reference entries appear as title-only lines. Every cap states what it
omitted (no silent caps), and an empty workspace still persists a digest
saying so — the pipeline shape is uniform, so ``require_latest`` consumers
never go conditional.

The stage locates the owning workspace by walking up from
``ctx.workspace_root`` (a plan run's ctx roots at the *run dir*) to the
nearest ``workspace.json``; a run mounted outside any workspace yields the
honest "not inside a workspace" digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.schemas import PlanArtifactRef

if TYPE_CHECKING:
    from molexp.workspace.folder import Folder

__all__ = ["AssembleKnowledgeContext"]

MAX_ITEMS = 12
"""KnowledgeItem bodies included in one digest."""

MAX_BODY_CHARS = 1200
"""Per-item body excerpt length."""

MAX_TOTAL_CHARS = 16_000
"""Whole-digest hard cap."""

#: Kind priority — pitfalls first, observations last.
_KIND_ORDER = (
    "FailureAnalysis",
    "Finding",
    "Decision",
    "Constraint",
    "Assumption",
    "ParameterRationale",
    "ProtocolNote",
    "OpenQuestion",
    "Observation",
)

_TS_FLOOR = datetime(1970, 1, 1, tzinfo=UTC)

_EMPTY_DIGEST = "no prior knowledge recorded in this workspace"
_NO_WORKSPACE_DIGEST = "no prior knowledge available (run is not mounted inside a workspace)"


def _find_workspace_root(start: Path) -> Path | None:
    """Walk up from *start* to the nearest dir holding ``workspace.json``."""
    for candidate in (start, *start.parents):
        if (candidate / "workspace.json").is_file():
            return candidate
    return None


class AssembleKnowledgeContext(Stage):
    """Digest the workspace's prior knowledge into one plan artifact."""

    name: ClassVar[str] = "assemble_knowledge_context"

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        digest = self._render_digest(Path(ctx.workspace_root))
        return ctx.artifact_store.put_text(
            kind="knowledge_context",
            text=digest,
            created_by=self.name,
            parent_ids=[],
        )

    def _render_digest(self, start: Path) -> str:
        from molexp.workspace import Bundle, Workspace, assemble_workspace_context
        from molexp.workspace.knowledge_item import KNOWLEDGE_ITEM_KIND

        root = _find_workspace_root(start)
        if root is None:
            return _NO_WORKSPACE_DIGEST
        refs = assemble_workspace_context(Workspace(root)).knowledge
        if not refs:
            return _EMPTY_DIGEST

        from molexp.workspace.errors import ConceptNotFoundError

        bundle = Bundle(root)
        items: list[tuple[int, datetime, str, str, Mapping[str, object], Folder]] = []
        titles: list[str] = []
        for ref in refs:
            if ref.type != KNOWLEDGE_ITEM_KIND:
                titles.append(f"- {ref.title} ({ref.type}) · {ref.path}")
                continue
            try:
                concept = bundle.get(ref.path)
            except ConceptNotFoundError, FileNotFoundError:
                # The inventory raced a concurrent deletion — record the gap
                # honestly instead of silently shrinking the digest.
                titles.append(f"- (vanished during digest) · {ref.path}")
                continue
            meta = concept.read_meta()
            if meta.get("status", "active") != "active":
                continue  # stale/superseded/conflicting knowledge never grounds a plan
            kind = str(meta.get("kind", "Observation"))
            priority = _KIND_ORDER.index(kind) if kind in _KIND_ORDER else len(_KIND_ORDER)
            items.append((priority, self._timestamp(meta), ref.path, ref.title, meta, concept))

        # kind priority asc, then recency desc (missing timestamps sort oldest).
        items.sort(key=lambda row: (row[0], -row[1].timestamp(), row[2]))

        lines = ["# Prior knowledge (workspace digest)", ""]
        for _, _, path, title, meta, concept in items[:MAX_ITEMS]:
            body = self._body(concept)
            lines.append(f"## [{meta.get('kind')}] {title}")
            lines.append(f"path: {path} · status: {meta.get('status', 'active')}")
            if body:
                lines.append(body)
            lines.append("")
        if len(items) > MAX_ITEMS:
            lines.append(f"(+{len(items) - MAX_ITEMS} more knowledge items not shown)")
        if titles:
            lines.append("")
            lines.append("## Other knowledge (titles only)")
            lines.extend(titles)

        digest = "\n".join(lines).strip()
        if len(digest) > MAX_TOTAL_CHARS:
            digest = digest[:MAX_TOTAL_CHARS] + "\n... (digest truncated at the total cap)"
        return digest or _EMPTY_DIGEST

    @staticmethod
    def _timestamp(meta: Mapping[str, object]) -> datetime:
        raw = meta.get("timestamp")
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                return _TS_FLOOR
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return _TS_FLOOR

    @staticmethod
    def _body(concept: Folder) -> str:
        body = (concept.read_index() or "").strip()
        if len(body) > MAX_BODY_CHARS:
            return body[:MAX_BODY_CHARS] + f"\n... (+{len(body) - MAX_BODY_CHARS} chars omitted)"
        return body
