"""LibraryIndex — the machine- and agent-readable index of a scope's library.

``Library.build_index()`` derives this from the authoritative sources (the
``assets.json`` notes + ``references.json``) and writes two sibling files
under ``library/``:

* ``index.json`` — structured, parsed by the UI and the agent's
  ``search_library`` tool;
* ``INDEX.md`` — a human- and agent-readable rendering (same spirit as
  ``.claude/specs/INDEX.md``), so a glance or a ``read_file`` surfaces every
  note and reference without opening each one.

Both are *derived* — never the only copy of anything (workspace
"one source of truth" law).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .reference import Reference

SCHEMA_VERSION = 1
INDEX_JSON_FILENAME = "index.json"
INDEX_MD_FILENAME = "INDEX.md"


class NoteEntry(BaseModel):
    """A note's index row — enough to triage without reading the body."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    title: str
    path: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    updated_at: str | None = None


class LibraryIndex(BaseModel):
    """The full library index for one scope."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    scope: str
    generated_at: str
    notes: tuple[NoteEntry, ...] = ()
    references: tuple[Reference, ...] = ()

    def to_markdown(self) -> str:
        """Render the index as a compact, agent-friendly markdown document."""
        lines: list[str] = [f"# Library — {self.scope}", ""]
        lines.append(
            f"_{len(self.notes)} note(s), {len(self.references)} reference(s). "
            f"Generated {self.generated_at}. Machine-readable: `index.json`._"
        )
        lines.append("")

        lines.append("## Notes")
        lines.append("")
        if self.notes:
            for n in self.notes:
                gloss = f" — {n.summary}" if n.summary else ""
                tags = f"  `{' '.join(n.tags)}`" if n.tags else ""
                cites = f"  (cites: {', '.join(n.refs)})" if n.refs else ""
                lines.append(f"- [{n.title}]({n.path}){gloss}{tags}{cites}")
        else:
            lines.append("_(none)_")
        lines.append("")

        lines.append("## References")
        lines.append("")
        if self.references:
            for r in self.references:
                ident_parts = []
                if r.arxiv:
                    ident_parts.append(f"arXiv:{r.arxiv}")
                if r.doi:
                    ident_parts.append(f"doi:{r.doi}")
                ident = f" [{'; '.join(ident_parts)}]" if ident_parts else ""
                year = f" ({r.year})" if r.year else ""
                authors = f" — {r.authors[0]} et al." if r.authors else ""
                tags = f"  `{' '.join(r.tags)}`" if r.tags else ""
                lines.append(f"- **{r.key}**: {r.title}{year}{authors}{ident}{tags}")
                if r.note:
                    lines.append(f"  - {r.note}")
        else:
            lines.append("_(none)_")
        lines.append("")
        return "\n".join(lines)
