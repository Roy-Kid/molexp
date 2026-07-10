"""Notes, references, and the OKF knowledge graph — Bundle, Note, ReferenceConcept, cite, backlinks.

Matches ``docs/guide/knowledge.md``.

Demonstrates:

1. ``Bundle(ws.root)`` — the management entry point for OKF Concepts.
2. ``bundle.create_note(...)`` — idempotent note creation with body + parent.
3. ``note.body()`` / ``note.set_body(...)`` / ``note.write_note_meta(NoteMeta(...))``.
4. ``note.tags()`` / ``note.status()`` — typed metadata read-back.
5. ``ReferenceConcept`` — one directory per work, with ``ReferenceMeta`` bib fields.
6. ``note.cite(ref, role="cites")`` — typed Markdown-link edges.
7. ``note.typed_out_edges()`` — outgoing edge reading.
8. ``bundle.backlinks(ref)`` — reverse lookups (recomputed from Markdown source).
9. ``bundle.notes()`` / ``bundle.references()`` — typed views.
10. ``bundle.search(...)`` — filtered concept search.

Run directly::

    python examples/knowledge/notes_and_references.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me
from molexp.workspace import Bundle, NoteMeta, ReferenceConcept, ReferenceMeta


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-knowledge-"))
    ws = me.Workspace(root, name="knowledge-demo")

    # ── 1. Bundle wraps the workspace root ─────────────────────────────
    bundle = Bundle(ws.root)

    # ── 2. Create a note under an experiment ────────────────────────────
    exp = ws.project("polymer-cg").experiment("solvation-sweep")
    note = bundle.create_note(
        "Analysis Notes",
        parent=exp,
        body="# Analysis Notes\n\nThe RDF is converged after 5 ns.\n",
    )
    print(f"Note created: {bundle.rel_path(note)}")

    # ── 3. Idempotent — calling again returns the same note ─────────────
    same = bundle.create_note("Analysis Notes", parent=exp)
    assert same.path == note.path, "create_note is idempotent"
    print("create_note is idempotent: same path")

    # ── 4. Read / write body and structured metadata ────────────────────
    print(f"body preview: {note.body()[:50]}...")
    note.set_body("# Analysis Notes\n\nUpdated: NVT at 300 K completed.\n")

    note.write_note_meta(NoteMeta(tags=["analysis", "rdf"], status="in-progress"))
    print(f"tags:   {note.tags()}")
    print(f"status: {note.status()}")

    # ── 5. Attach a literature reference ────────────────────────────────
    ref = exp.add_folder(ReferenceConcept(parent=exp, name="frenkel-smit-2002"))
    ref.write_reference_meta(
        ReferenceMeta(
            title="Understanding Molecular Simulation",
            authors=("Daan Frenkel", "Berend Smit"),
            year=2002,
        )
    )
    ref.set_citation("Frenkel & Smit, *Understanding Molecular Simulation* (2002).\n")
    print(f"\nReference created: {bundle.rel_path(ref)}")

    # ── 6. Cite the reference from the note ─────────────────────────────
    note.cite(ref, role="cites")
    print(f"body after citation:\n{note.body()}")

    # ── 7. Typed outgoing edges ────────────────────────────────────────
    print("outgoing edges:")
    for edge in note.typed_out_edges():
        print(f"  role={edge.role} → {edge.target}")

    # ── 8. Backlinks — reverse lookups ──────────────────────────────────
    print("\nbacklinks to the reference:")
    for backlink in bundle.backlinks(ref):
        print(f"  {bundle.rel_path(backlink.source)} links here (role={backlink.role})")

    # ── 9. Typed views over the bundle ──────────────────────────────────
    print(f"\nall notes:      {[bundle.rel_path(n) for n in bundle.notes()]}")
    print(f"all references: {[bundle.rel_path(r) for r in bundle.references()]}")

    # ── 10. Search ───────────────────────────────────────────────────────
    results = bundle.search("RDF", concept_type="note.note")
    print(f"search 'RDF' in notes: {len(results.hits)} hit(s)")
    for hit in results.hits:
        print(f"  {hit.entry.path}")


if __name__ == "__main__":
    asyncio.run(main())
