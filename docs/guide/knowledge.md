# Notes and Literature

A workspace does not only record what a computation produced — it also records why you ran it, what you concluded, and which papers you built on. In MolExp that knowledge lives in the same directory tree as your projects, experiments, and runs, encoded in the **Open Knowledge Format (OKF)**: plain directories, plain YAML, plain Markdown. There is no separate notes database to keep in sync with the filesystem, because the filesystem *is* the database.

## The mental model

One paragraph carries the whole design. A note is a directory: its path is its identity, a small `meta.yaml` marks what kind of Concept the directory is (`note.note`, `reference.reference`, ...), and `index.md` holds the narrative. The Markdown links inside `index.md` are not decoration — they **are** the knowledge graph. Every link is a typed edge to another Concept, and reverse lookups (backlinks) are recomputed from those links rather than stored in a second index. Heavy payloads such as PDFs are *pointed at* through a path recorded in metadata — never copied into the workspace.

Because every workspace entity (`Workspace`, `Project`, `Experiment`, `Run`) is itself a `Folder` with a `meta.yaml`, notes and references mount anywhere in the hierarchy and can link to anything in it: a note under an experiment can cite a paper, reference a run, or point at a sibling note, all with the same Markdown-link edge.

The management entry point is the `Bundle` façade. A bundle wraps a root directory (typically the workspace root) and exposes the Concept tree beneath it: `create_note`, `link`, `walk`, `backlinks`, `search`, `import_zotero`.

!!! note "Opening the workspace in Obsidian (or any Markdown editor)"
    Because notes are plain directories of Markdown, you can open the workspace root as an Obsidian vault and every `index.md` renders and cross-links normally. Two caveats: MolExp's knowledge graph is built from **standard Markdown links** (`[text](../other-concept)`) — Obsidian-style `[[wikilinks]]` are *not* parsed as edges, so write links in standard form (or configure Obsidian to prefer Markdown links) if you want them to count as citations/backlinks. And each Concept's narrative lives in `index.md` inside its own directory — the "folder note" convention — which reads best in Obsidian with a folder-note plugin enabled.

## Write a note on an experiment

Create a note under an experiment by passing the experiment as the note's `parent`. Names are slugified into directory names, and `create_note` is idempotent — calling it again with the same name returns the existing note instead of duplicating it.

```python
from molexp.workspace import Bundle, Workspace

ws = Workspace("./lab", name="Lab")
exp = ws.project("polymer-cg").experiment("solvation-sweep")

bundle = Bundle(ws.root)
note = bundle.create_note(
    "Analysis Notes",
    parent=exp,
    body="# Analysis Notes\n\nThe RDF is converged after 5 ns.\n",
)

print(bundle.rel_path(note))
# projects/polymer-cg/experiments/solvation-sweep/analysis-notes
```

The `body` is the note's `index.md`; read it back with `note.body()` and replace it with `note.set_body(...)`. Structured document metadata — categorical tags and a lifecycle status — lives in the note's `meta.yaml` as a typed `NoteMeta` payload, written with `write_note_meta`:

```python
from molexp.workspace import NoteMeta

note.write_note_meta(NoteMeta(tags=["analysis", "rdf"], status="draft"))

print(note.tags())    # ['analysis', 'rdf']
print(note.status())  # draft
```

On disk the note is an ordinary directory next to the experiment's other contents:

```text
lab/projects/polymer-cg/experiments/solvation-sweep/
└── analysis-notes/
    ├── meta.yaml       # type: note.note — plus tags and status
    ├── index.md        # the narrative; its links are the graph edges
    └── metadata.json   # Folder mount bookkeeping (derived)
```

## Attach a literature reference

A reference is its own Concept type, `ReferenceConcept`: one directory per work, with the structured bibliographic record (`ReferenceMeta`) in `meta.yaml` and the human-readable citation text in `index.md`. Mount it wherever it belongs — here, next to the note under the same experiment — using the generic `add_folder` verb every `Folder` supports:

```python
from molexp.workspace import ReferenceConcept, ReferenceMeta

ref = exp.add_folder(ReferenceConcept(parent=exp, name="frenkel-smit-2002"))
ref.write_reference_meta(
    ReferenceMeta(
        title="Understanding Molecular Simulation",
        authors=("Daan Frenkel", "Berend Smit"),
        year=2002,
        pdf_path="/home/me/Zotero/storage/ABCD1234/frenkel-smit.pdf",
    )
)
ref.set_citation("Frenkel & Smit, *Understanding Molecular Simulation* (2002).\n")
```

`ReferenceMeta` also carries `doi`, `venue`, `url`, and provenance fields (`source`, `source_key`); `pdf_path` records where the PDF already lives — the bytes stay in place.

Citing the reference from the note writes one typed Markdown link into the note's `index.md`:

```python
note.cite(ref, role="cites")

print(note.body())
# # Analysis Notes
#
# The RDF is converged after 5 ns.
# - [@cites frenkel-smit-2002](../frenkel-smit-2002)
```

The last line is the whole persistence story: a relative Markdown link, with the edge's role riding in the link label (`@cites ...`). The role vocabulary is fixed — `derived_from`, `cites`, `supersedes`, `records`, `references` — and `references` is the default, so a plain unlabeled link still reads back as a valid edge. `Bundle.link(src, dst, role=...)` writes the same edge between any two Concepts when the source is not a `Note`.

## Query the graph: typed edges and backlinks

Outgoing edges are read straight from a Concept's `index.md`. `typed_out_edges()` returns `Edge` rows pairing the resolved target path with the declared role:

```python
for edge in note.typed_out_edges():
    print(edge.role, "->", edge.target)
# cites -> /.../experiments/solvation-sweep/frenkel-smit-2002
```

The reverse direction is a derived view. `Bundle.backlinks(concept)` walks the bundle and returns a list of `Backlink` objects — each one a `NamedTuple` wrapper `Backlink(source, role)`, not a bare Concept: `source` is the `Folder` whose `index.md` holds the link, and `role` is that edge's declared role. No reverse index is written to disk; the answer is always recomputed from the Markdown source of truth.

```python
for backlink in bundle.backlinks(ref):
    print(f"{bundle.rel_path(backlink.source)} links here (role={backlink.role})")
# projects/polymer-cg/experiments/solvation-sweep/analysis-notes links here (role=cites)
```

Typed views over the whole bundle come from the same walk, and `bundle.search(text, concept_type=..., tag=...)` filters a derived index when the tree grows large:

```python
print([bundle.rel_path(n) for n in bundle.notes()])
print([bundle.rel_path(r) for r in bundle.references()])
```

## Import a Zotero library

If your papers already live in Zotero, you do not have to retype them. `molexp knowledge import-zotero` links a local Zotero library into a workspace: it opens Zotero's own `zotero.sqlite` **strictly read-only** and materializes each item as a `ReferenceConcept` under `<workspace>/references/` — bibliographic fields into `meta.yaml`, and each item's PDF pointed at inside Zotero's own `storage/` tree. No bytes are copied, and your Zotero library is never modified.

```console
$ molexp knowledge import-zotero ~/Zotero/zotero.sqlite --dest ./lab
OK Imported 2 reference(s) into /home/me/lab
  references/frnkl2002  Understanding Molecular Simulation (2002)
  references/krmr1990  Dynamics of entangled linear polymer melts (1990)
```

The argument is the `zotero.sqlite` file in your Zotero data directory (passing the directory itself also works); `--dest` is the workspace to import into, defaulting to the current directory. Close Zotero before importing — the running application holds a lock on its database, and the command will tell you exactly that rather than fail cryptically. The import is idempotent on the Zotero item key: re-running it updates the existing reference directories instead of duplicating them, and the linked source is recorded in `sources.json` at the workspace root.

If something is wrong, the command says so in plain language and exits non-zero:

```console
$ molexp knowledge import-zotero ~/Zotero/zotero.sqlite --dest /tmp/scratch
Error: Destination is not a molexp workspace: /tmp/scratch
Initialise one first with molexp init /tmp/scratch.
```

The same import is one call in Python — `Bundle.import_zotero` — so scripts and the CLI share a single code path:

```python
# docs: skip — needs a real Zotero library on the executing machine
from pathlib import Path

refs = bundle.import_zotero(Path.home() / "Zotero" / "zotero.sqlite")
print(f"linked {len(refs)} Zotero items")
```

Once imported, the references are ordinary Concepts: cite them from notes with `note.cite(ref)`, find who cites them with `bundle.backlinks(ref)`, and filter them with `bundle.references()` — exactly as in the sections above.
