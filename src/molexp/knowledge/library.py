"""``Library`` — the OKF bundle façade for ``molexp.knowledge``.

A :class:`Library` wraps a *bundle root* (a directory) and exposes the whole
Concept tree beneath it — at any depth — as one management entry point. A
directory is a **Concept** iff it directly holds ``meta.yaml`` (okf-01-03);
the ``_ops/`` operational sidecar and its descendants are never Concepts, and
plain organizational dirs (no ``meta.yaml``) are walked *through* but never
*yielded*.

It is a thin runtime container (explicit ``__init__``, no pydantic), mirroring
the lazy-façade idiom of ``molexp.workspace.library.Library``: it records the
root path and does **no** disk I/O on construction. Every Concept it produces
is an okf-01-03 :class:`Folder`; the semantic graph lives in markdown
(``index.md`` links), so :meth:`link` round-trips through
:meth:`Folder.out_edges`.

This is the okf-01-04 skeleton: ``walk`` / ``get`` / ``put`` / ``link`` only.
``build_index()``, ``search()``, Concept subtypes, references/notes and Zotero
import are deferred to later specs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from .errors import ConceptNotFoundError
from .folder import OPS_DIR, Folder, _is_concept_dir
from .models import ConceptMeta

__all__ = ["Library"]


class Library:
    """A management façade over an OKF bundle (a Concept-directory tree)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Record the bundle *root*; perform no disk I/O (lazy)."""
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The bundle root directory."""
        return self._root

    # ── identity helpers ─────────────────────────────────────────────────

    def rel_path(self, concept: Folder) -> str:
        """Return *concept*'s identity: its POSIX path relative to the root."""
        return Path(concept.resolve()).relative_to(self._root).as_posix()

    def _folder_for(self, path: Path) -> Folder:
        """Build the :class:`Folder` whose identity is the Concept dir *path*."""
        return Folder(name=path.name, root=path.parent)

    # ── walk / get / put / link ──────────────────────────────────────────

    def walk(self) -> Iterator[Folder]:
        """Yield every Concept under the root, depth-first (preorder).

        A dir is yielded iff it holds ``meta.yaml``. The ``_ops/`` sidecar
        (and everything beneath it) is skipped; non-Concept organizational
        dirs are descended into but not yielded.
        """
        yield from self._walk_dir(self._root)

    def _walk_dir(self, directory: Path) -> Iterator[Folder]:
        if not directory.is_dir():
            return
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or entry.name == OPS_DIR:
                continue
            if _is_concept_dir(entry):
                yield self._folder_for(entry)
            yield from self._walk_dir(entry)

    def get(self, rel_path: str | os.PathLike[str]) -> Folder:
        """Resolve a bundle-relative path to its :class:`Folder`.

        Raises:
            ConceptNotFoundError: if *rel_path* is not a Concept dir.
        """
        rel = PurePosixPath(os.fspath(rel_path))
        target = self._root.joinpath(*rel.parts)
        if not _is_concept_dir(target):
            raise ConceptNotFoundError(str(rel_path))
        return self._folder_for(target)

    def put(self, concept: Folder) -> Folder:
        """Idempotently materialize *concept* (write ``meta.yaml`` if absent)."""
        if not _is_concept_dir(concept.resolve()):
            concept.write_meta(ConceptMeta(type=concept.concept_type))
        return concept

    def link(self, src: Folder, dst: Folder, *, text: str | None = None) -> None:
        """Record a semantic edge ``src → dst`` as a markdown link in ``src``.

        Appends a real markdown link (relative to ``src``) to ``src/index.md``
        so :meth:`Folder.out_edges` resolves it back to *dst*. The graph lives
        in markdown, never in ``meta.yaml``. This skeleton appends
        unconditionally; dedup/edge-typing are deferred to okf-03.
        """
        rel = os.path.relpath(Path(dst.resolve()), Path(src.resolve()))
        label = text or dst.name
        line = f"- [{label}]({rel})\n"
        src.write_index(src.read_index() + line)
