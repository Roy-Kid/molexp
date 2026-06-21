"""Shared fixtures for ``molexp.knowledge`` tests.

The :func:`bundle` fixture materializes a small OKF bundle on disk exercising
every shape :class:`molexp.knowledge.Library` must handle: nested Concepts
(a Concept inside a Concept), a Concept nested under a *non*-Concept
organizational dir, an ``_ops/`` sidecar (which — with its descendants — must
be skipped), and a loose non-dir file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.knowledge import ConceptMeta


def _concept(path: Path, concept_type: str = "folder") -> None:
    """Materialize a Concept dir at *path* (a dir holding ``meta.yaml``)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.yaml").write_text(ConceptMeta(type=concept_type).to_yaml(), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Build an OKF bundle and return its root dir.

    Layout (Concepts marked ``*``)::

        bundle/
        ├── alpha/        *   concept
        │   └── beta/     *   concept nested under a concept
        ├── delta/        *   concept
        │   └── _ops/         sidecar — skipped, with all descendants
        │       ├── state.json
        │       └── nested_fake/  (has meta.yaml but lives under _ops → skipped)
        ├── group/            non-concept organizational dir (no meta.yaml)
        │   └── gamma/    *   concept nested under a non-concept dir
        └── loose.txt         loose file (not a dir)

    Expected Concept rel-paths: ``alpha``, ``alpha/beta``, ``delta``,
    ``group/gamma`` — depth-first preorder over sorted entries.
    """
    root = tmp_path / "bundle"
    root.mkdir()

    _concept(root / "alpha")
    _concept(root / "alpha" / "beta")

    _concept(root / "delta")
    ops = root / "delta" / "_ops"
    ops.mkdir()
    (ops / "state.json").write_text("{}", encoding="utf-8")
    _concept(ops / "nested_fake")  # meta.yaml present but under _ops → never walked

    (root / "group").mkdir()  # organizational, no meta.yaml
    _concept(root / "group" / "gamma")

    (root / "loose.txt").write_text("x", encoding="utf-8")

    return root
