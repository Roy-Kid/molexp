"""Tests for :class:`molexp.workspace.Bundle` — the OKF bundle façade.

``Bundle`` wraps a bundle root and exposes the whole Concept-directory tree as
a single management entry point: ``walk`` (depth-first Concept enumeration),
``get`` (path-as-identity resolution), ``put`` (idempotent materialization) and
``link`` (a semantic edge written as a markdown link into ``index.md``, so it
round-trips through :meth:`Folder.out_edges`).

It is the OKF Concept-bundle surface for notes + literature, reached via
path-as-identity resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Bundle, ConceptNotFoundError, Folder

# A concept ``type`` deliberately NOT in the concept-type registry, so it
# reconstructs as the base workspace ``Folder`` (vs. a knowledge subclass).
CONCEPT_KIND = "bundle.concept"


def _concept(name: str, root_path: Path) -> Folder:
    """Materialize a generic Concept dir (metadata.json + meta.yaml) on disk."""
    folder = Folder(name=name, kind=CONCEPT_KIND, root_path=str(root_path))
    folder.materialize()  # metadata.json — base Folder.from_disk reads it
    folder.write_meta()  # meta.yaml — the OKF Concept marker
    return folder


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Build an OKF bundle on disk and return its root.

    Layout (Concept dirs hold ``meta.yaml``)::

        <root>/alpha/            (concept)
        <root>/alpha/beta/       (concept, nested)
        <root>/delta/            (concept)
        <root>/delta/_ops/       (sidecar — never a concept)
        <root>/delta/_ops/nested_fake/meta.yaml   (planted; must be skipped)
        <root>/group/            (plain org dir — NOT a concept)
        <root>/group/gamma/      (concept, under a non-concept dir)
        <root>/loose.txt         (loose file — never a concept)
    """
    root = tmp_path / "bundle"
    root.mkdir()

    _concept("alpha", root)
    _concept("beta", root / "alpha")
    _concept("delta", root)

    # _ops sidecar with a planted meta.yaml that must never resurrect a concept
    ops_fake = root / "delta" / "_ops" / "nested_fake"
    ops_fake.mkdir(parents=True)
    (ops_fake / "meta.yaml").write_text("type: bundle.concept\nid: nested_fake\n")

    # plain organizational dir (no meta.yaml) with a concept nested beneath it
    (root / "group").mkdir()
    _concept("gamma", root / "group")

    (root / "loose.txt").write_text("not a concept\n")
    return root


# ── public surface (ac-001) ──────────────────────────────────────────────────


def test_exported_and_distinct() -> None:
    import molexp.workspace as workspace

    assert "Bundle" in workspace.__all__
    assert workspace.Bundle is Bundle
    # The legacy per-scope Library surface is gone (wsokf-11).
    assert not hasattr(workspace, "Library")


# ── walk() depth-first concept enumeration (ac-002 / ac-003) ──────────────────


def test_walk_yields_exactly_meta_bearing_dirs(bundle: Path) -> None:
    b = Bundle(bundle)
    rels = {b.rel_path(f) for f in b.walk()}
    assert rels == {"alpha", "alpha/beta", "delta", "group/gamma"}


def test_walk_order_is_depth_first_preorder(bundle: Path) -> None:
    b = Bundle(bundle)
    rels = [b.rel_path(f) for f in b.walk()]
    assert rels == ["alpha", "alpha/beta", "delta", "group/gamma"]


def test_walk_skips_ops_or_nonconcept(bundle: Path) -> None:
    b = Bundle(bundle)
    rels = {b.rel_path(f) for f in b.walk()}
    # the _ops sidecar and its descendants are never walked …
    assert not any(r.startswith("delta/_ops") for r in rels)
    # … a meta.yaml planted under _ops does not resurrect it
    assert "delta/_ops/nested_fake" not in rels
    # … a non-concept organizational dir is not itself a concept …
    assert "group" not in rels
    # … but a concept nested under it still surfaces …
    assert "group/gamma" in rels
    # … and loose files are never concepts.
    assert "loose.txt" not in rels


def test_walk_yields_folder_instances(bundle: Path) -> None:
    b = Bundle(bundle)
    walked = list(b.walk())
    assert walked  # non-empty
    assert all(isinstance(f, Folder) for f in walked)
    assert all((Path(f.resolve()) / "meta.yaml").is_file() for f in walked)


# ── get() path-as-identity (ac-004) ──────────────────────────────────────────


def test_get_resolves_known_concept_to_folder(bundle: Path) -> None:
    b = Bundle(bundle)
    f = b.get("alpha/beta")
    assert isinstance(f, Folder)
    assert Path(f.resolve()) == bundle / "alpha" / "beta"
    assert b.rel_path(f) == "alpha/beta"


def test_get_resolves_top_level_concept(bundle: Path) -> None:
    b = Bundle(bundle)
    f = b.get("delta")
    assert Path(f.resolve()) == bundle / "delta"


def test_get_unknown_path_raises_concept_not_found(bundle: Path) -> None:
    b = Bundle(bundle)
    with pytest.raises(ConceptNotFoundError):
        b.get("does/not/exist")


def test_get_non_concept_dir_raises_concept_not_found(bundle: Path) -> None:
    b = Bundle(bundle)
    # "group" exists on disk but has no meta.yaml → not a Concept
    with pytest.raises(ConceptNotFoundError):
        b.get("group")


# ── put() idempotent materialization (ac-005) ────────────────────────────────


def test_put_materializes_concept(bundle: Path) -> None:
    b = Bundle(bundle)
    epsilon = Folder(name="epsilon", kind=CONCEPT_KIND, root_path=str(bundle))
    epsilon.materialize()  # write base metadata.json so it can reconstruct
    assert not (Path(epsilon.resolve()) / "meta.yaml").is_file()
    b.put(epsilon)
    assert (Path(epsilon.resolve()) / "meta.yaml").is_file()
    # type preserved on materialization
    assert b.get("epsilon").read_meta()["type"] == CONCEPT_KIND


def test_put_is_idempotent(bundle: Path) -> None:
    b = Bundle(bundle)
    epsilon = Folder(name="epsilon", kind=CONCEPT_KIND, root_path=str(bundle))
    epsilon.materialize()
    b.put(epsilon)
    b.put(epsilon)  # second put must not raise nor duplicate
    rels = [b.rel_path(f) for f in b.walk()]
    assert rels.count("epsilon") == 1


# ── link() markdown-graph round-trip (ac-006) ────────────────────────────────


def test_link_round_trips_through_out_edges(bundle: Path) -> None:
    b = Bundle(bundle)
    src = b.get("alpha")
    dst = b.get("delta")

    b.link(src, dst)

    # the edge lives in markdown, not yaml
    index_text = src.read_index()
    assert "delta" in index_text
    assert "](" in index_text  # a real markdown link was written
    meta_text = (Path(src.resolve()) / "meta.yaml").read_text(encoding="utf-8")
    assert "delta" not in meta_text  # not smuggled into structured metadata

    # round-trip: Folder.out_edges() resolves the markdown link back to dst
    edges = {Path(p) for p in b.get("alpha").out_edges()}
    assert Path(dst.resolve()) in edges


def test_link_to_nested_concept_round_trips(bundle: Path) -> None:
    b = Bundle(bundle)
    src = b.get("delta")
    dst = b.get("alpha/beta")

    b.link(src, dst)

    edges = {Path(p) for p in b.get("delta").out_edges()}
    assert Path(dst.resolve()) in edges


# ── typed walk / get via the registry (ac-007) ───────────────────────────────


def test_walk_typed_concepts(tmp_path: Path) -> None:
    from molexp.workspace import Experiment, Project, Run, Workspace

    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    ws.add_project("p").add_experiment("e").add_run(id="r")

    # the bundle root sits ABOVE the workspace concept dir
    b = Bundle(tmp_path)
    by_rel = {b.rel_path(f): f for f in b.walk()}

    assert isinstance(by_rel["lab"], Workspace)
    assert isinstance(by_rel["lab/projects/p"], Project)
    assert isinstance(by_rel["lab/projects/p/experiments/e"], Experiment)
    assert isinstance(
        by_rel["lab/projects/p/experiments/e/runs/run-r"],
        Run,
    )


def test_get_typed_concept(tmp_path: Path) -> None:
    from molexp.workspace import Project, Workspace

    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    ws.add_project("p")

    b = Bundle(tmp_path)
    assert isinstance(b.get("lab"), Workspace)
    assert isinstance(b.get("lab/projects/p"), Project)
