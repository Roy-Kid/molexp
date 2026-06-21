"""Tests for :class:`molexp.knowledge.Library` — the OKF bundle façade.

``Library`` wraps a bundle root and exposes the whole Concept-directory tree
as a single management entry point: ``walk`` (depth-first Concept
enumeration), ``get`` (path-as-identity resolution), ``put`` (idempotent
materialization) and ``link`` (a semantic edge written as a markdown link
into ``index.md``, so it round-trips through ``Folder.out_edges()``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.knowledge import ConceptNotFoundError, Folder, Library

# ── public surface (ac-001) ──────────────────────────────────────────────────


def test_library_is_exported_from_package() -> None:
    import molexp.knowledge as knowledge

    assert "Library" in knowledge.__all__
    assert knowledge.Library is Library


# ── walk() depth-first concept enumeration (ac-002 / ac-003) ──────────────────


def test_walk_yields_exactly_meta_bearing_dirs(bundle: Path) -> None:
    lib = Library(bundle)
    rels = {lib.rel_path(f) for f in lib.walk()}
    assert rels == {"alpha", "alpha/beta", "delta", "group/gamma"}


def test_walk_order_is_depth_first_preorder(bundle: Path) -> None:
    lib = Library(bundle)
    rels = [lib.rel_path(f) for f in lib.walk()]
    assert rels == ["alpha", "alpha/beta", "delta", "group/gamma"]


def test_walk_excludes_ops_sidecar_and_non_concept_dirs(bundle: Path) -> None:
    lib = Library(bundle)
    rels = {lib.rel_path(f) for f in lib.walk()}
    # _ops sidecar and its descendants are never walked …
    assert not any(r.startswith("delta/_ops") for r in rels)
    # … a meta.yaml planted under _ops does not resurrect it
    assert "delta/_ops/nested_fake" not in rels
    # … a non-concept organizational dir is not itself a concept …
    assert "group" not in rels
    # … but a concept nested *under* it still surfaces.
    assert "group/gamma" in rels


def test_walk_yields_folder_instances(bundle: Path) -> None:
    lib = Library(bundle)
    walked = list(lib.walk())
    assert walked  # non-empty
    assert all(isinstance(f, Folder) for f in walked)
    # each yielded Folder resolves to a real concept dir
    assert all((Path(f.resolve()) / "meta.yaml").is_file() for f in walked)


# ── get() path-as-identity (ac-004 / ac-005) ─────────────────────────────────


def test_get_resolves_known_concept_to_folder(bundle: Path) -> None:
    lib = Library(bundle)
    f = lib.get("alpha/beta")
    assert isinstance(f, Folder)
    assert Path(f.resolve()) == bundle / "alpha" / "beta"
    assert lib.rel_path(f) == "alpha/beta"


def test_get_resolves_top_level_concept(bundle: Path) -> None:
    lib = Library(bundle)
    f = lib.get("delta")
    assert Path(f.resolve()) == bundle / "delta"


def test_get_unknown_path_raises_concept_not_found(bundle: Path) -> None:
    lib = Library(bundle)
    with pytest.raises(ConceptNotFoundError):
        lib.get("does/not/exist")


def test_get_non_concept_dir_raises_concept_not_found(bundle: Path) -> None:
    lib = Library(bundle)
    # "group" exists on disk but has no meta.yaml → not a Concept
    with pytest.raises(ConceptNotFoundError):
        lib.get("group")


# ── put() idempotent materialization (ac-006) ────────────────────────────────


def test_put_materializes_concept(bundle: Path) -> None:
    lib = Library(bundle)
    epsilon = Folder(name="epsilon", root=bundle, concept_type="run")
    assert not (Path(epsilon.resolve()) / "meta.yaml").is_file()
    lib.put(epsilon)
    assert (Path(epsilon.resolve()) / "meta.yaml").is_file()
    # type preserved on materialization
    assert lib.get("epsilon").read_meta().type == "run"


def test_put_is_idempotent(bundle: Path) -> None:
    lib = Library(bundle)
    epsilon = Folder(name="epsilon", root=bundle)
    lib.put(epsilon)
    lib.put(epsilon)  # second put must not raise
    rels = [lib.rel_path(f) for f in lib.walk()]
    assert rels.count("epsilon") == 1


# ── link() markdown-graph round-trip (ac-007) ────────────────────────────────


def test_link_round_trips_through_out_edges(bundle: Path) -> None:
    lib = Library(bundle)
    src = lib.get("alpha")
    dst = lib.get("delta")

    lib.link(src, dst)

    # the edge lives in markdown, not yaml
    index_text = src.read_index()
    assert "delta" in index_text
    assert "](" in index_text  # a real markdown link was written
    meta_text = (Path(src.resolve()) / "meta.yaml").read_text(encoding="utf-8")
    assert "delta" not in meta_text  # not smuggled into structured metadata

    # round-trip: Folder.out_edges() resolves the markdown link back to dst
    edges = {Path(p) for p in lib.get("alpha").out_edges()}
    assert Path(dst.resolve()) in edges


def test_link_to_nested_concept_round_trips(bundle: Path) -> None:
    lib = Library(bundle)
    src = lib.get("delta")
    dst = lib.get("alpha/beta")

    lib.link(src, dst)

    edges = {Path(p) for p in lib.get("delta").out_edges()}
    assert Path(dst.resolve()) in edges


# ── typed walk / get via the registry (okf-02) ───────────────────────────────


def test_walk_yields_typed_concepts(tmp_path: Path) -> None:
    from molexp.knowledge import Experiment, Project, Run, Workspace

    ws = Workspace(name="lab", root=tmp_path)
    ws.add_project("p").add_experiment("e").add_run("r")

    lib = Library(tmp_path)  # bundle root sits above the workspace concept
    by_rel = {lib.rel_path(f): f for f in lib.walk()}

    assert isinstance(by_rel["lab"], Workspace)
    assert isinstance(by_rel["lab/p"], Project)
    assert isinstance(by_rel["lab/p/e"], Experiment)
    assert isinstance(by_rel["lab/p/e/r"], Run)


def test_get_returns_typed_concept(tmp_path: Path) -> None:
    from molexp.knowledge import Project, Workspace

    ws = Workspace(name="lab", root=tmp_path)
    ws.add_project("p")

    lib = Library(tmp_path)
    assert isinstance(lib.get("lab"), Workspace)
    assert isinstance(lib.get("lab/p"), Project)
