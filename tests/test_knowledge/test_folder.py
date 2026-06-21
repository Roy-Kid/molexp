"""Tests for :class:`molexp.knowledge.Folder` — the OKF Concept-on-disk base.

A Concept is a directory whose path is its identity, physically split into
``meta.yaml`` (structured) + ``index.md`` (narrative + markdown-link graph)
+ optional ``log.md``, with hot machine state isolated under ``_ops/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molexp.atomicio as atomicio
from molexp.knowledge import ConceptMeta, ConceptNotFoundError, Folder

# ── construction + path lifecycle (ac-001 / ac-002) ──────────────────────────


def test_init_performs_zero_filesystem_writes(tmp_path: Path) -> None:
    Folder(name="alpha", root=tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing written until path()/write_*


def test_resolve_is_side_effect_free_path_lazily_mkdirs(tmp_path: Path) -> None:
    f = Folder(name="alpha", root=tmp_path)
    resolved = f.resolve()
    assert not Path(resolved).exists()  # resolve() creates nothing
    created = f.path()
    assert Path(created).is_dir()  # path() mkdirs
    f.path()  # idempotent — no error on second call
    assert Path(created).is_dir()


def test_resolve_appends_name_under_root(tmp_path: Path) -> None:
    f = Folder(name="My Concept", root=tmp_path)
    # identity = slugified name
    assert Path(f.resolve()) == tmp_path / "my-concept"


# ── meta.yaml / index.md / log.md (ac-003 / ac-004 / ac-005 / ac-012) ────────


def test_meta_round_trips_via_atomicio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = Folder(name="alpha", root=tmp_path)
    calls: list[str] = []
    real = atomicio.atomic_write_text

    def spy(path: Path, content: str, **kw: object) -> None:
        calls.append(str(path))
        real(path, content, **kw)

    monkeypatch.setattr(atomicio, "atomic_write_text", spy)

    meta = ConceptMeta(type="run", id="r1")
    f.write_meta(meta)
    assert f.read_meta() == meta
    # on-disk meta.yaml is pure YAML
    import yaml

    raw = yaml.safe_load((Path(f.resolve()) / "meta.yaml").read_text())
    assert isinstance(raw, dict)
    assert raw["type"] == "run"
    # write routed through atomicio.atomic_write_text
    assert any("meta.yaml" in c for c in calls)


def test_index_and_log_round_trip_log_is_timestamped_ordered(tmp_path: Path) -> None:
    f = Folder(name="alpha", root=tmp_path)
    f.write_index("# Alpha\n\nbody\n")
    assert f.read_index() == "# Alpha\n\nbody\n"

    f.append_log("created")
    f.append_log("ran")
    log = f.read_log()
    i_created = log.index("created")
    i_ran = log.index("ran")
    assert 0 <= i_created < i_ran  # order preserved
    # each entry carries an ISO-ish timestamp (year present)
    assert log.count("2026") + log.count("202") >= 2 or ":" in log


def test_physical_split_meta_and_index_are_separate_files(tmp_path: Path) -> None:
    f = Folder(name="alpha", root=tmp_path)
    f.write_meta(ConceptMeta(type="run"))
    f.write_index("# Alpha\n\nNarrative, no frontmatter.\n")
    d = Path(f.resolve())
    assert (d / "meta.yaml").is_file()
    assert (d / "index.md").is_file()
    index_text = (d / "index.md").read_text()
    assert not index_text.lstrip().startswith("---")  # no YAML frontmatter
    import yaml

    assert isinstance(yaml.safe_load((d / "meta.yaml").read_text()), dict)


def test_all_concept_writes_route_through_atomicio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = Folder(name="alpha", root=tmp_path)
    text_calls: list[str] = []
    json_calls: list[str] = []
    real_text = atomicio.atomic_write_text
    real_json = atomicio.atomic_write_json

    def spy_text(path: Path, content: str, **kw: object) -> None:
        text_calls.append(str(path))
        real_text(path, content, **kw)

    def spy_json(path: Path, data: object) -> None:
        json_calls.append(str(path))
        real_json(path, data)

    monkeypatch.setattr(atomicio, "atomic_write_text", spy_text)
    monkeypatch.setattr(atomicio, "atomic_write_json", spy_json)

    f.write_meta(ConceptMeta(type="run"))
    f.write_index("body")
    f.append_log("entry")
    f.write_ops_json("state", {"k": 1})

    assert any("meta.yaml" in c for c in text_calls)
    assert any("index.md" in c for c in text_calls)
    assert any("log.md" in c for c in text_calls)
    assert any("state.json" in c for c in json_calls)


# ── markdown-link graph (ac-006) ─────────────────────────────────────────────


def test_out_edges_returns_only_in_bundle_concept_targets(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    # two child concepts (have meta.yaml)
    child_a = root.add_folder("child-a")
    child_b = root.add_folder("child-b")
    # a non-concept dir (no meta.yaml)
    (Path(root.resolve()) / "plain").mkdir()

    root.write_index(
        "# Bundle\n\n"
        "- [A](child-a/index.md)\n"
        "- [B](child-b)\n"
        "- [ext](https://example.com)\n"
        "- [plain](plain)\n"
    )

    edges = {Path(p) for p in root.out_edges()}
    assert edges == {Path(child_a.resolve()), Path(child_b.resolve())}

    scan = root.links()
    assert any("example.com" in e for e in scan.external)
    assert any("plain" in o for o in scan.other)


# ── five-verb CRUD (ac-007 / ac-008 / ac-009) ────────────────────────────────


def test_add_folder_idempotent_on_slug(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    first = root.add_folder("My Child")
    second = root.add_folder("My Child")
    third = root.add_folder("my-child")  # slug of the same name
    assert Path(first.resolve()) == Path(second.resolve()) == Path(third.resolve())
    # exactly one child dir created
    children = [p for p in Path(root.resolve()).iterdir() if (p / "meta.yaml").is_file()]
    assert len(children) == 1


def test_get_has_honor_name_and_slug(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("My Child")
    assert root.has_folder("My Child")
    assert root.has_folder("my-child")
    assert Path(root.get_folder("My Child").resolve()) == Path(
        root.get_folder("my-child").resolve()
    )


def test_child_is_concept_iff_meta_yaml_present(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("real")  # has meta.yaml
    (Path(root.resolve()) / "fake").mkdir()  # no meta.yaml
    names = {f.name for f in root.list_folders()}
    assert "real" in names
    assert "fake" not in names


def test_remove_folder_deletes_and_missing_raises(tmp_path: Path) -> None:
    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("doomed")
    root.remove_folder("doomed")
    assert not (Path(root.resolve()) / "doomed").exists()
    with pytest.raises(ConceptNotFoundError):
        root.get_folder("doomed")
    with pytest.raises(ConceptNotFoundError):
        root.remove_folder("never-existed")


# ── _ops sidecar isolation (ac-010 / ac-011) ─────────────────────────────────


def test_ops_write_isolated_from_meta(tmp_path: Path) -> None:
    f = Folder(name="alpha", root=tmp_path)
    f.write_meta(ConceptMeta(type="run", id="r1"))
    before = (Path(f.resolve()) / "meta.yaml").read_text()

    f.write_ops_json("state", {"status": "running", "pid": 1234})
    assert (Path(f.resolve()) / "_ops" / "state.json").is_file()

    after = (Path(f.resolve()) / "meta.yaml").read_text()
    assert before == after  # hot state never touched meta.yaml
    assert f.read_ops_json("state") == {"status": "running", "pid": 1234}


def test_update_ops_json_uses_file_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = Folder(name="alpha", root=tmp_path)
    f.write_ops_json("state", {"n": 1})

    used = {"lock": False}
    real_lock = atomicio.file_lock

    def spy_lock(path: Path, **kw: object):
        used["lock"] = True
        return real_lock(path, **kw)

    monkeypatch.setattr(atomicio, "file_lock", spy_lock)

    result = f.update_ops_json("state", lambda cur: {**cur, "n": cur["n"] + 1})
    assert result["n"] == 2
    assert f.read_ops_json("state")["n"] == 2
    assert used["lock"] is True


def test_ops_dir_lazy_mkdir(tmp_path: Path) -> None:
    f = Folder(name="alpha", root=tmp_path)
    d = f.ops_dir()
    assert Path(d).is_dir()
    assert Path(d).name == "_ops"


# ── type-aware reconstruction via the registry (okf-02) ───────────────────────


def test_get_folder_reconstructs_registered_subclass(tmp_path: Path) -> None:
    from molexp.knowledge import Project

    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("proj", concept_type="project")
    # a fresh, uncached Folder must reconstruct the typed subclass from disk
    fresh = Folder(name="bundle", root=tmp_path)
    child = fresh.get_folder("proj")
    assert isinstance(child, Project)
    assert child.read_meta().type == "project"


def test_list_folders_reconstructs_types_unknown_falls_back(tmp_path: Path) -> None:
    from molexp.knowledge import Project

    root = Folder(name="bundle", root=tmp_path)
    root.add_folder("proj", concept_type="project")
    root.add_folder("weird", concept_type="totally-unknown-type")

    fresh = Folder(name="bundle", root=tmp_path)
    by_name = {f.name: f for f in fresh.list_folders()}
    assert isinstance(by_name["proj"], Project)
    # unknown type → base Folder (forward-compatible), not a subclass
    assert type(by_name["weird"]) is Folder


def test_add_folder_returns_typed_instance(tmp_path: Path) -> None:
    from molexp.knowledge import Experiment

    root = Folder(name="bundle", root=tmp_path)
    child = root.add_folder("exp", concept_type="experiment")
    assert isinstance(child, Experiment)
