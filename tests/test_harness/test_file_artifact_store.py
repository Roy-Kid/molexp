"""Tests for FileArtifactStore (Phase 1 artifact persistence).

Locks the contract per spec §FileArtifactStore:
- put_json / put_text / put_file write content + ref + kind-index via
  workspace.atomic_write_json / atomic_write_text
- ArtifactRef.sha256 == compute_content_hash(path).removeprefix("sha256:")
- idempotent: identical content under the same kind returns the same ref
- list_by_kind returns refs in creation order
- latest_by_kind returns most recent or None
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from molexp.workspace.utils import compute_content_hash


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    """Per-test isolated artifact root."""
    root = tmp_path / "artifacts"
    return root


@pytest.fixture()
def store(store_root: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=store_root)


def _list_relative(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_put_json_creates_content_ref_and_index_files(store, store_root: Path) -> None:
    ref = store.put_json(
        kind="workflow_ir",
        obj={"name": "demo", "version": 1},
        created_by="harness",
        parent_ids=[],
    )
    files = _list_relative(store_root)
    # Content file under artifacts/<kind>/<id>.json
    assert f"workflow_ir/{ref.id}.json" in files
    # Ref under artifacts/_refs/<id>.json
    assert f"_refs/{ref.id}.json" in files
    # Kind index under artifacts/_index/<kind>.json
    assert "_index/workflow_ir.json" in files


def test_put_json_sha256_matches_workspace_compute(store, store_root: Path) -> None:
    ref = store.put_json(
        kind="workflow_ir",
        obj={"name": "demo"},
        created_by="harness",
        parent_ids=[],
    )
    content_path = store_root / "workflow_ir" / f"{ref.id}.json"
    expected = compute_content_hash(content_path).removeprefix("sha256:")
    assert ref.sha256 == expected
    # Sanity: bare hex (no prefix)
    assert ":" not in ref.sha256


def test_put_text_sha256_matches_workspace_compute(store, store_root: Path) -> None:
    ref = store.put_text(
        kind="log",
        text="hello harness\n",
        created_by="harness",
        parent_ids=[],
    )
    content_path = store_root / "log" / f"{ref.id}.txt"
    expected = compute_content_hash(content_path).removeprefix("sha256:")
    assert ref.sha256 == expected


def test_put_file_sha256_matches_workspace_compute(store, store_root: Path, tmp_path: Path) -> None:
    src = tmp_path / "trajectory.dcd"
    src.write_bytes(b"FAKE-DCD-PAYLOAD" * 100)
    ref = store.put_file(
        kind="output_file",
        path=src,
        created_by="harness",
        parent_ids=[],
    )
    # Stored under output_file/<id>-<original_name>; locate via the ref URI.
    stored = Path(ref.uri.removeprefix("file://"))
    assert stored.exists()
    expected = compute_content_hash(stored).removeprefix("sha256:")
    assert ref.sha256 == expected


def test_put_json_is_idempotent_on_identical_content(store) -> None:
    obj = {"name": "demo", "version": 1}
    ref1 = store.put_json(kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[])
    ref2 = store.put_json(kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[])
    assert ref1.id == ref2.id
    assert ref1.sha256 == ref2.sha256
    # list_by_kind should not duplicate.
    refs = store.list_by_kind("workflow_ir")
    assert len(refs) == 1


def test_put_text_is_idempotent_on_identical_content(store) -> None:
    ref1 = store.put_text(kind="log", text="abc\n", created_by="harness", parent_ids=[])
    ref2 = store.put_text(kind="log", text="abc\n", created_by="harness", parent_ids=[])
    assert ref1.id == ref2.id


def test_same_content_different_kinds_yields_distinct_ids(store) -> None:
    """Identical bytes under two kinds MUST yield two distinct ids.

    Regression: with a sha-only id, the second ``put_*`` overwrote the
    first ref's metadata (kind, parent_ids). Now we hash ``kind:sha`` to
    keep them apart and preserve the audit trail per kind.
    """
    text = "shared payload"
    a = store.put_text(kind="log", text=text, created_by="x", parent_ids=[])
    b = store.put_text(kind="stdout", text=text, created_by="x", parent_ids=[])
    assert a.id != b.id
    assert a.kind == "log"
    assert b.kind == "stdout"
    # Both still resolve to their original content.
    assert store.get(a.id) == text.encode()
    assert store.get(b.id) == text.encode()
    # And both appear under their respective kind indexes.
    assert any(r.id == a.id for r in store.list_by_kind("log"))
    assert any(r.id == b.id for r in store.list_by_kind("stdout"))


def test_idempotent_hit_merges_new_parent_ids(store) -> None:
    """Re-deriving an artifact via a new parent path adds to its parent_ids.

    Without this, calling ``put_*`` a second time with the same content
    under the same kind silently dropped the new parent_ids — provenance
    edges from the alternate derivation path would be missing.
    """
    obj = {"v": 1}
    parent_a = store.put_text(kind="log", text="A", created_by="x", parent_ids=[])
    parent_b = store.put_text(kind="log", text="B", created_by="x", parent_ids=[])

    first = store.put_json(
        kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[parent_a.id]
    )
    second = store.put_json(
        kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[parent_b.id]
    )
    assert first.id == second.id  # idempotent on content
    # The second call's parent_ids MUST appear in the returned ref AND on disk.
    assert parent_a.id in second.parent_ids
    assert parent_b.id in second.parent_ids
    reloaded = store.get_ref(first.id)
    assert parent_a.id in reloaded.parent_ids
    assert parent_b.id in reloaded.parent_ids


def test_get_and_get_ref_roundtrip(store) -> None:
    ref = store.put_json(
        kind="workflow_ir",
        obj={"k": "v"},
        created_by="harness",
        parent_ids=[],
    )
    raw = store.get(ref.id)
    assert json.loads(raw) == {"k": "v"}
    same_ref = store.get_ref(ref.id)
    assert same_ref == ref


def test_get_missing_raises_artifact_not_found(store) -> None:
    from molexp.harness.errors import ArtifactNotFoundError

    with pytest.raises(ArtifactNotFoundError):
        store.get("does-not-exist")


def test_list_by_kind_returns_creation_order(store) -> None:
    a = store.put_json(kind="log", obj={"i": 0}, created_by="harness", parent_ids=[])
    b = store.put_json(kind="log", obj={"i": 1}, created_by="harness", parent_ids=[])
    c = store.put_json(kind="log", obj={"i": 2}, created_by="harness", parent_ids=[])
    refs = store.list_by_kind("log")
    assert [r.id for r in refs] == [a.id, b.id, c.id]


def test_list_by_kind_isolates_kinds(store) -> None:
    store.put_json(kind="log", obj={"i": 0}, created_by="harness", parent_ids=[])
    store.put_json(kind="workflow_ir", obj={"i": 0}, created_by="harness", parent_ids=[])
    assert len(store.list_by_kind("log")) == 1
    assert len(store.list_by_kind("workflow_ir")) == 1


def test_put_json_round_trips_unknown_kind(store, store_root: Path) -> None:
    """Custom (non-well-known) artifact kinds round-trip end-to-end.

    Spec ac-005: agent-layer modes register their own kinds (``intent_spec``,
    ``plan_graph``, …) under the open ``ArtifactKind = str`` contract; the
    store must accept them without schema migration. Regression coverage —
    asserts the per-kind index path is purely string-keyed.
    """
    ref = store.put_json(
        kind="intent_spec",
        obj={"k": 1},
        created_by="t",
        parent_ids=[],
    )
    assert ref.kind == "intent_spec"

    files = _list_relative(store_root)
    assert f"intent_spec/{ref.id}.json" in files
    assert "_index/intent_spec.json" in files

    reloaded = store.get_ref(ref.id)
    assert reloaded.kind == "intent_spec"
    assert reloaded == ref


def test_latest_by_kind_returns_most_recent_or_none(store) -> None:
    assert store.latest_by_kind("log") is None
    a = store.put_json(kind="log", obj={"i": 0}, created_by="harness", parent_ids=[])
    b = store.put_json(kind="log", obj={"i": 1}, created_by="harness", parent_ids=[])
    assert store.latest_by_kind("log") == b
    assert store.latest_by_kind("workflow_ir") is None
    _ = a  # silence


def test_file_artifact_store_uses_workspace_atomic_helpers() -> None:
    """Spec ac-003: FileArtifactStore source imports atomic_write_json
    and atomic_write_text from molexp.workspace."""
    from molexp.harness.store import file_artifact_store as mod

    src = inspect.getsource(mod)
    assert "atomic_write_json" in src
    assert "atomic_write_text" in src
    assert "from molexp.workspace" in src
