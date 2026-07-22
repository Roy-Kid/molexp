"""Tests for ``FileArtifactStore`` (harness artifact persistence).

Mirrors ``molexp.harness.store.file_artifact_store``. Locks the contract:
- ``put_json`` / ``put_text`` / ``put_file`` write content + ref + kind-index;
- ``PlanArtifactRef.sha256`` == ``compute_content_hash(path)`` bare hex;
- idempotent per ``(kind, content)`` — identical bytes under one kind reuse the
  ref, identical bytes under two kinds stay distinct, and an idempotent hit
  unions new ``parent_ids`` (both regressions of a sha-only id overwrite);
- ``get`` / ``get_ref`` / ``list_by_kind`` (creation order) / ``latest_by_kind``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace.utils import compute_content_hash


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    """Per-test isolated artifact root."""
    return tmp_path / "artifacts"


@pytest.fixture()
def store(store_root: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=store_root)


def _list_relative(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


class TestFileArtifactStore:
    def test_put_json_writes_content_ref_and_kind_index(self, store, store_root: Path) -> None:
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

    def test_put_json_sha256_matches_workspace_compute(self, store, store_root: Path) -> None:
        ref = store.put_json(
            kind="workflow_ir",
            obj={"name": "demo"},
            created_by="harness",
            parent_ids=[],
        )
        content_path = store_root / "workflow_ir" / f"{ref.id}.json"
        expected = compute_content_hash(content_path).removeprefix("sha256:")
        assert ref.sha256 == expected
        # Bare hex, no prefix.
        assert ":" not in ref.sha256

    def test_put_file_stores_under_original_name_with_matching_sha256(
        self, store, tmp_path: Path
    ) -> None:
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

    def test_put_json_is_idempotent_on_identical_content(self, store) -> None:
        obj = {"name": "demo", "version": 1}
        ref1 = store.put_json(kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[])
        ref2 = store.put_json(kind="workflow_ir", obj=obj, created_by="harness", parent_ids=[])
        assert ref1.id == ref2.id
        assert ref1.sha256 == ref2.sha256
        # list_by_kind should not duplicate.
        refs = store.list_by_kind("workflow_ir")
        assert len(refs) == 1

    def test_same_content_different_kinds_yields_distinct_ids(self, store) -> None:
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

    def test_idempotent_hit_merges_new_parent_ids(self, store) -> None:
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

    def test_get_and_get_ref_roundtrip(self, store) -> None:
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

    def test_get_missing_raises_artifact_not_found(self, store) -> None:
        from molexp.harness.errors import ArtifactNotFoundError

        with pytest.raises(ArtifactNotFoundError):
            store.get("does-not-exist")

    def test_list_by_kind_returns_creation_order(self, store) -> None:
        a = store.put_json(kind="log", obj={"i": 0}, created_by="harness", parent_ids=[])
        b = store.put_json(kind="log", obj={"i": 1}, created_by="harness", parent_ids=[])
        c = store.put_json(kind="log", obj={"i": 2}, created_by="harness", parent_ids=[])
        refs = store.list_by_kind("log")
        assert [r.id for r in refs] == [a.id, b.id, c.id]

    def test_latest_by_kind_returns_most_recent_or_none(self, store) -> None:
        assert store.latest_by_kind("log") is None
        store.put_json(kind="log", obj={"i": 0}, created_by="harness", parent_ids=[])
        b = store.put_json(kind="log", obj={"i": 1}, created_by="harness", parent_ids=[])
        assert store.latest_by_kind("log") == b
        assert store.latest_by_kind("workflow_ir") is None
