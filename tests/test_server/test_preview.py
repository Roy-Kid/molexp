"""Tests for sidecar-backed dataset preview.

Covers the spec ``dataset-preview-recipes`` acceptance criteria:

* ac-001/ac-002 — the trust gate: discovery never imports the sidecar, and
  the explicit import never runs its ``if __name__ == "__main__"`` block.
* ac-003 — ``load_sidecar_reader`` requires exactly one
  ``BaseTrajectoryReader`` subclass (zero / two / broken are typed errors).
* ac-004 — missing / zero / two / broken sidecars surface as typed 4xx,
  never a 500.
* ac-005 — the preview route serves extxyz frame bytes (and a PNG snapshot
  when molvis is installed).
* ac-006 — the host owns the frame cap via ``islice`` on the reader.
* ac-007 — the asset listing carries a sidecar-existence boolean.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

# The synthetic fixture and the QM9-shaped readers all build molpy.Frames.
molpy = pytest.importorskip("molpy")

from molexp.server.preview import (  # noqa: E402
    DEFAULT_PREVIEW_LIMIT,
    AmbiguousReaderError,
    NoReaderInSidecarError,
    PreviewReaderError,
    PreviewSidecarNotFoundError,
    load_sidecar_reader,
    preview_frames,
    resolve_sidecar,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "fake_sidecar.py"

# A sidecar that defines no BaseTrajectoryReader subclass.
_ZERO_READER_SRC = "VALUE = 42\n"

# A sidecar that defines two concrete BaseTrajectoryReader subclasses.
_TWO_READER_SRC = """
import molpy
from molpy.io import BaseTrajectoryReader


class ReaderA(BaseTrajectoryReader):
    def read_frame(self, i):
        return molpy.Frame()

    @property
    def n_frames(self):
        return 1


class ReaderB(BaseTrajectoryReader):
    def read_frame(self, i):
        return molpy.Frame()

    @property
    def n_frames(self):
        return 1
"""

# A sidecar that raises while its module body executes.
_BROKEN_SRC = "raise RuntimeError('sidecar boom')\n"


def _make_sidecar_dataset(
    dir_path: Path,
    *,
    stem: str = "qm9",
    ext: str = ".bin",
    sidecar_src: str | None = None,
) -> Path:
    """Create ``<stem><ext>`` plus its same-stem ``<stem>.py`` sidecar.

    ``sidecar_src=None`` copies the committed ``fake_sidecar.py`` (the
    happy-path reader); pass a string to install a variant sidecar.
    Returns the dataset path.
    """
    dataset = dir_path / f"{stem}{ext}"
    dataset.write_bytes(b"fake dataset content")
    sidecar = dir_path / f"{stem}.py"
    sidecar.write_text(
        _FIXTURE.read_text(encoding="utf-8") if sidecar_src is None else sidecar_src,
        encoding="utf-8",
    )
    return dataset


def _register_inplace_asset(workspace, dataset_path: Path, *, asset_id: str = "ds-1"):
    """Register a workspace-scoped DataAsset pointing at an in-place file.

    Unlike ``import_asset`` (which moves the file into an opaque ``payload``
    directory and loses the original stem), this keeps the dataset where it
    is so the same-stem sidecar convention applies.
    """
    from molexp.workspace.assets import AssetScope, DataAsset

    rel = dataset_path.relative_to(Path(workspace.root))
    now = datetime.now()
    asset = DataAsset(
        asset_id=asset_id,
        name=dataset_path.name,
        scope=AssetScope(kind="workspace", ids=()),
        path=rel,
        created_at=now,
        updated_at=now,
        source_path=str(dataset_path),
        import_action="symlink",
    )
    workspace.catalog.register(asset)
    return asset


# ── ac-001 / ac-002 : trust gate ──────────────────────────────────────────


def test_resolve_does_not_import_the_sidecar(tmp_path, monkeypatch):
    sentinel = tmp_path / "import.sentinel"
    monkeypatch.setenv("MOLEXP_TEST_IMPORT_SENTINEL", str(sentinel))

    dataset = _make_sidecar_dataset(tmp_path)
    info = resolve_sidecar(dataset)

    assert info is not None
    assert info.sidecar_path == tmp_path / "qm9.py"
    assert not sentinel.exists(), "discovery must not execute the sidecar module body"


def test_resolve_returns_none_without_sibling(tmp_path):
    dataset = tmp_path / "plain.bin"
    dataset.write_bytes(b"x")
    assert resolve_sidecar(dataset) is None


def test_load_runs_body_but_not_main_guard(tmp_path, monkeypatch):
    import_sentinel = tmp_path / "import.sentinel"
    main_sentinel = tmp_path / "main.sentinel"
    monkeypatch.setenv("MOLEXP_TEST_IMPORT_SENTINEL", str(import_sentinel))
    monkeypatch.setenv("MOLEXP_TEST_MAIN_SENTINEL", str(main_sentinel))

    dataset = _make_sidecar_dataset(tmp_path)
    reader = load_sidecar_reader(dataset)

    from molpy.io import BaseTrajectoryReader

    assert isinstance(reader, BaseTrajectoryReader)
    assert import_sentinel.exists(), "explicit load must execute the module body"
    assert not main_sentinel.exists(), "explicit load must not run the __main__ guard"


# ── ac-003 : exactly-one reader ────────────────────────────────────────────


def test_load_zero_readers_raises(tmp_path):
    dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_ZERO_READER_SRC)
    with pytest.raises(NoReaderInSidecarError):
        load_sidecar_reader(dataset)


def test_load_two_readers_raises(tmp_path):
    dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_TWO_READER_SRC)
    with pytest.raises(AmbiguousReaderError):
        load_sidecar_reader(dataset)


def test_load_broken_sidecar_raises(tmp_path):
    dataset = _make_sidecar_dataset(tmp_path, sidecar_src=_BROKEN_SRC)
    with pytest.raises(PreviewReaderError):
        load_sidecar_reader(dataset)


def test_load_missing_sidecar_raises(tmp_path):
    dataset = tmp_path / "plain.bin"
    dataset.write_bytes(b"x")
    with pytest.raises(PreviewSidecarNotFoundError):
        load_sidecar_reader(dataset)


# ── ac-006 : host owns the frame cap ───────────────────────────────────────


def test_preview_frames_caps_at_limit(tmp_path):
    dataset = _make_sidecar_dataset(tmp_path)  # FakeReader yields 5 frames
    frames = list(preview_frames(dataset, limit=3))
    assert len(frames) == 3


def test_preview_frames_below_limit_returns_all(tmp_path):
    dataset = _make_sidecar_dataset(tmp_path)
    frames = list(preview_frames(dataset, limit=100))
    assert len(frames) == 5


def test_default_preview_limit_is_positive():
    assert DEFAULT_PREVIEW_LIMIT > 0


# ── ac-004 / ac-005 : route ────────────────────────────────────────────────


def test_route_frames_serves_extxyz(client, workspace):
    dataset = _make_sidecar_dataset(Path(workspace.root))
    asset = _register_inplace_asset(workspace, dataset)

    resp = client.get(f"/api/assets/{asset.asset_id}/preview", params={"format": "frames"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("chemical/x-xyz")
    assert b"C " in resp.content  # extxyz atom line


def test_route_missing_sidecar_is_404(client, workspace):
    dataset = Path(workspace.root) / "plain.bin"
    dataset.write_bytes(b"x")
    asset = _register_inplace_asset(workspace, dataset)

    resp = client.get(f"/api/assets/{asset.asset_id}/preview")
    assert resp.status_code == 404


def test_route_unknown_asset_is_404(client):
    resp = client.get("/api/assets/nope/preview")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("src", "expected"),
    [(_ZERO_READER_SRC, 422), (_TWO_READER_SRC, 422), (_BROKEN_SRC, 422)],
)
def test_route_bad_sidecar_is_422_not_500(client, workspace, src, expected):
    dataset = _make_sidecar_dataset(Path(workspace.root), sidecar_src=src)
    asset = _register_inplace_asset(workspace, dataset)

    resp = client.get(f"/api/assets/{asset.asset_id}/preview", params={"format": "frames"})
    assert resp.status_code == expected


def test_route_png_serves_image(client, workspace):
    pytest.importorskip("molvis")
    dataset = _make_sidecar_dataset(Path(workspace.root))
    asset = _register_inplace_asset(workspace, dataset)

    resp = client.get(f"/api/assets/{asset.asset_id}/preview", params={"format": "png"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")


# ── ac-007 : listing flag ──────────────────────────────────────────────────


def test_listing_surfaces_sidecar_flag(client, workspace):
    root = Path(workspace.root)
    ds_with = _make_sidecar_dataset(root, stem="qm9")
    _register_inplace_asset(workspace, ds_with, asset_id="with-sidecar")
    ds_without = root / "plain.bin"
    ds_without.write_bytes(b"x")
    _register_inplace_asset(workspace, ds_without, asset_id="no-sidecar")

    resp = client.get("/api/assets")
    assert resp.status_code == 200
    by_id = {a["id"]: a for a in resp.json()}
    assert by_id["with-sidecar"]["has_preview_sidecar"] is True
    assert by_id["no-sidecar"]["has_preview_sidecar"] is False


# ── register-in-place → preview (index-driven, no auto-discovery) ──────────


def test_register_then_preview_follows_the_index(client, workspace):
    # A QM9-style dataset + sidecar sit in the workspace tree.
    _make_sidecar_dataset(Path(workspace.root), stem="qm9")

    # Register the file in place (no upload, no copy) — this is the explicit
    # step that puts it in the catalog index.
    reg = client.post("/api/assets/data/register", json={"path": "qm9.bin"})
    assert reg.status_code == 201
    body = reg.json()
    assert body["has_preview_sidecar"] is True

    # Preview follows the registered asset id.
    resp = client.get(f"/api/assets/{body['id']}/preview", params={"format": "frames"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("chemical/x-xyz")
    # Atom-count line is correct (guards the molpy XYZ-writer fix end-to-end).
    assert resp.content.splitlines()[0] == b"2"


def test_register_rejects_path_outside_workspace(client):
    resp = client.post("/api/assets/data/register", json={"path": "/etc/passwd"})
    assert resp.status_code == 400


# ── ac-008 : QM9 reference reader (skip-gated) ─────────────────────────────

# The reference sidecar lives outside the molexp repo (it ships next to a
# user's dataset, not inside the package). Reading real QM9 needs a multi-GB
# download, so this test loads the reader through the real sidecar machinery
# and injects a tiny stand-in record source that mimics molix QM9Source's
# per-sample mapping (``Z`` / ``pos`` / ``targets``).
_QM9_REFERENCE = Path("/Users/roykid/work/molcrafts/qm9.py")


def test_qm9_reference_reader_builds_frames(tmp_path):
    if not _QM9_REFERENCE.is_file():
        pytest.skip("QM9 reference sidecar not present in this checkout")

    dataset = _make_sidecar_dataset(
        tmp_path, stem="qm9", sidecar_src=_QM9_REFERENCE.read_text(encoding="utf-8")
    )

    reader = load_sidecar_reader(dataset)
    assert type(reader).__name__ == "QM9Reader"

    # Inject a stand-in for molix QM9Source: an indexable, len()-able sequence
    # of per-sample mappings. Bypasses the real molix/torch dataset.
    reader._source = [
        {
            "Z": [6, 1, 1, 1, 1],
            "pos": [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
            "targets": {"mu": 1.23, "gap": 0.5},
        },
    ]

    assert reader.n_frames == 1
    frame = reader.read_frame(0)
    atoms = frame["atoms"]
    cols = {k: list(atoms[k]) for k in ("element", "x", "y", "z")}
    assert len({len(v) for v in cols.values()}) == 1  # equal-length columns
    assert cols["element"][0] == "C"
    assert any(k.startswith("qm9.") for k in frame.metadata)
