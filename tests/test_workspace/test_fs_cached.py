"""Unit tests for :class:`CachedRemoteFileSystem`.

Drives a fake :class:`FileSystem` that records call counts so we can
assert "second read is mirror-served, zero inner calls".  No SSH.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import IO, Any

import pytest

from molexp.workspace.fs import StatResult
from molexp.workspace.fs_cached import (
    INDEX_FILE_NAMES,
    CachedRemoteFileSystem,
)
from molexp.workspace.fs_local import LocalFileSystem


class _FakeRemoteFS:
    """Counts every call; backs a string-keyed in-memory store."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.calls: Counter[str] = Counter()

    def _hit(self, name: str) -> None:
        self.calls[name] += 1

    # ── Path ops ──
    @staticmethod
    def join(*parts: str) -> str:
        return "/".join(str(p).strip("/") for p in parts if p)

    @staticmethod
    def dirname(path: str) -> str:
        s = str(path)
        return s.rsplit("/", 1)[0] if "/" in s else "."

    @staticmethod
    def basename(path: str) -> str:
        return str(path).rsplit("/", 1)[-1]

    @staticmethod
    def resolve(path: str) -> str:
        return str(path)

    @staticmethod
    def is_absolute(path: str) -> bool:
        return str(path).startswith("/")

    # ── Existence ──
    def exists(self, path: str) -> bool:
        self._hit("exists")
        key = str(path)
        return key in self.files or key in self.dirs

    def is_dir(self, path: str) -> bool:
        self._hit("is_dir")
        return str(path) in self.dirs

    def is_file(self, path: str) -> bool:
        self._hit("is_file")
        return str(path) in self.files

    # ── Read ──
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        self._hit("read_text")
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key].decode(encoding)

    def read_bytes(self, path: str) -> bytes:
        self._hit("read_bytes")
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def open(self, path: str, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:
        import io as _io

        return _io.StringIO(self.read_text(path, encoding=encoding))

    # ── Write ──
    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._hit("write_text")
        self.files[str(path)] = content.encode("utf-8")

    def write_bytes(self, path: str, content: bytes, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._hit("write_bytes")
        self.files[str(path)] = content

    # ── Mutations ──
    def rename(self, src: str, dst: str) -> None:
        self._hit("rename")
        self.files[str(dst)] = self.files.pop(str(src))

    def remove(self, path: str, *, recursive: bool = False) -> None:
        self._hit("remove")
        key = str(path)
        if recursive:
            for k in list(self.files):
                if k.startswith(key.rstrip("/") + "/") or k == key:
                    del self.files[k]
            self.dirs.discard(key)
        else:
            self.files.pop(key, None)
            self.dirs.discard(key)

    def copy(self, src: str, dst: str) -> None:
        self._hit("copy")
        self.files[str(dst)] = self.files[str(src)]

    def copytree(self, src: str, dst: str, *, dirs_exist_ok: bool = False) -> None:  # noqa: ARG002
        self._hit("copytree")
        prefix = str(src).rstrip("/") + "/"
        for k, v in list(self.files.items()):
            if k.startswith(prefix):
                self.files[str(dst).rstrip("/") + "/" + k[len(prefix) :]] = v

    # ── Dir ops ──
    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:  # noqa: ARG002
        self._hit("mkdir")
        self.dirs.add(str(path))

    def listdir(self, path: str) -> list[str]:
        self._hit("listdir")
        prefix = str(path).rstrip("/") + "/"
        names: set[str] = set()
        for k in (*self.files.keys(), *self.dirs):
            if k.startswith(prefix):
                tail = k[len(prefix) :]
                names.add(tail.split("/", 1)[0])
        return sorted(names)

    def glob(self, path: str, pattern: str) -> list[str]:  # noqa: ARG002
        return []

    def rglob(self, path: str, pattern: str) -> list[str]:  # noqa: ARG002
        return []

    # ── Metadata ──
    def stat(self, path: str) -> StatResult:
        self._hit("stat")
        key = str(path)
        if key in self.files:
            return StatResult(size=len(self.files[key]), mtime=42.0, is_dir=False, is_file=True)
        if key in self.dirs:
            return StatResult(size=0, mtime=42.0, is_dir=True, is_file=False)
        raise FileNotFoundError(key)

    def lstat(self, path: str) -> StatResult:
        return self.stat(path)

    def touch(self, path: str) -> None:
        self._hit("touch")
        self.files.setdefault(str(path), b"")

    def chmod(self, path: str, mode: int) -> None:
        self._hit("chmod")

    def getsize(self, path: str) -> int:
        self._hit("getsize")
        return len(self.files[str(path)])

    def symlink(self, src: str, dst: str) -> None:
        self._hit("symlink")

    # ── Atomic I/O ──
    def atomic_write_json(self, path: str, data: object) -> None:
        import json as _json

        self._hit("atomic_write_json")
        self.files[str(path)] = (_json.dumps(data) + "\n").encode("utf-8")

    def atomic_write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:  # noqa: ARG002
        self._hit("atomic_write_text")
        self.files[str(path)] = content.encode("utf-8")


@pytest.fixture
def fake() -> _FakeRemoteFS:
    fake = _FakeRemoteFS()
    fake.files["/scratch/me/log.txt"] = b"hello"
    fake.dirs.add("/scratch/me")
    return fake


@pytest.fixture
def cached(fake: _FakeRemoteFS, tmp_path: Path) -> CachedRemoteFileSystem:
    return CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=300)


# ── Core read caching ──────────────────────────────────────────────────


@pytest.mark.unit
def test_first_read_fetches_inner_second_read_hits_mirror(
    cached: CachedRemoteFileSystem, fake: _FakeRemoteFS
):
    first = cached.read_bytes("/scratch/me/log.txt")
    second = cached.read_bytes("/scratch/me/log.txt")
    assert first == b"hello"
    assert second == b"hello"
    assert fake.calls["read_bytes"] == 1, f"saw {fake.calls!r}"


@pytest.mark.unit
def test_read_text_uses_mirror_bytes(cached: CachedRemoteFileSystem, fake: _FakeRemoteFS):
    cached.read_text("/scratch/me/log.txt")
    cached.read_text("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 1
    assert fake.calls["read_text"] == 0


@pytest.mark.unit
def test_mirror_layout_strips_leading_slash(
    cached: CachedRemoteFileSystem, fake: _FakeRemoteFS, tmp_path: Path
):
    cached.read_bytes("/scratch/me/log.txt")
    expected = tmp_path / "mirror" / "files" / "scratch" / "me" / "log.txt"
    assert expected.read_bytes() == b"hello"
    assert fake.calls["read_bytes"] == 1


# ── Negative cache ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_missing_short_circuits_subsequent_exists(
    cached: CachedRemoteFileSystem, fake: _FakeRemoteFS
):
    assert cached.exists("/scratch/me/nope") is False
    fake.calls.clear()
    assert cached.exists("/scratch/me/nope") is False
    assert fake.calls["exists"] == 0


@pytest.mark.unit
def test_read_missing_propagates_filenotfound(cached: CachedRemoteFileSystem):
    with pytest.raises(FileNotFoundError):
        cached.read_bytes("/scratch/me/nope")


# ── Invalidation on write ──────────────────────────────────────────────


@pytest.mark.unit
def test_write_invalidates_cache(cached: CachedRemoteFileSystem, fake: _FakeRemoteFS):
    cached.read_bytes("/scratch/me/log.txt")
    cached.write_text("/scratch/me/log.txt", "new content")
    fake.calls.clear()
    assert cached.read_text("/scratch/me/log.txt") == "new content"
    assert fake.calls["read_bytes"] == 1, "must re-fetch after write"


@pytest.mark.unit
def test_rename_invalidates_both_ends(cached: CachedRemoteFileSystem, fake: _FakeRemoteFS):
    cached.read_bytes("/scratch/me/log.txt")
    cached.rename("/scratch/me/log.txt", "/scratch/me/log2.txt")
    fake.calls.clear()
    assert cached.read_bytes("/scratch/me/log2.txt") == b"hello"
    assert fake.calls["read_bytes"] == 1


@pytest.mark.unit
def test_remove_invalidates_entry(cached: CachedRemoteFileSystem, fake: _FakeRemoteFS):
    cached.read_bytes("/scratch/me/log.txt")
    cached.remove("/scratch/me/log.txt")
    fake.calls.clear()
    with pytest.raises(FileNotFoundError):
        cached.read_bytes("/scratch/me/log.txt")


# ── TTL expiry ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ttl_zero_always_revalidates(fake: _FakeRemoteFS, tmp_path: Path):
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=0)
    cached.read_bytes("/scratch/me/log.txt")
    fake.calls.clear()
    cached.read_bytes("/scratch/me/log.txt")
    # TTL=0 means no fast path — every read re-fetches inner bytes.
    assert fake.calls["read_bytes"] == 1


@pytest.mark.unit
def test_ttl_expiry_triggers_refetch(
    fake: _FakeRemoteFS, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=10)
    base = time.time()
    monkeypatch.setattr("molexp.workspace.fs_cached.time.time", lambda: base)
    cached.read_bytes("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 1

    # Within TTL — mirror hit
    monkeypatch.setattr("molexp.workspace.fs_cached.time.time", lambda: base + 5)
    cached.read_bytes("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 1

    # After TTL — re-fetch
    monkeypatch.setattr("molexp.workspace.fs_cached.time.time", lambda: base + 20)
    cached.read_bytes("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 2


# ── Sidecar persistence ────────────────────────────────────────────────


@pytest.mark.unit
def test_sidecar_round_trip_across_instances(fake: _FakeRemoteFS, tmp_path: Path):
    mirror_root = tmp_path / "mirror"
    first = CachedRemoteFileSystem(fake, mirror_root=mirror_root, ttl_seconds=300)
    first.read_bytes("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 1

    second = CachedRemoteFileSystem(fake, mirror_root=mirror_root, ttl_seconds=300)
    # Cached_paths should include the entry from the first instance.
    assert "/scratch/me/log.txt" in second.cached_paths()
    fake.calls.clear()
    second.read_bytes("/scratch/me/log.txt")
    assert fake.calls["read_bytes"] == 0, "must serve from mirror after re-instantiation"


# ── invalidate() public surface ────────────────────────────────────────


@pytest.mark.unit
def test_invalidate_scope_indices_drops_only_index_files(
    fake: _FakeRemoteFS, tmp_path: Path
):
    fake.files["/scratch/me/project.json"] = b'{"items":[]}'
    fake.files["/scratch/me/runs/a/stdout.log"] = b"log bytes"
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    cached.read_bytes("/scratch/me/project.json")
    cached.read_bytes("/scratch/me/runs/a/stdout.log")

    dropped = cached.invalidate(scope="indices")
    assert dropped == 1
    assert "/scratch/me/runs/a/stdout.log" in cached.cached_paths()
    assert "/scratch/me/project.json" not in cached.cached_paths()


@pytest.mark.unit
def test_invalidate_scope_all_clears_everything(
    fake: _FakeRemoteFS, tmp_path: Path
):
    fake.files["/scratch/me/extra.txt"] = b"x"
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    cached.read_bytes("/scratch/me/log.txt")
    cached.read_bytes("/scratch/me/extra.txt")

    dropped = cached.invalidate(scope="all")
    assert dropped == 2
    assert cached.cached_paths() == []
    assert not (tmp_path / "mirror" / "files").exists()


@pytest.mark.unit
def test_invalidate_specific_path(fake: _FakeRemoteFS, tmp_path: Path):
    fake.files["/scratch/me/extra.txt"] = b"x"
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    cached.read_bytes("/scratch/me/log.txt")
    cached.read_bytes("/scratch/me/extra.txt")

    cached.invalidate("/scratch/me/log.txt")
    assert "/scratch/me/extra.txt" in cached.cached_paths()
    assert "/scratch/me/log.txt" not in cached.cached_paths()


@pytest.mark.unit
def test_invalidate_rejects_unknown_scope(cached: CachedRemoteFileSystem):
    with pytest.raises(ValueError, match="unknown scope"):
        cached.invalidate(scope="bogus")


# ── Stat caching ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_stat_serves_from_cache_after_read(cached: CachedRemoteFileSystem, fake: _FakeRemoteFS):
    cached.read_bytes("/scratch/me/log.txt")
    fake.calls.clear()
    info = cached.stat("/scratch/me/log.txt")
    assert info.is_file is True
    assert fake.calls["stat"] == 0


# ── Sanity ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_index_file_names_well_defined():
    assert "workspace.json" in INDEX_FILE_NAMES
    assert "stdout.log" not in INDEX_FILE_NAMES


@pytest.mark.unit
def test_path_ops_delegate_to_inner_without_io(fake: _FakeRemoteFS, tmp_path: Path):
    cached = CachedRemoteFileSystem(fake, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    cached.join("/a", "b", "c")
    cached.dirname("/a/b/c")
    cached.basename("/a/b/c")
    cached.resolve("/a/b/c")
    # None of these trigger remote I/O.
    assert sum(fake.calls.values()) == 0


@pytest.mark.unit
def test_local_filesystem_satisfies_inner_protocol(tmp_path: Path):
    """Sanity: CachedRemoteFileSystem can wrap a LocalFileSystem (useful for tests)."""
    local = LocalFileSystem()
    cached = CachedRemoteFileSystem(local, mirror_root=tmp_path / "mirror", ttl_seconds=300)
    target = tmp_path / "src.txt"
    target.write_text("ok")
    assert cached.read_text(str(target)) == "ok"
