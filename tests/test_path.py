"""Tests for :class:`molexp.Path` — cross-host POSIX path primitive."""

from __future__ import annotations

import os
import pickle
from pathlib import PurePosixPath

import pytest

from molexp import Path


class TestIdentity:
    """Type identity and inheritance — the reason for subclassing at all."""

    def test_meaningful_predicate(self) -> None:
        """``isinstance(p, molexp.Path)`` is a useful runtime check."""
        assert isinstance(Path("/a"), Path)
        assert not isinstance(PurePosixPath("/a"), Path)
        assert not isinstance("/a", Path)


class TestPathArithmetic:
    """The ergonomic win over ``str`` — ``/``, ``.parent``, ``.name``, etc."""

    def test_truediv_with_str(self) -> None:
        p = Path("/scratch") / "user"
        assert isinstance(p, Path)
        assert str(p) == "/scratch/user"

    def test_parent(self) -> None:
        assert Path("/a/b/c").parent == Path("/a/b")
        assert isinstance(Path("/a/b/c").parent, Path)


class TestFsPath:
    """``__fspath__`` makes ``os.fspath()`` and many APIs accept ``Path``."""

    def test_os_fspath(self) -> None:
        assert os.fspath(Path("/a/b")) == "/a/b"


class TestEqualityAndHash:
    """Equality / hashing — inherited from ``PurePosixPath``, by string identity."""

    def test_neq_str(self) -> None:
        """A Path is not equal to its string form (PurePath semantics)."""
        assert Path("/a/b") != "/a/b"


class TestPickle:
    """Persistence — subclassed PurePath must survive a pickle round-trip."""

    def test_pickle_roundtrip(self) -> None:
        original = Path("/scratch/user/run_0")
        restored = pickle.loads(pickle.dumps(original))
        assert restored == original
        assert isinstance(restored, Path)


class TestLocalIO:
    """I/O methods delegate to :class:`pathlib.Path` for the local filesystem."""

    def test_read_write_text_roundtrip(self, tmp_path) -> None:
        p = Path(str(tmp_path / "hello.txt"))
        p.write_text("hi\n")
        assert p.read_text() == "hi\n"

    def test_exists_is_file_is_dir(self, tmp_path) -> None:
        p = Path(str(tmp_path / "x.txt"))
        assert not p.exists()
        p.write_text("x")
        assert p.exists()
        assert p.is_file()
        assert not p.is_dir()
        d = Path(str(tmp_path))
        assert d.is_dir()
        assert not d.is_file()

    def test_mkdir_and_iterdir(self, tmp_path) -> None:
        root = Path(str(tmp_path / "tree"))
        root.mkdir(parents=True)
        (root / "a").write_text("a")
        (root / "b").write_text("b")
        names = sorted(child.name for child in root.iterdir())
        assert names == ["a", "b"]
        for child in root.iterdir():
            assert isinstance(child, Path)


class TestSlots:
    """``__slots__ = ()`` prevents accidental attribute attachment."""

    def test_cannot_set_arbitrary_attr(self) -> None:
        p = Path("/a")
        with pytest.raises(AttributeError):
            p.arbitrary_attr = "boom"  # type: ignore[attr-defined]
