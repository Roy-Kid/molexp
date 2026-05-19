"""Tests for :class:`molexp.Path` — cross-host POSIX path primitive."""

from __future__ import annotations

import os
import pickle
from pathlib import PurePosixPath

import pytest

from molexp import Path


class TestIdentity:
    """Type identity and inheritance — the reason for subclassing at all."""

    def test_is_subclass_of_pure_posix_path(self) -> None:
        assert issubclass(Path, PurePosixPath)

    def test_isinstance_pure_posix_path(self) -> None:
        assert isinstance(Path("/a/b"), PurePosixPath)

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

    def test_truediv_chain(self) -> None:
        p = Path("/scratch") / "user" / "experiment_1" / "run_0"
        assert str(p) == "/scratch/user/experiment_1/run_0"

    def test_truediv_with_path(self) -> None:
        p = Path("/scratch") / Path("user")
        assert isinstance(p, Path)
        assert str(p) == "/scratch/user"

    def test_parent(self) -> None:
        assert Path("/a/b/c").parent == Path("/a/b")
        assert isinstance(Path("/a/b/c").parent, Path)

    def test_name(self) -> None:
        assert Path("/a/b/c").name == "c"

    def test_parts(self) -> None:
        assert Path("/a/b/c").parts == ("/", "a", "b", "c")

    def test_joinpath(self) -> None:
        p = Path("/a").joinpath("b", "c")
        assert isinstance(p, Path)
        assert str(p) == "/a/b/c"

    def test_with_name(self) -> None:
        assert Path("/a/b/c").with_name("d") == Path("/a/b/d")

    def test_with_suffix(self) -> None:
        assert Path("/a/b/c.txt").with_suffix(".json") == Path("/a/b/c.json")


class TestFsPath:
    """``__fspath__`` makes ``os.fspath()`` and many APIs accept ``Path``."""

    def test_fspath_returns_str(self) -> None:
        assert Path("/a/b").__fspath__() == "/a/b"

    def test_os_fspath(self) -> None:
        assert os.fspath(Path("/a/b")) == "/a/b"

    def test_str(self) -> None:
        assert str(Path("/a/b")) == "/a/b"


class TestEqualityAndHash:
    """Equality / hashing — inherited from ``PurePosixPath``, by string identity."""

    def test_eq_same_path(self) -> None:
        assert Path("/a/b") == Path("/a/b")

    def test_eq_with_pure_posix_path(self) -> None:
        """Cross-type equality is symmetric and string-based (PurePath semantics)."""
        assert Path("/a/b") == PurePosixPath("/a/b")
        assert PurePosixPath("/a/b") == Path("/a/b")

    def test_neq_different_path(self) -> None:
        assert Path("/a/b") != Path("/a/c")

    def test_neq_str(self) -> None:
        """A Path is not equal to its string form (PurePath semantics)."""
        assert Path("/a/b") != "/a/b"

    def test_hashable(self) -> None:
        d = {Path("/a"): 1, Path("/b"): 2}
        assert d[Path("/a")] == 1
        assert d[Path("/b")] == 2


class TestPickle:
    """Persistence — subclassed PurePath must survive a pickle round-trip."""

    def test_pickle_roundtrip(self) -> None:
        original = Path("/scratch/user/run_0")
        restored = pickle.loads(pickle.dumps(original))
        assert restored == original
        assert isinstance(restored, Path)


class TestNoLocalIO:
    """A pure path must not expose I/O methods that would silently hit local FS."""

    @pytest.mark.parametrize(
        "method",
        ["exists", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "iterdir"],
    )
    def test_no_io_methods(self, method: str) -> None:
        """``PurePosixPath`` deliberately lacks these — guarding against accidental reintroduction."""
        assert not hasattr(Path("/a"), method), (
            f"Path should NOT have {method!r} — it would silently hit the local filesystem"
        )


class TestSlots:
    """``__slots__ = ()`` prevents accidental attribute attachment."""

    def test_cannot_set_arbitrary_attr(self) -> None:
        p = Path("/a")
        with pytest.raises(AttributeError):
            p.arbitrary_attr = "boom"  # type: ignore[attr-defined]
