"""RemoteFileSystem — delegates to a molq Transport, implementing FileSystem."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable
from typing import IO, Any, cast

from molq.transport import Transport

from .fs import PathArg, StatResult


class RemoteFileSystem:
    """FileSystem backed by a molq Transport (local or SSH).

    Accepts ``str`` or any :class:`os.PathLike[str]` (incl. :class:`molexp.Path`);
    paths are normalized to ``str`` via :func:`os.fspath` before string ops or
    transport calls.  All path arguments are interpreted on the transport's
    filesystem.  No raw shell commands — every operation delegates to
    :class:`molq.transport.Transport` methods.
    """

    def __init__(self, transport: Transport) -> None:
        # molq.transport.Transport is a narrow Protocol; this class also uses
        # filesystem ops (is_dir/listdir/rename/copy/stat/…) that the concrete
        # transports provide but the Protocol doesn't statically declare. Widen
        # to Any so the dynamic surface resolves without changing runtime behavior.
        self._t: Any = transport

    # ── Path ops (static — string manipulation only) ────────────────────

    @staticmethod
    def join(*parts: PathArg) -> str:
        return "/".join(os.fspath(p).strip("/") for p in parts if p)

    @staticmethod
    def dirname(path: PathArg) -> str:
        s = os.fspath(path)
        return s.rsplit("/", 1)[0] if "/" in s else "."

    @staticmethod
    def basename(path: PathArg) -> str:
        s = os.fspath(path)
        return s.rsplit("/", 1)[-1] if s else ""

    @staticmethod
    def resolve(path: PathArg) -> str:
        return os.fspath(path)

    @staticmethod
    def is_absolute(path: PathArg) -> bool:
        return os.fspath(path).startswith(("/", "~/"))

    # ── Delegated to Transport ──────────────────────────────────────────

    def exists(self, path: PathArg) -> bool:
        return self._t.exists(os.fspath(path))

    def is_dir(self, path: PathArg) -> bool:
        return self._t.is_dir(os.fspath(path))

    def is_file(self, path: PathArg) -> bool:
        return self._t.is_file(os.fspath(path))

    def mkdir(self, path: PathArg, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._t.mkdir(os.fspath(path), parents=parents, exist_ok=exist_ok)

    def listdir(self, path: PathArg) -> list[str]:
        return self._t.listdir(os.fspath(path))

    def read_text(self, path: PathArg, encoding: str = "utf-8") -> str:  # noqa: ARG002
        return self._t.read_text(os.fspath(path))

    def read_bytes(self, path: PathArg) -> bytes:
        return self._t.read_bytes(os.fspath(path))

    def open(self, path: PathArg, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:  # noqa: ARG002
        return io.StringIO(self._t.read_text(os.fspath(path)))

    def write_text(self, path: PathArg, content: str, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._t.write_text(os.fspath(path), content)

    def write_bytes(self, path: PathArg, content: bytes, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._t.write_bytes(os.fspath(path), content)

    def rename(self, src: PathArg, dst: PathArg) -> None:
        self._t.rename(os.fspath(src), os.fspath(dst))

    def remove(self, path: PathArg, *, recursive: bool = False) -> None:
        self._t.remove(os.fspath(path), recursive=recursive)

    def copy(self, src: PathArg, dst: PathArg) -> None:
        self._t.copy(os.fspath(src), os.fspath(dst))

    def copytree(self, src: PathArg, dst: PathArg, *, dirs_exist_ok: bool = False) -> None:  # noqa: ARG002
        self._t.copytree(os.fspath(src), os.fspath(dst))

    def stat(self, path: PathArg) -> StatResult:
        d = cast("dict[str, Any]", self._t.stat(os.fspath(path)))
        return StatResult(**d)

    def lstat(self, path: PathArg) -> StatResult:
        return self.stat(path)

    def touch(self, path: PathArg) -> None:
        self._t.touch(os.fspath(path))

    def chmod(self, path: PathArg, mode: int) -> None:
        self._t.chmod(os.fspath(path), mode)

    def getsize(self, path: PathArg) -> int:
        return self._t.getsize(os.fspath(path))

    def symlink(self, src: PathArg, dst: PathArg) -> None:
        self._t.symlink(os.fspath(src), os.fspath(dst))

    def glob(self, path: PathArg, pattern: str) -> Iterable[str]:
        base = os.fspath(path)
        for name in self._t.listdir(base):
            if _glob_match(name, pattern):
                yield self.join(base, name)

    def rglob(self, path: PathArg, pattern: str) -> Iterable[str]:
        base = os.fspath(path)
        for name in self._t.listdir(base):
            full = self.join(base, name)
            if _glob_match(name, pattern):
                yield full
            if self._t.is_dir(full):
                yield from self.rglob(full, pattern)

    # ── Atomic I/O ──────────────────────────────────────────────────────

    def atomic_write_json(self, path: PathArg, data: object) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        self._t.write_text(os.fspath(path), content)

    def atomic_write_text(self, path: PathArg, content: str, *, encoding: str = "utf-8") -> None:  # noqa: ARG002
        self._t.write_text(os.fspath(path), content)


def _glob_match(name: str, pattern: str) -> bool:
    """Simple glob matching: * matches anything, otherwise exact."""
    import fnmatch

    return fnmatch.fnmatch(name, pattern)
