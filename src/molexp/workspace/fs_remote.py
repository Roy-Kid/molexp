"""RemoteFileSystem — delegates to a molq Transport, implementing FileSystem."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from typing import IO, Any

from molq.transport import Transport

from .fs import StatResult


class RemoteFileSystem:
    """FileSystem backed by a molq Transport (local or SSH).

    All path arguments are interpreted on the transport's filesystem.
    No raw shell commands — every operation delegates to Transport methods.
    """

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    # ── Path ops (static — string manipulation only) ────────────────────

    @staticmethod
    def join(*parts: str) -> str:
        return "/".join(p.strip("/") for p in parts if p)

    @staticmethod
    def dirname(path: str) -> str:
        return path.rsplit("/", 1)[0] if "/" in path else "."

    @staticmethod
    def basename(path: str) -> str:
        return path.rsplit("/", 1)[-1] if path else ""

    @staticmethod
    def resolve(path: str) -> str:
        return path

    @staticmethod
    def is_absolute(path: str) -> bool:
        return path.startswith(("/", "~/"))

    # ── Delegated to Transport ──────────────────────────────────────────

    def exists(self, path: str) -> bool:
        return self._t.exists(path)

    def is_dir(self, path: str) -> bool:
        return self._t.is_dir(path)

    def is_file(self, path: str) -> bool:
        return self._t.is_file(path)

    def mkdir(self, path: str, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._t.mkdir(path, parents=parents, exist_ok=exist_ok)

    def listdir(self, path: str) -> list[str]:
        return self._t.listdir(path)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:  # noqa: ARG002
        return self._t.read_text(path)

    def read_bytes(self, path: str) -> bytes:
        return self._t.read_bytes(path)

    def open(self, path: str, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:  # noqa: ARG002
        return io.StringIO(self._t.read_text(path))

    def write_text(self, path: str, content: str, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._t.write_text(path, content)

    def write_bytes(self, path: str, content: bytes, *, mode: int = 0o600) -> None:  # noqa: ARG002
        self._t.write_bytes(path, content)

    def rename(self, src: str, dst: str) -> None:
        self._t.rename(src, dst)

    def remove(self, path: str, *, recursive: bool = False) -> None:
        self._t.remove(path, recursive=recursive)

    def copy(self, src: str, dst: str) -> None:
        self._t.copy(src, dst)

    def copytree(self, src: str, dst: str, *, dirs_exist_ok: bool = False) -> None:  # noqa: ARG002
        self._t.copytree(src, dst)

    def stat(self, path: str) -> StatResult:
        d = self._t.stat(path)
        return StatResult(**d)  # type: ignore[arg-type]

    def lstat(self, path: str) -> StatResult:
        return self.stat(path)

    def touch(self, path: str) -> None:
        self._t.touch(path)

    def chmod(self, path: str, mode: int) -> None:
        self._t.chmod(path, mode)

    def getsize(self, path: str) -> int:
        return self._t.getsize(path)

    def symlink(self, src: str, dst: str) -> None:
        self._t.symlink(src, dst)

    def glob(self, path: str, pattern: str) -> Iterable[str]:
        for name in self._t.listdir(path):
            if _glob_match(name, pattern):
                yield self.join(path, name)

    def rglob(self, path: str, pattern: str) -> Iterable[str]:
        for name in self._t.listdir(path):
            full = self.join(path, name)
            if _glob_match(name, pattern):
                yield full
            if self._t.is_dir(full):
                yield from self.rglob(full, pattern)

    # ── Atomic I/O ──────────────────────────────────────────────────────

    def atomic_write_json(self, path: str, data: object) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        self._t.write_text(path, content)

    def atomic_write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> None:  # noqa: ARG002
        self._t.write_text(path, content)


def _glob_match(name: str, pattern: str) -> bool:
    """Simple glob matching: * matches anything, otherwise exact."""
    import fnmatch

    return fnmatch.fnmatch(name, pattern)
