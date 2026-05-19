"""FileSystem abstraction — the workspace layer speaks this, never raw pathlib.

Two implementations: LocalFileSystem (wraps pathlib/os/shutil) and
RemoteFileSystem (wraps molq Transport + shell commands for missing ops).

All path arguments are interpreted on this filesystem.  ``str`` and any
:class:`os.PathLike[str]` (notably :class:`molexp.Path` /
:class:`pathlib.PurePosixPath`) are accepted; implementations normalize
to ``str`` internally before doing string operations or shelling out.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import IO, Any, Protocol, runtime_checkable

PathArg = str | os.PathLike[str]
"""Anything that ``os.fspath()`` can turn into a POSIX path string.

Includes plain ``str``, :class:`molexp.Path`, :class:`pathlib.PurePosixPath`,
and any third-party object implementing ``__fspath__() -> str``.
"""


@dataclass(frozen=True)
class StatResult:
    """Cross-platform stat result. Fields match the subset workspace needs."""

    size: int
    mtime: float
    is_dir: bool
    is_file: bool


@runtime_checkable
class FileSystem(Protocol):
    """Where files and directories actually live.

    Implementations: LocalFileSystem (stdlib), RemoteFileSystem (SSH via molq Transport).
    Third-party plugins may implement this Protocol without importing workspace internals.
    """

    # ── Path operations ──────────────────────────────────────────────────

    def join(self, *parts: PathArg) -> str: ...
    def dirname(self, path: PathArg) -> str: ...
    def basename(self, path: PathArg) -> str: ...
    def resolve(self, path: PathArg) -> str: ...
    def is_absolute(self, path: PathArg) -> bool: ...

    # ── Existence / type ─────────────────────────────────────────────────

    def exists(self, path: PathArg) -> bool: ...
    def is_dir(self, path: PathArg) -> bool: ...
    def is_file(self, path: PathArg) -> bool: ...

    # ── Directory operations ─────────────────────────────────────────────

    def mkdir(self, path: PathArg, *, parents: bool = True, exist_ok: bool = True) -> None: ...
    def listdir(self, path: PathArg) -> list[str]: ...
    def glob(self, path: PathArg, pattern: str) -> Iterable[str]: ...
    def rglob(self, path: PathArg, pattern: str) -> Iterable[str]: ...

    # ── Read ─────────────────────────────────────────────────────────────

    def read_text(self, path: PathArg, encoding: str = "utf-8") -> str: ...
    def read_bytes(self, path: PathArg) -> bytes: ...
    def open(self, path: PathArg, mode: str = "r", encoding: str = "utf-8") -> IO[Any]: ...

    # ── Write ────────────────────────────────────────────────────────────

    def write_text(self, path: PathArg, content: str, *, mode: int = 0o600) -> None: ...
    def write_bytes(self, path: PathArg, content: bytes, *, mode: int = 0o600) -> None: ...

    # ── Mutations ────────────────────────────────────────────────────────

    def rename(self, src: PathArg, dst: PathArg) -> None: ...
    def remove(self, path: PathArg, *, recursive: bool = False) -> None: ...
    def copy(self, src: PathArg, dst: PathArg) -> None: ...
    def copytree(self, src: PathArg, dst: PathArg, *, dirs_exist_ok: bool = False) -> None: ...

    # ── Metadata ─────────────────────────────────────────────────────────

    def stat(self, path: PathArg) -> StatResult: ...
    def lstat(self, path: PathArg) -> StatResult: ...
    def touch(self, path: PathArg) -> None: ...
    def chmod(self, path: PathArg, mode: int) -> None: ...
    def getsize(self, path: PathArg) -> int: ...

    # ── Symlinks ─────────────────────────────────────────────────────────

    def symlink(self, src: PathArg, dst: PathArg) -> None: ...

    # ── Atomic I/O ───────────────────────────────────────────────────────

    def atomic_write_json(self, path: PathArg, data: object) -> None: ...
    def atomic_write_text(self, path: PathArg, content: str, *, encoding: str = "utf-8") -> None: ...
