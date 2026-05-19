"""LocalFileSystem — wraps pathlib, os, and shutil behind the FileSystem Protocol."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from .fs import FileSystem, PathArg, StatResult


class LocalFileSystem:
    """Concrete FileSystem backed by the local filesystem via stdlib.

    Accepts ``str`` or any :class:`os.PathLike[str]` (incl. :class:`molexp.Path`);
    returns ``str``.  Internally wraps every argument with :class:`pathlib.Path`,
    which accepts ``PathLike`` natively.
    """

    # ── Path operations ──────────────────────────────────────────────────

    @staticmethod
    def join(*parts: PathArg) -> str:
        return str(Path(*(os.fspath(p) for p in parts)))

    @staticmethod
    def dirname(path: PathArg) -> str:
        return str(Path(path).parent)

    @staticmethod
    def basename(path: PathArg) -> str:
        return str(Path(path).name)

    @staticmethod
    def resolve(path: PathArg) -> str:
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def is_absolute(path: PathArg) -> bool:
        return Path(path).is_absolute()

    # ── Existence / type ─────────────────────────────────────────────────

    @staticmethod
    def exists(path: PathArg) -> bool:
        return Path(path).exists()

    @staticmethod
    def is_dir(path: PathArg) -> bool:
        return Path(path).is_dir()

    @staticmethod
    def is_file(path: PathArg) -> bool:
        return Path(path).is_file()

    # ── Directory operations ─────────────────────────────────────────────

    @staticmethod
    def mkdir(path: PathArg, *, parents: bool = True, exist_ok: bool = True) -> None:
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)

    @staticmethod
    def listdir(path: PathArg) -> list[str]:
        return [p.name for p in Path(path).iterdir()]

    @staticmethod
    def glob(path: PathArg, pattern: str) -> Iterable[str]:
        for p in Path(path).glob(pattern):
            yield str(p)

    @staticmethod
    def rglob(path: PathArg, pattern: str) -> Iterable[str]:
        for p in Path(path).rglob(pattern):
            yield str(p)

    # ── Read ─────────────────────────────────────────────────────────────

    @staticmethod
    def read_text(path: PathArg, encoding: str = "utf-8") -> str:
        return Path(path).read_text(encoding=encoding)

    @staticmethod
    def read_bytes(path: PathArg) -> bytes:
        return Path(path).read_bytes()

    @staticmethod
    def open(path: PathArg, mode: str = "r", encoding: str = "utf-8") -> IO[Any]:
        return Path(path).open(mode=mode, encoding=encoding)  # type: ignore[return]

    # ── Write ────────────────────────────────────────────────────────────

    @staticmethod
    def write_text(path: PathArg, content: str, *, mode: int = 0o600) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        os.chmod(p, mode)  # noqa: PTH101

    @staticmethod
    def write_bytes(path: PathArg, content: bytes, *, mode: int = 0o600) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        os.chmod(p, mode)  # noqa: PTH101

    # ── Mutations ────────────────────────────────────────────────────────

    @staticmethod
    def rename(src: PathArg, dst: PathArg) -> None:
        os.rename(src, dst)  # noqa: PTH104

    @staticmethod
    def remove(path: PathArg, *, recursive: bool = False) -> None:
        p = Path(path)
        if not p.exists():
            return
        if recursive and p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    @staticmethod
    def copy(src: PathArg, dst: PathArg) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def copytree(src: PathArg, dst: PathArg, *, dirs_exist_ok: bool = False) -> None:
        shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)

    # ── Metadata ─────────────────────────────────────────────────────────

    @staticmethod
    def stat(path: PathArg) -> StatResult:
        p = Path(path)
        st = os.stat(p)  # noqa: PTH116
        return StatResult(
            size=st.st_size,
            mtime=st.st_mtime,
            is_dir=p.is_dir(),
            is_file=p.is_file(),
        )

    @staticmethod
    def lstat(path: PathArg) -> StatResult:
        p = Path(path)
        st = os.lstat(p)
        return StatResult(
            size=st.st_size,
            mtime=st.st_mtime,
            is_dir=p.is_dir(),
            is_file=p.is_file(),
        )

    @staticmethod
    def touch(path: PathArg) -> None:
        Path(path).touch()

    @staticmethod
    def chmod(path: PathArg, mode: int) -> None:
        os.chmod(path, mode)  # noqa: PTH101

    @staticmethod
    def getsize(path: PathArg) -> int:
        return os.path.getsize(path)  # noqa: PTH202

    # ── Symlinks ─────────────────────────────────────────────────────────

    @staticmethod
    def symlink(src: PathArg, dst: PathArg) -> None:
        Path(dst).symlink_to(src)

    # ── Atomic I/O ───────────────────────────────────────────────────────

    @staticmethod
    def atomic_write_json(path: PathArg, data: object) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
                fh.write("\n")
            os.chmod(tmp, 0o600)  # noqa: PTH101
            os.replace(tmp, p)  # noqa: PTH105
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)  # noqa: PTH108
            raise

    @staticmethod
    def atomic_write_text(path: PathArg, content: str, *, encoding: str = "utf-8") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=encoding) as fh:
                fh.write(content)
            os.chmod(tmp, 0o600)  # noqa: PTH101
            os.replace(tmp, p)  # noqa: PTH105
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)  # noqa: PTH108
            raise


# Verify structural conformance at import time — no runtime cost after first import.
_local_fs = LocalFileSystem()
assert isinstance(_local_fs, FileSystem), "LocalFileSystem must satisfy the FileSystem Protocol"
