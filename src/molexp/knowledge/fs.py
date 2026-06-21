"""Pluggable filesystem seam for ``molexp.knowledge`` (okf-05-04).

`Folder` / `Library` route every disk operation through a :class:`FileSystem`
so a bundle can live on a remote/HPC backend (the parity gap that blocks the
agent rehome). :class:`LocalFileSystem` is the default and reproduces the
pre-seam behavior exactly (stdlib :mod:`pathlib` + :mod:`molexp.atomicio`), so
existing callers are unaffected.

A transitional in-layer duplicate of ``molexp.workspace.fs`` (knowledge cannot
import its peer); it disappears when ``workspace`` is removed. Concrete remote
backends are out of scope here — an upstream layer injects its own Protocol
implementation.
"""

from __future__ import annotations

import json
import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable

import molexp.atomicio as atomicio


@runtime_checkable
class FileSystem(Protocol):
    """The disk operations a knowledge bundle needs (path = identity)."""

    def read_text(self, path: Path) -> str: ...
    def write_text(self, path: Path, content: str) -> None: ...
    def append_text(self, path: Path, content: str) -> None: ...
    def read_bytes(self, path: Path) -> bytes: ...
    def write_bytes(self, path: Path, data: bytes) -> None: ...
    def remove(self, path: Path) -> None: ...
    def read_json(self, path: Path) -> object: ...
    def write_json(self, path: Path, data: object) -> None: ...
    def exists(self, path: Path) -> bool: ...
    def is_file(self, path: Path) -> bool: ...
    def is_dir(self, path: Path) -> bool: ...
    def mkdir(self, path: Path) -> None: ...
    def iterdir(self, path: Path) -> list[Path]: ...
    def rmtree(self, path: Path) -> None: ...
    def lock(self, path: Path) -> AbstractContextManager[object]: ...


class LocalFileSystem:
    """Default :class:`FileSystem`: stdlib ``pathlib`` + ``molexp.atomicio``."""

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        atomicio.atomic_write_text(Path(path), content)

    def append_text(self, path: Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)

    def read_bytes(self, path: Path) -> bytes:
        return Path(path).read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    def remove(self, path: Path) -> None:
        Path(path).unlink(missing_ok=True)

    def read_json(self, path: Path) -> object:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: Path, data: object) -> None:
        atomicio.atomic_write_json(Path(path), data)

    def exists(self, path: Path) -> bool:
        return Path(path).exists()

    def is_file(self, path: Path) -> bool:
        return Path(path).is_file()

    def is_dir(self, path: Path) -> bool:
        return Path(path).is_dir()

    def mkdir(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def iterdir(self, path: Path) -> list[Path]:
        return list(Path(path).iterdir())

    def rmtree(self, path: Path) -> None:
        shutil.rmtree(Path(path))

    def lock(self, path: Path) -> AbstractContextManager[object]:
        return atomicio.file_lock(Path(path))


__all__ = ["FileSystem", "LocalFileSystem"]
