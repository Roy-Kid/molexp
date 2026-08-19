"""The user-facing byte-exit under one directory root.

A :class:`FileStore` is *files in a place*. It does not own a disk: the
:class:`~molexp.workspace.fs.FileSystem` (the disk) is injected, usually
from :attr:`Workspace.fs`. Relative paths only; escape is rejected.

``put`` is atomic (temp + rename on the local disk; the FileSystem
atomic helpers on a remote). Catalog registration is orthogonal.
"""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path, PurePosixPath

from molexp.atomicio import atomic_write_bytes, atomic_write_json, atomic_write_text

from .fs import FileSystem
from .fs_local import LocalFileSystem

_PutData = Path | bytes | bytearray | dict | list | str


class FileStore:
    """Write files under *root* on *fs*. Relative paths only."""

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        fs: FileSystem | None = None,
    ) -> None:
        self._root = fspath(root)
        self._fs: FileSystem = fs if fs is not None else LocalFileSystem()

    @property
    def root(self) -> str:
        return self._root

    def resolve(self, relpath: str | Path) -> Path:
        """Return ``root/relpath`` if it stays inside *root*."""
        rel = PurePosixPath(fspath(relpath))
        if rel.is_absolute():
            raise ValueError(f"FileStore: path must be relative, got {relpath!r}")
        if ".." in rel.parts:
            raise ValueError(f"FileStore: path {relpath!r} escapes root {self._root}")
        return Path(self._fs.join(self._root, str(rel)))

    def mkdir(self, relpath: str | Path) -> Path:
        """Create *relpath* under the root and return a local :class:`~pathlib.Path`."""
        target = self.resolve(relpath)
        self._fs.mkdir(str(target), parents=True, exist_ok=True)
        return target

    def put(self, relpath: str | Path, data: _PutData) -> Path:
        """Atomically write *data* at *relpath*. Returns the destination path."""
        target = self.resolve(relpath)
        if isinstance(self._fs, LocalFileSystem):
            if isinstance(data, (bytes, bytearray)):
                atomic_write_bytes(target, bytes(data))
            elif isinstance(data, Path):
                src = Path(data)
                try:
                    already = target.exists() and src.resolve().samefile(target.resolve())
                except OSError:
                    already = False
                if not already:
                    atomic_write_bytes(target, src.read_bytes())
            elif isinstance(data, (dict, list)):
                atomic_write_json(target, data)
            else:
                atomic_write_text(target, str(data))
            return target
        self._put_via_fs(str(target), data)
        return target

    def _put_via_fs(self, target: str, data: _PutData) -> None:
        disk = self._fs
        parent = disk.dirname(target)
        if parent:
            disk.mkdir(parent, parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            disk.write_bytes(target, bytes(data))
        elif isinstance(data, Path):
            disk.write_bytes(target, data.read_bytes())
        elif isinstance(data, (dict, list)):
            disk.atomic_write_json(target, data)
        else:
            disk.atomic_write_text(target, str(data))

    def append(self, relpath: str | Path, line: str) -> Path:
        """Append *line* (a trailing newline is added if missing)."""
        target = self.resolve(relpath)
        payload = line if line.endswith("\n") else line + "\n"
        if isinstance(self._fs, LocalFileSystem):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(payload)
            return target
        dest = str(target)
        existing = self._fs.read_text(dest) if self._fs.exists(dest) else ""
        self._fs.atomic_write_text(dest, existing + payload)
        return target
