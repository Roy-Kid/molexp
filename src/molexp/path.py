"""Cross-host POSIX path with local-FS I/O delegation.

Subclasses :class:`pathlib.PurePosixPath` for pure path arithmetic (parts,
``/`` joining, ``parent`` / ``name`` / ``with_suffix`` …), and adds a
small set of I/O methods (:meth:`read_text`, :meth:`write_text`,
:meth:`exists`, :meth:`iterdir`, :meth:`mkdir`, …) that delegate to
:class:`pathlib.Path` for the **local host's** filesystem.

For cross-host workspaces (e.g. paths that conceptually live on an SSH-
mounted cluster), use :class:`molexp.workspace.fs.FileSystem` instead of
the I/O methods here — the FileSystem layer dispatches via ``molq``
Transport. The methods on this class always hit the local filesystem.

Why subclass instead of using ``pathlib.Path`` directly:
  1. Single project-internal type identity — ``isinstance(p, molexp.Path)``
     is a meaningful predicate.
  2. Single import surface — ``from molexp import Path`` instead of
     ``from pathlib import Path`` scattered across the codebase.
  3. Stable POSIX semantics on every host — ``pathlib.Path`` binds to
     ``PosixPath`` / ``WindowsPath`` per-host, which is wrong for paths
     that travel through workspace metadata.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from pathlib import PurePosixPath


class Path(PurePosixPath):
    """A POSIX path with local-FS I/O via :class:`pathlib.Path` delegation.

    Pure path arithmetic is inherited from :class:`pathlib.PurePosixPath`;
    the I/O methods (:meth:`read_text` / :meth:`write_text` /
    :meth:`exists` / …) forward to ``pathlib.Path(str(self))`` and act
    on the **local host's** filesystem. For non-local paths, use
    :class:`molexp.workspace.fs.FileSystem`.
    """

    __slots__ = ()

    def _local(self) -> pathlib.Path:
        return pathlib.Path(str(self))

    def read_text(self, encoding: str | None = "utf-8", errors: str | None = None) -> str:
        return self._local().read_text(encoding=encoding, errors=errors)

    def read_bytes(self) -> bytes:
        return self._local().read_bytes()

    def write_text(
        self,
        data: str,
        encoding: str | None = "utf-8",
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        return self._local().write_text(data, encoding=encoding, errors=errors, newline=newline)

    def write_bytes(self, data: bytes) -> int:
        return self._local().write_bytes(data)

    def exists(self) -> bool:
        return self._local().exists()

    def is_file(self) -> bool:
        return self._local().is_file()

    def is_dir(self) -> bool:
        return self._local().is_dir()

    def iterdir(self) -> Iterator[Path]:
        for child in self._local().iterdir():
            yield Path(str(child))

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        self._local().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def stat(self) -> os.stat_result:
        return self._local().stat()

    def unlink(self, missing_ok: bool = False) -> None:
        self._local().unlink(missing_ok=missing_ok)


__all__ = ["Path"]
