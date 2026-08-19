"""``archive_folder_zip`` — Folder directory → zip bytes (single writer).

The one workspace-layer zip archiver shared by agent export, server
``export_run``, and any future CLI. All I/O goes through the workspace
disk (``folder._disk()``)
(local and remote backends). Consumers import this module directly
(``from molexp.workspace.archive import archive_folder_zip``) — not
re-exported from ``molexp.workspace`` (same pattern as ``git_projection``).

Known debt (accepted):
* mode bits are not preserved (``ZipInfo.external_attr = 0``);
* whole archive is buffered in memory as ``bytes``;
* symlink traversal follows ``FileSystem.rglob`` semantics.
"""

from __future__ import annotations

import io
import time
import zipfile

from molexp.path import Path as MolexpPath

from .folder import Folder

__all__ = ["archive_folder_zip"]


def archive_folder_zip(folder: Folder) -> bytes:
    """Zip every file under *folder* into a DEFLATED in-memory archive.

    Args:
        folder: Any :class:`Folder` (Run, AgentSession, …). Uses
            :meth:`Folder.resolve` (no lazy mkdir) and the workspace disk.

    Returns:
        Zip file bytes. Missing or non-directory roots yield an empty but
        valid zip (no exception, no directory creation).
    """
    fs = folder._disk()
    root = str(folder.resolve())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if fs.is_dir(root):
            entries: list[tuple[tuple[str, ...], str, str]] = []
            for path in fs.rglob(root, "*"):
                if not fs.is_file(path):
                    continue
                arcname = MolexpPath(path).relative_to(root).as_posix()
                entries.append((MolexpPath(arcname).parts, arcname, path))

            for _parts, arcname, path in sorted(entries, key=lambda t: t[0]):
                data = fs.read_bytes(path)
                info = zipfile.ZipInfo(filename=arcname)
                info.compress_type = zipfile.ZIP_DEFLATED
                try:
                    mtime = fs.stat(path).mtime
                    info.date_time = time.localtime(mtime)[:6]
                except Exception:
                    pass
                zf.writestr(info, data)

    return buffer.getvalue()
