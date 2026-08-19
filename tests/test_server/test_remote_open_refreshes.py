"""Remote open must block on a force-refresh of the local pin."""

from __future__ import annotations

from unittest.mock import MagicMock

from molexp.server.deps.resolution import _ensure_remote_ready
from molexp.workspace.fs_cached import CachedRemoteFileSystem


def test_ensure_remote_ready_force_refreshes_async() -> None:
    """Linking a remote root always force-refreshes (async for progress bar).

    ``block_index=False`` so the UI can poll GET /cache/status while the
    recursive file-count + fetch walk runs.
    """
    fs = MagicMock(spec=CachedRemoteFileSystem)
    ws = MagicMock()
    ws.fs = fs

    _ensure_remote_ready(ws)

    fs.prepare.assert_called_once_with(ws, block_index=False, refresh_on_open=True)


def test_ensure_remote_ready_skips_local_fs() -> None:
    ws = MagicMock()
    ws.fs = object()  # not CachedRemoteFileSystem
    _ensure_remote_ready(ws)  # no raise
