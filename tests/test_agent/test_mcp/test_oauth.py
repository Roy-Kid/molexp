"""Unit tests for ``molexp.agent.mcp.oauth`` — MCP OAuth integration helpers."""

from __future__ import annotations

import asyncio
import os
import stat
import sys

import pytest

from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.oauth import (
    FileTokenStorage,
    OAuthFlowSession,
    OAuthSessionRegistry,
    storage_for,
)
from molexp.agent.mcp.store import McpScope, McpStore


@pytest.fixture
def isolated_user_dir(tmp_path, monkeypatch):
    """Redirect ``USER_DIR`` so user-scope writes land in tmp."""
    user_dir = tmp_path / "user_home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", user_dir)
    return user_dir


@pytest.fixture
def store(tmp_path, isolated_user_dir):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    return McpStore(workspace_root)


class TestFileTokenStorage:
    """JSON-backed ``TokenStorage`` — one file per ``(scope, server)`` pair."""

    def test_round_trips_tokens(self, tmp_path):
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage(tmp_path, "srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="abc",
                    token_type="Bearer",
                    refresh_token="rrr",
                    expires_in=3600,
                )
            )
        )
        assert storage.path.exists()
        got = asyncio.run(storage.get_tokens())
        assert got is not None
        assert got.access_token == "abc"
        assert got.refresh_token == "rrr"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
    def test_chmods_token_file_to_owner_only(self, tmp_path):
        """Refresh tokens must not leak to other local users: file mode is 0o600."""
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage(tmp_path, "srv")
        asyncio.run(storage.set_tokens(OAuthToken(access_token="x", token_type="Bearer")))
        mode = stat.S_IMODE(os.stat(storage.path).st_mode)  # noqa: PTH116
        assert mode == 0o600

    def test_corrupt_file_returns_none(self, tmp_path):
        storage = FileTokenStorage(tmp_path, "srv")
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        storage.path.write_text("not json at all {{")
        assert asyncio.run(storage.get_tokens()) is None
        assert asyncio.run(storage.get_client_info()) is None


class TestOAuthFlowSession:
    """Futures-based bridge between SDK callbacks and HTTP request handlers."""

    def test_callback_round_trips(self):
        sess = OAuthFlowSession()
        sess.submit_callback("CODE", "STATE")
        code, state = asyncio.run(sess.callback_handler())
        assert code == "CODE"
        assert state == "STATE"

    def test_duplicate_callback_returns_false(self):
        sess = OAuthFlowSession()
        assert sess.submit_callback("CODE", None) is True
        assert sess.submit_callback("CODE-2", None) is False


class TestOAuthSessionRegistry:
    """Process-wide registry of in-flight OAuth flows keyed by ``(scope, name)``."""

    def test_create_supersedes_and_cancels_existing(self):
        reg = OAuthSessionRegistry()
        s1 = reg.create("workspace", "srv")
        s2 = reg.create("workspace", "srv")
        assert reg.get("workspace", "srv") is s2
        # The superseded session is cancelled; the fresh one is not.
        assert s1.cancelled is True
        assert s2.cancelled is False


class TestStorageFor:
    """``storage_for`` — the per-scope ``FileTokenStorage`` factory."""

    def test_workspace_scope_uses_workspace_root(self, store, tmp_path):
        storage = storage_for(store, McpScope.WORKSPACE, "srv")
        assert storage.path.is_relative_to(tmp_path / "workspace" / ".mcp_oauth")
