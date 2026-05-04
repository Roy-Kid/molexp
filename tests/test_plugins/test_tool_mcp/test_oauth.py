"""Unit tests for the MCP OAuth integration helpers."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys

import pytest

from molexp.plugins.tool_mcp import store as mcp_mod
from molexp.plugins.tool_mcp.oauth import (
    FileTokenStorage,
    OAuthFlowSession,
    OAuthSessionRegistry,
    storage_for,
)
from molexp.plugins.tool_mcp.store import McpScope, McpStore


@pytest.fixture
def isolated_user_dir(tmp_path, monkeypatch):
    """Redirect USER_DIR so user-scope writes land in tmp."""
    user_dir = tmp_path / "user_home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", user_dir)
    return user_dir


@pytest.fixture
def store(tmp_path, isolated_user_dir):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    return McpStore(workspace_root)


# ── FileTokenStorage ───────────────────────────────────────────────────────


def test_file_token_storage_get_returns_none_when_empty(tmp_path):
    storage = FileTokenStorage(tmp_path, "srv")
    assert asyncio.run(storage.get_tokens()) is None
    assert asyncio.run(storage.get_client_info()) is None


def test_file_token_storage_round_trips_tokens(tmp_path):
    from mcp.shared.auth import OAuthToken

    storage = FileTokenStorage(tmp_path, "srv")
    tok = OAuthToken(
        access_token="abc",
        token_type="Bearer",
        refresh_token="rrr",
        expires_in=3600,
    )
    asyncio.run(storage.set_tokens(tok))
    assert storage.path.exists()
    got = asyncio.run(storage.get_tokens())
    assert got is not None
    assert got.access_token == "abc"
    assert got.refresh_token == "rrr"


def test_file_token_storage_round_trips_client_info(tmp_path):
    from mcp.shared.auth import OAuthClientInformationFull

    storage = FileTokenStorage(tmp_path, "srv")
    info = OAuthClientInformationFull(
        client_id="cid",
        redirect_uris=["http://localhost/cb"],  # type: ignore[arg-type]
    )
    asyncio.run(storage.set_client_info(info))
    got = asyncio.run(storage.get_client_info())
    assert got is not None
    assert got.client_id == "cid"


def test_file_token_storage_clear_returns_true_when_present(tmp_path):
    from mcp.shared.auth import OAuthToken

    storage = FileTokenStorage(tmp_path, "srv")
    asyncio.run(storage.set_tokens(OAuthToken(access_token="x", token_type="Bearer")))
    assert storage.clear() is True
    assert storage.clear() is False  # idempotent


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_file_token_storage_chmods_to_owner_only(tmp_path):
    from mcp.shared.auth import OAuthToken

    storage = FileTokenStorage(tmp_path, "srv")
    asyncio.run(storage.set_tokens(OAuthToken(access_token="x", token_type="Bearer")))
    mode = stat.S_IMODE(os.stat(storage.path).st_mode)
    assert mode == 0o600


def test_file_token_storage_corrupt_file_returns_none(tmp_path):
    storage = FileTokenStorage(tmp_path, "srv")
    storage.path.parent.mkdir(parents=True, exist_ok=True)
    storage.path.write_text("not json at all {{")
    assert asyncio.run(storage.get_tokens()) is None
    assert asyncio.run(storage.get_client_info()) is None


def test_file_token_storage_invalid_token_payload_returns_none(tmp_path):
    storage = FileTokenStorage(tmp_path, "srv")
    storage.path.parent.mkdir(parents=True, exist_ok=True)
    storage.path.write_text(json.dumps({"tokens": {"unexpected": "shape"}}))
    assert asyncio.run(storage.get_tokens()) is None


# ── OAuthFlowSession ───────────────────────────────────────────────────────


def test_flow_session_callback_round_trips():
    sess = OAuthFlowSession()
    sess.submit_callback("CODE", "STATE")
    code, state = asyncio.run(sess.callback_handler())
    assert code == "CODE"
    assert state == "STATE"


def test_flow_session_redirect_handler_resolves_future():
    sess = OAuthFlowSession()

    async def go():
        await sess.redirect_handler("https://idp/authorize")
        return await sess.authorize_url_future

    assert asyncio.run(go()) == "https://idp/authorize"


def test_flow_session_duplicate_callback_returns_false():
    sess = OAuthFlowSession()
    assert sess.submit_callback("CODE", None) is True
    assert sess.submit_callback("CODE-2", None) is False


def test_flow_session_cancel_does_not_raise_on_resolved_future():
    sess = OAuthFlowSession()
    sess.submit_callback("c", "s")
    sess.cancel()  # idempotent


# ── OAuthSessionRegistry ───────────────────────────────────────────────────


def test_registry_create_supersedes_existing():
    reg = OAuthSessionRegistry()
    s1 = reg.create("workspace", "srv")
    s2 = reg.create("workspace", "srv")
    assert reg.get("workspace", "srv") is s2
    # The superseded session is marked cancelled — any subsequent awaiter
    # of either of its futures will see CancelledError on first access.
    assert s1.cancelled is True
    assert s2.cancelled is False


def test_registry_get_returns_none_for_unknown():
    reg = OAuthSessionRegistry()
    assert reg.get("workspace", "no-such") is None


def test_registry_discard_is_idempotent():
    reg = OAuthSessionRegistry()
    reg.create("workspace", "srv")
    reg.discard("workspace", "srv")
    reg.discard("workspace", "srv")  # second call must not raise


# ── storage_for ────────────────────────────────────────────────────────────


def test_storage_for_workspace_uses_workspace_root(store, tmp_path):
    storage = storage_for(store, McpScope.WORKSPACE, "srv")
    assert storage.path.is_relative_to(tmp_path / "workspace" / ".mcp_oauth")


def test_storage_for_user_uses_user_dir(store, isolated_user_dir):
    storage = storage_for(store, McpScope.USER, "srv")
    assert storage.path.is_relative_to(isolated_user_dir / ".mcp_oauth")
