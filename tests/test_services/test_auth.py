"""Unit tests for molexp.services.auth (filesystem users + sessions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.services.auth import (
    AuthError,
    AuthService,
    reset_auth_service,
    set_auth_enabled,
    set_auth_root,
)
from molexp.services.auth.passwords import hash_password, verify_password
from molexp.services.auth.policy import can_write, workspace_allowed


@pytest.fixture(autouse=True)
def _isolate_auth(tmp_path: Path):
    set_auth_root(tmp_path / "auth")
    set_auth_enabled(False)
    yield
    reset_auth_service()


class TestPasswords:
    def test_hash_and_verify_roundtrip(self) -> None:
        h = hash_password("s3cret")
        assert h.startswith("$argon2")
        assert verify_password(h, "s3cret")
        assert not verify_password(h, "wrong")


class TestUserStore:
    def test_bootstrap_first_user_is_admin(self) -> None:
        svc = AuthService()
        user = svc.bootstrap_admin("pw", username="admin")
        assert user.role == "admin"
        assert user.username == "admin"
        assert svc.has_users()

    def test_create_second_user_respects_role(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        bob = svc.create_user("bob", "pw2", role="viewer")
        assert bob.role == "viewer"

    def test_cannot_delete_last_admin(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        with pytest.raises(AuthError, match="last admin"):
            svc.delete_user("admin")

    def test_duplicate_user_errors(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        with pytest.raises(AuthError):
            svc.create_user("admin", "other")


class TestSessions:
    def test_login_logout_roundtrip(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw", username="admin")
        user, session = svc.login("admin", "pw")
        assert user.username == "admin"
        assert svc.resolve_session(session.session_id) is not None
        svc.logout(session.session_id)
        assert svc.resolve_session(session.session_id) is None

    def test_bad_password(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        with pytest.raises(AuthError) as exc:
            svc.login("admin", "nope")
        assert exc.value.code == "invalid_credentials"

    def test_switch_revokes_old_session(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        svc.create_user("bob", "bobpw", role="operator")
        _, s1 = svc.login("admin", "pw")
        user, s2 = svc.switch(s1.session_id, "bob", "bobpw")
        assert user.username == "bob"
        assert svc.resolve_session(s1.session_id) is None
        assert svc.resolve_session(s2.session_id) is not None


class TestPolicy:
    def test_viewer_cannot_write(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        viewer = svc.create_user("v", "pw", role="viewer")
        assert not can_write(viewer)
        with pytest.raises(AuthError):
            svc.assert_can_write(viewer)

    def test_workspace_allowlist(self) -> None:
        svc = AuthService()
        svc.bootstrap_admin("pw")
        bob = svc.create_user("bob", "pw", role="operator", workspaces=["lab-a"])
        assert workspace_allowed(bob, "lab-a")
        assert not workspace_allowed(bob, "lab-b")
        assert svc.filter_workspaces(bob, ["lab-a", "lab-b", "lab-c"]) == ["lab-a"]


class TestRateLimit:
    def test_lockout_after_failures(self) -> None:
        from molexp.services.auth.rate_limit import LoginRateLimiter

        limiter = LoginRateLimiter(max_failures=3, window_seconds=60, lockout_seconds=60)
        svc = AuthService(rate_limiter=limiter)
        svc.bootstrap_admin("correct")
        for _ in range(3):
            with pytest.raises(AuthError) as exc:
                svc.login("admin", "wrong")
            assert exc.value.code == "invalid_credentials"
        with pytest.raises(AuthError) as exc:
            svc.login("admin", "wrong")
        assert exc.value.code == "rate_limited"
        # Even the correct password is blocked while locked out.
        with pytest.raises(AuthError) as exc:
            svc.login("admin", "correct")
        assert exc.value.code == "rate_limited"
