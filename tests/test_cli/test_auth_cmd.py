"""CLI ``molexp auth`` (gh-shaped) + serve auth preflight."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from molexp.cli import app
from molexp.cli.workspace._serve_auth import configure_serve_auth, is_loopback_host
from molexp.services.auth import (
    reset_auth_service,
    set_auth_enabled,
    set_auth_root,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "auth"
    set_auth_root(root)
    set_auth_enabled(False)
    monkeypatch.setenv("MOLEXP_AUTH_PASSWORD", "test-pass")
    yield
    reset_auth_service()


class TestLoopback:
    def test_localhost_is_loopback(self) -> None:
        assert is_loopback_host("localhost")
        assert is_loopback_host("127.0.0.1")

    def test_all_interfaces_not_loopback(self) -> None:
        assert not is_loopback_host("0.0.0.0")


class TestAuthLoginCli:
    def test_login_bootstraps_admin(self) -> None:
        result = runner.invoke(app, ["auth", "login", "-u", "admin"])
        assert result.exit_code == 0, result.output
        assert "created user" in result.output or "logged in" in result.output

        listed = runner.invoke(app, ["auth", "users", "list"])
        assert listed.exit_code == 0
        assert "admin" in listed.output

    def test_status_after_login(self) -> None:
        runner.invoke(app, ["auth", "login", "-u", "admin"])
        status = runner.invoke(app, ["auth", "status"])
        assert status.exit_code == 0
        assert "admin" in status.output

    def test_token_prints_session(self) -> None:
        runner.invoke(app, ["auth", "login", "-u", "admin"])
        token = runner.invoke(app, ["auth", "token"])
        assert token.exit_code == 0
        assert token.output.strip()  # non-empty token


class TestServeAuthPreflight:
    def test_non_loopback_without_auth_refused(self) -> None:
        with pytest.raises(typer.Exit):
            configure_serve_auth(host="0.0.0.0", auth_flag=False, require_user=None)

    def test_auth_without_users_refused(self) -> None:
        with pytest.raises(typer.Exit):
            configure_serve_auth(host="localhost", auth_flag=True, require_user=None)

    def test_auth_with_user_ok(self) -> None:
        runner.invoke(app, ["auth", "login", "-u", "admin"])
        configure_serve_auth(host="localhost", auth_flag=True, require_user="admin")
        from molexp.services.auth import is_auth_enabled

        assert is_auth_enabled() is True

    def test_auth_require_missing_user_fails(self) -> None:
        runner.invoke(app, ["auth", "login", "-u", "admin"])
        with pytest.raises(typer.Exit):
            configure_serve_auth(host="localhost", auth_flag=True, require_user="ghost")
