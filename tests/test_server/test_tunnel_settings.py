"""CLI flags overlay ``molexp config`` ``tunnel.*``; env is never consulted."""

from __future__ import annotations

import pytest

from molexp.server.tunnel.base import http_origin
from molexp.server.tunnel.settings import TunnelError, resolve_tunnel_settings


class TestResolveTunnelSettings:
    def test_defaults_to_cloudflared_quick(self) -> None:
        settings = resolve_tunnel_settings(config={})
        assert settings.via == "cloudflared"
        assert settings.mode == "quick"
        assert settings.token is None
        assert settings.bin is None

    def test_config_via_zrok_defaults_public(self) -> None:
        settings = resolve_tunnel_settings(config={"tunnel": {"via": "zrok"}})
        assert settings.via == "zrok"
        assert settings.mode == "public"

    def test_cli_overrides_config(self) -> None:
        settings = resolve_tunnel_settings(
            via="zrok",
            mode="reserved",
            token="cli-token",
            bin="/opt/zrok",
            config={
                "tunnel": {
                    "via": "cloudflared",
                    "mode": "named",
                    "token": "file-token",
                    "bin": "/opt/cloudflared",
                }
            },
        )
        assert settings.via == "zrok"
        assert settings.mode == "reserved"
        assert settings.token == "cli-token"
        assert settings.bin == "/opt/zrok"

    def test_config_fills_token_and_bin(self) -> None:
        settings = resolve_tunnel_settings(
            config={
                "tunnel": {
                    "via": "cloudflared",
                    "mode": "named",
                    "token": "stored",
                    "bin": "/usr/local/bin/cloudflared",
                    "hostname": "lab.example.com",
                }
            }
        )
        assert settings.token == "stored"
        assert settings.bin == "/usr/local/bin/cloudflared"
        assert settings.hostname == "lab.example.com"

    def test_unknown_via_raises(self) -> None:
        with pytest.raises(TunnelError, match="--via"):
            resolve_tunnel_settings(via="ngrok", config={})

    def test_blank_config_strings_are_ignored(self) -> None:
        settings = resolve_tunnel_settings(
            config={"tunnel": {"via": "  ", "token": "", "bin": "   "}}
        )
        assert settings.via == "cloudflared"
        assert settings.token is None
        assert settings.bin is None

    def test_env_is_not_a_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOLEXP_TUNNEL", "1")
        monkeypatch.setenv("MOLEXP_TUNNEL_BIN", "/from/env")
        monkeypatch.setenv("CLOUDFLARED_TUNNEL_TOKEN", "env-token")
        settings = resolve_tunnel_settings(config={})
        assert settings.via == "cloudflared"
        assert settings.bin is None
        assert settings.token is None


class TestHttpOrigin:
    def test_ipv4(self) -> None:
        assert http_origin("127.0.0.1", 8000) == "http://127.0.0.1:8000"

    def test_brackets_raw_ipv6(self) -> None:
        assert http_origin("::1", 5173) == "http://[::1]:5173"

    def test_keeps_already_bracketed_ipv6(self) -> None:
        assert http_origin("[::1]", 5173) == "http://[::1]:5173"
