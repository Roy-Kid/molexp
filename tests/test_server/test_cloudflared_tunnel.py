"""cloudflared tunnel helper — parse URL + spawn wiring (mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from molexp.server.tunnel.cloudflared import (
    CloudflaredTunnel,
    TunnelError,
    parse_quick_tunnel_url,
    resolve_cloudflared_bin,
)


class TestParseQuickTunnelUrl:
    def test_extracts_trycloudflare(self) -> None:
        line = "2024-01-01 INF |  https://random-words-here.trycloudflare.com"
        assert parse_quick_tunnel_url(line) == "https://random-words-here.trycloudflare.com"

    def test_none_without_match(self) -> None:
        assert parse_quick_tunnel_url("starting tunnel") is None


class TestResolveBin:
    def test_missing_raises_without_downloading(self) -> None:
        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch(
                "molexp.server.tunnel.fetch.bin_dir",
                return_value=Path("/no/such/local-bin"),
            ),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared") as fetch,
            pytest.raises(TunnelError, match="cloudflared"),
        ):
            resolve_cloudflared_bin()
        fetch.assert_not_called()

    def test_uses_cached_binary(self, tmp_path: Path) -> None:
        cached = tmp_path / "cloudflared"
        cached.write_text("#!/bin/sh\n")
        cached.chmod(0o755)
        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch("molexp.server.tunnel.fetch.bin_dir", return_value=tmp_path),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared") as fetch,
        ):
            assert resolve_cloudflared_bin() == str(cached.resolve())
            fetch.assert_not_called()

    def test_explicit_path(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "cloudflared"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        assert resolve_cloudflared_bin(bin_path) == str(bin_path.resolve())

    def test_env_is_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        decoy = tmp_path / "decoy"
        decoy.write_text("#!/bin/sh\n")
        decoy.chmod(0o755)
        monkeypatch.setenv("MOLEXP_TUNNEL_BIN", str(decoy))
        with patch(
            "molexp.server.tunnel.base.shutil.which",
            return_value="/usr/bin/cloudflared",
        ):
            assert resolve_cloudflared_bin() == "/usr/bin/cloudflared"


class TestCloudflaredTunnelStart:
    def test_quick_command(self) -> None:
        fake = MagicMock()
        fake.stdout = iter(
            [
                "INF Thank you for trying Cloudflare Tunnel\n",
                "INF |  https://demo-share.trycloudflare.com\n",
            ]
        )
        fake.poll.return_value = None
        with (
            patch(
                "molexp.server.tunnel.cloudflared.resolve_cloudflared_bin",
                return_value="/usr/bin/cloudflared",
            ),
            patch(
                "molexp.server.tunnel.base.subprocess.Popen",
                return_value=fake,
            ) as popen,
        ):
            urls: list[str] = []
            t = CloudflaredTunnel(local_port=8000, mode="quick", on_url=urls.append)
            t.start()
            # drive reader synchronously (thread may race — also set via wait)
            t._read_output()
        cmd = popen.call_args[0][0]
        assert cmd[0] == "/usr/bin/cloudflared"
        assert "tunnel" in cmd
        assert "--url" in cmd
        assert "http://127.0.0.1:8000" in cmd
        assert t.public_url == "https://demo-share.trycloudflare.com"
        assert urls == ["https://demo-share.trycloudflare.com"]

    def test_quick_command_uses_local_host(self) -> None:
        """``--url`` is the origin the caller asked the tunnel to dial."""
        fake = MagicMock()
        fake.stdout = iter(())
        fake.poll.return_value = None
        with (
            patch(
                "molexp.server.tunnel.cloudflared.resolve_cloudflared_bin",
                return_value="/usr/bin/cloudflared",
            ),
            patch("molexp.server.tunnel.base.subprocess.Popen", return_value=fake) as popen,
        ):
            CloudflaredTunnel(local_port=5173, mode="quick", local_host="localhost").start()
        cmd = popen.call_args[0][0]
        assert "http://localhost:5173" in cmd

    def test_named_requires_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLOUDFLARED_TUNNEL_TOKEN", "from-env")
        with (
            patch(
                "molexp.server.tunnel.cloudflared.resolve_cloudflared_bin",
                return_value="/usr/bin/cloudflared",
            ),
            pytest.raises(TunnelError, match="token"),
        ):
            CloudflaredTunnel(local_port=8000, mode="named", hostname="lab.example.com")

    def test_named_command_uses_token(self) -> None:
        fake = MagicMock()
        fake.stdout = iter([])
        fake.poll.return_value = None
        with (
            patch(
                "molexp.server.tunnel.cloudflared.resolve_cloudflared_bin",
                return_value="/usr/bin/cloudflared",
            ),
            patch(
                "molexp.server.tunnel.base.subprocess.Popen",
                return_value=fake,
            ) as popen,
        ):
            t = CloudflaredTunnel(
                local_port=8000,
                mode="named",
                hostname="lab.example.com",
                token="secret-token",
            )
            t.start()
        cmd = popen.call_args[0][0]
        assert cmd == [
            "/usr/bin/cloudflared",
            "tunnel",
            "run",
            "--token",
            "secret-token",
        ]
        assert t.public_url == "https://lab.example.com"
