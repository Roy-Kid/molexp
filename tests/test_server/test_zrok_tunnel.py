"""zrok tunnel helper — parse URL + spawn wiring (mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from molexp.server.tunnel import open_tunnel, resolve_tunnel_settings
from molexp.server.tunnel.zrok import (
    TunnelError,
    ZrokTunnel,
    parse_zrok_share_url,
    resolve_zrok_bin,
)


class TestParseZrokShareUrl:
    def test_extracts_hosted_frontend(self) -> None:
        line = "[   0.4]    INFO main.(*sharePublicCommand).run: https://vcikdowjf9uv.share.zrok.io"
        assert parse_zrok_share_url(line) == "https://vcikdowjf9uv.share.zrok.io"

    def test_none_without_match(self) -> None:
        assert parse_zrok_share_url("sharing target: http://127.0.0.1:8000") is None


class TestResolveZrokBin:
    def test_prefers_zrok_then_zrok2(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/zrok2" if name == "zrok2" else None

        with patch("molexp.server.tunnel.base.shutil.which", side_effect=which):
            assert resolve_zrok_bin() == "/usr/bin/zrok2"

    def test_explicit_path(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "zrok"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        assert resolve_zrok_bin(bin_path) == str(bin_path.resolve())

    def test_missing_raises_without_downloading(self) -> None:
        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch(
                "molexp.server.tunnel.fetch.bin_dir",
                return_value=Path("/no/such/local-bin"),
            ),
            patch("molexp.server.tunnel.fetch.fetch_zrok") as fetch,
            pytest.raises(TunnelError, match="zrok"),
        ):
            resolve_zrok_bin()
        fetch.assert_not_called()


class TestZrokTunnelStart:
    def test_public_command(self) -> None:
        fake = MagicMock()
        fake.stdout = iter(
            [
                "INFO sharing target: 'http://127.0.0.1:8000'\n",
                "INFO https://demo123.share.zrok.io\n",
            ]
        )
        fake.poll.return_value = None
        with (
            patch(
                "molexp.server.tunnel.zrok.resolve_zrok_bin",
                return_value="/usr/bin/zrok",
            ),
            patch(
                "molexp.server.tunnel.base.subprocess.Popen",
                return_value=fake,
            ) as popen,
        ):
            urls: list[str] = []
            t = ZrokTunnel(local_port=8000, mode="public", on_url=urls.append)
            t.start()
            t._read_output()
        assert popen.call_args[0][0] == [
            "/usr/bin/zrok",
            "share",
            "public",
            "http://127.0.0.1:8000",
            "--headless",
        ]
        assert t.public_url == "https://demo123.share.zrok.io"
        assert urls == ["https://demo123.share.zrok.io"]

    def test_reserved_requires_token(self) -> None:
        with (
            patch(
                "molexp.server.tunnel.zrok.resolve_zrok_bin",
                return_value="/usr/bin/zrok",
            ),
            pytest.raises(TunnelError, match="token"),
        ):
            ZrokTunnel(local_port=8000, mode="reserved")

    def test_reserved_command_uses_token(self) -> None:
        fake = MagicMock()
        fake.stdout = iter([])
        fake.poll.return_value = None
        with (
            patch(
                "molexp.server.tunnel.zrok.resolve_zrok_bin",
                return_value="/usr/bin/zrok",
            ),
            patch(
                "molexp.server.tunnel.base.subprocess.Popen",
                return_value=fake,
            ) as popen,
        ):
            t = ZrokTunnel(local_port=8000, mode="reserved", token="sharetok")
            t.start()
        assert popen.call_args[0][0] == [
            "/usr/bin/zrok",
            "share",
            "reserved",
            "sharetok",
            "--headless",
        ]


class TestOpenTunnelFactory:
    def test_via_zrok(self) -> None:
        settings = resolve_tunnel_settings(via="zrok", config={})
        with patch(
            "molexp.server.tunnel.zrok.resolve_zrok_bin",
            return_value="/usr/bin/zrok",
        ):
            handle = open_tunnel(local_port=9000, settings=settings)
        assert isinstance(handle, ZrokTunnel)
        assert handle.binary == "/usr/bin/zrok"
