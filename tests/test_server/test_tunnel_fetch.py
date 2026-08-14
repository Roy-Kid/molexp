"""User-space fetch of official tunnel clients (no network)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from molexp.server.tunnel.base import TunnelError, locate_tunnel_bin
from molexp.server.tunnel.fetch import (
    _extract_member,
    ensure_tunnel_client,
    fetch_cloudflared,
    fetch_zrok,
    github_latest_asset,
    interpret_download_reply,
    platform_pair,
)


def _asset_response(assets: list[dict[str, str]]) -> MagicMock:
    body = json.dumps({"assets": assets}).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestPlatformPair:
    def test_linux_amd64(self) -> None:
        with (
            patch("molexp.server.tunnel.fetch.sys_platform", return_value="linux"),
            patch("molexp.server.tunnel.fetch.platform.machine", return_value="x86_64"),
        ):
            assert platform_pair() == ("linux", "amd64")

    def test_linux_arm64_gh200(self) -> None:
        with (
            patch("molexp.server.tunnel.fetch.sys_platform", return_value="linux"),
            patch("molexp.server.tunnel.fetch.platform.machine", return_value="aarch64"),
        ):
            assert platform_pair() == ("linux", "arm64")

    def test_unknown_cpu(self) -> None:
        with (
            patch("molexp.server.tunnel.fetch.sys_platform", return_value="linux"),
            patch("molexp.server.tunnel.fetch.platform.machine", return_value="ppc64le"),
            pytest.raises(TunnelError, match="CPU"),
        ):
            platform_pair()


class TestGithubLatestAsset:
    def test_picks_matching_asset(self) -> None:
        resp = _asset_response(
            [
                {
                    "name": "cloudflared-linux-amd64.deb",
                    "browser_download_url": "https://example/deb",
                },
                {
                    "name": "cloudflared-linux-amd64",
                    "browser_download_url": "https://example/bin",
                },
            ]
        )
        urlopen = MagicMock(return_value=resp)
        url = github_latest_asset(
            "cloudflare/cloudflared",
            lambda name: name == "cloudflared-linux-amd64",
            urlopen=urlopen,
        )
        assert url == "https://example/bin"

    def test_none_match_raises(self) -> None:
        resp = _asset_response([{"name": "notes.txt", "browser_download_url": "https://example/n"}])
        with pytest.raises(TunnelError, match=r"no .* asset"):
            github_latest_asset(
                "cloudflare/cloudflared",
                lambda name: name == "cloudflared-linux-amd64",
                urlopen=MagicMock(return_value=resp),
            )


class TestExtractMember:
    def test_tar_gz_picks_named_binary(self, tmp_path: Path) -> None:
        archive = tmp_path / "zrok_1.0.0_linux_amd64.tar.gz"
        dest = tmp_path / "zrok"
        payload = b"#!/bin/sh\necho zrok\n"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo(name="zrok_1.0.0/zrok")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        _extract_member(archive, dest, ("zrok", "zrok2"))
        assert dest.read_bytes() == payload
        assert dest.stat().st_mode & 0o111


class TestFetchCloudflared:
    def test_writes_linux_binary(self, tmp_path: Path) -> None:
        dest = tmp_path / "cloudflared"
        payload = b"cf-bin"
        list_resp = _asset_response(
            [
                {
                    "name": "cloudflared-linux-amd64",
                    "browser_download_url": "https://example/cloudflared-linux-amd64",
                }
            ]
        )
        bin_resp = MagicMock()
        bin_resp.read.return_value = payload
        bin_resp.__enter__.return_value = bin_resp
        bin_resp.__exit__.return_value = False

        def urlopen(req: object, timeout: float = 0) -> MagicMock:
            url = getattr(req, "full_url", str(req))
            if "api.github.com" in url:
                return list_resp
            return bin_resp

        with (
            patch("molexp.server.tunnel.fetch.sys_platform", return_value="linux"),
            patch("molexp.server.tunnel.fetch.platform.machine", return_value="x86_64"),
        ):
            fetch_cloudflared(dest, urlopen=urlopen)
        assert dest.read_bytes() == payload
        assert dest.stat().st_mode & 0o111


class TestFetchZrok:
    def test_extracts_from_tarball(self, tmp_path: Path) -> None:
        dest = tmp_path / "zrok"
        payload = b"#!/bin/sh\necho zrok\n"
        tarball = tmp_path / "dl.tar.gz"
        with tarfile.open(tarball, "w:gz") as tf:
            info = tarfile.TarInfo(name="zrok")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        list_resp = _asset_response(
            [
                {
                    "name": "zrok_1.2.3_linux_amd64.tar.gz",
                    "browser_download_url": "https://example/zrok_1.2.3_linux_amd64.tar.gz",
                }
            ]
        )

        def urlopen(req: object, timeout: float = 0) -> MagicMock:
            url = getattr(req, "full_url", str(req))
            if "api.github.com" in url:
                return list_resp
            resp = MagicMock()
            resp.read.return_value = tarball.read_bytes()
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            return resp

        with (
            patch("molexp.server.tunnel.fetch.sys_platform", return_value="linux"),
            patch("molexp.server.tunnel.fetch.platform.machine", return_value="x86_64"),
        ):
            fetch_zrok(dest, urlopen=urlopen)
        assert dest.read_bytes() == payload


class TestInterpretDownloadReply:
    def test_yes_uses_default_dir(self, tmp_path: Path) -> None:
        assert interpret_download_reply("y", binary="cloudflared", default_dir=tmp_path) == (
            tmp_path / "cloudflared"
        )
        assert interpret_download_reply("Yes", binary="zrok", default_dir=tmp_path) == (
            tmp_path / "zrok"
        )

    def test_no_or_empty_declines(self, tmp_path: Path) -> None:
        assert interpret_download_reply("", binary="cloudflared", default_dir=tmp_path) is None
        assert interpret_download_reply("n", binary="cloudflared", default_dir=tmp_path) is None
        assert interpret_download_reply("N", binary="cloudflared", default_dir=tmp_path) is None
        assert interpret_download_reply("no", binary="cloudflared", default_dir=tmp_path) is None

    def test_custom_directory_appends_binary(self, tmp_path: Path) -> None:
        dest = interpret_download_reply(
            str(tmp_path / "tools"),
            binary="cloudflared",
            default_dir=tmp_path,
        )
        assert dest == tmp_path / "tools" / "cloudflared"

    def test_custom_file_path_kept(self, tmp_path: Path) -> None:
        file_path = tmp_path / "opt" / "cloudflared"
        dest = interpret_download_reply(
            str(file_path),
            binary="cloudflared",
            default_dir=tmp_path,
        )
        assert dest == file_path

    def test_trailing_slash_is_directory(self, tmp_path: Path) -> None:
        dest = interpret_download_reply(
            str(tmp_path) + "/",
            binary="zrok",
            default_dir=tmp_path,
        )
        assert dest == tmp_path / "zrok"


class TestEnsureAsksFirst:
    def test_uses_existing_without_asking(self, tmp_path: Path) -> None:
        cached = tmp_path / "cloudflared"
        cached.write_text("#!/bin/sh\n")
        cached.chmod(0o755)
        asked: list[tuple[str, Path]] = []

        def ask(binary: str, default_dir: Path) -> Path | None:
            asked.append((binary, default_dir))
            return default_dir / binary

        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch("molexp.server.tunnel.fetch.bin_dir", return_value=tmp_path),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared") as fetch,
        ):
            path = ensure_tunnel_client(
                via="cloudflared",
                explicit=None,
                ask=ask,
            )
        assert path == str(cached.resolve())
        assert asked == []
        fetch.assert_not_called()

    def test_declined_does_not_download(self, tmp_path: Path) -> None:
        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch("molexp.server.tunnel.fetch.bin_dir", return_value=tmp_path),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared") as fetch,
            pytest.raises(TunnelError, match="declined"),
        ):
            ensure_tunnel_client(
                via="cloudflared",
                explicit=None,
                ask=lambda _binary, _default: None,
            )
        fetch.assert_not_called()

    def test_yes_downloads_to_local_bin(self, tmp_path: Path) -> None:
        def fake_fetch(dest: Path) -> None:
            dest.write_text("#!/bin/sh\n")
            dest.chmod(0o755)

        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch("molexp.server.tunnel.fetch.bin_dir", return_value=tmp_path),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared", side_effect=fake_fetch),
        ):
            path = ensure_tunnel_client(
                via="cloudflared",
                explicit=None,
                ask=lambda binary, default_dir: default_dir / binary,
            )
        assert path == str((tmp_path / "cloudflared").resolve())

    def test_custom_path_download(self, tmp_path: Path) -> None:
        dest = tmp_path / "custom" / "cloudflared"

        def fake_fetch(path: Path) -> None:
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)

        with (
            patch("molexp.server.tunnel.base.shutil.which", return_value=None),
            patch("molexp.server.tunnel.fetch.bin_dir", return_value=tmp_path),
            patch("molexp.server.tunnel.fetch.fetch_cloudflared", side_effect=fake_fetch),
        ):
            path = ensure_tunnel_client(
                via="cloudflared",
                explicit=None,
                ask=lambda _binary, _default: dest,
            )
        assert path == str(dest.resolve())

    def test_locate_prefers_path(self) -> None:
        with patch("molexp.server.tunnel.base.shutil.which", return_value="/usr/bin/zrok"):
            assert locate_tunnel_bin(("zrok", "zrok2")) == "/usr/bin/zrok"
