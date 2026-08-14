"""``molexp serve --dev`` — web checkout discovery and subprocess wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from molexp.cli.workspace.serve import (
    _DEFAULT_UI_PORT,
    _find_web_dir,
    _start_web_dev_server,
    _stop_web_dev_server,
)


class TestFindWebDir:
    def test_discovers_checkout_web(self) -> None:
        """Editable install from this repo must resolve ``apps/web/package.json``."""
        found = _find_web_dir()
        assert found is not None
        assert (found / "package.json").is_file()
        assert found.name == "web"
        assert found.parent.name == "apps"

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        web = tmp_path / "custom-web"
        web.mkdir()
        (web / "package.json").write_text('{"name":"x"}')
        monkeypatch.setenv("MOLEXP_WEB_DIR", str(web))
        assert _find_web_dir() == web.resolve()

    def test_env_override_missing_package_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MOLEXP_WEB_DIR", str(tmp_path))
        assert _find_web_dir() is None


class TestStartWebDevServer:
    def test_builds_npm_command_with_ports(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        fake_proc = MagicMock()
        with (
            patch("molexp.cli.workspace.serve.shutil.which", return_value="/usr/bin/npm"),
            patch("molexp.cli.workspace.serve.subprocess.Popen", return_value=fake_proc) as popen,
        ):
            proc = _start_web_dev_server(api_port=9000, web_port=5173, web_dir=tmp_path)
        assert proc is fake_proc
        args, kwargs = popen.call_args
        cmd = args[0]
        assert cmd[0] == "/usr/bin/npm"
        assert cmd[1:4] == ["run", "dev:api", "--"]
        assert "--port=5173" in cmd
        assert all(not str(a).startswith("--host=") for a in cmd)
        # API port must go through env (rsbuild rejects unknown CLI flags).
        assert all(not str(a).startswith("--api-port") for a in cmd)
        assert kwargs["env"]["MOLEXP_API_PORT"] == "9000"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["start_new_session"] is True

    def test_exits_when_npm_missing(self, tmp_path: Path) -> None:
        with (
            patch("molexp.cli.workspace.serve.shutil.which", return_value=None),
            pytest.raises(typer.Exit) as exc,
        ):
            _start_web_dev_server(api_port=8000, web_port=_DEFAULT_UI_PORT, web_dir=tmp_path)
        assert exc.value.exit_code == 1


class TestStopWebDevServer:
    def test_noop_when_none(self) -> None:
        _stop_web_dev_server(None)

    def test_noop_when_already_exited(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 0
        _stop_web_dev_server(proc)
        proc.terminate.assert_not_called()


class TestTunnelLocalPort:
    def test_dev_uses_ui_port(self) -> None:
        """``--dev --tunnel`` punches the Rsbuild UI port, not the API."""
        from molexp.cli.workspace.serve import tunnel_local_port

        assert tunnel_local_port(dev=True, api_port=8000, ui_port=5173) == 5173

    def test_non_dev_uses_api_port(self) -> None:
        """Bundled-UI serve tunnels the API process (UI lives there)."""
        from molexp.cli.workspace.serve import tunnel_local_port

        assert tunnel_local_port(dev=False, api_port=8000, ui_port=5173) == 8000


class TestTunnelLocalHost:
    def test_dev_uses_discovered_bind(self) -> None:
        """``--dev --tunnel`` dials the bind we read from the Dev UI."""
        from molexp.cli.workspace.serve import tunnel_local_host

        assert tunnel_local_host(dev=True, api_host="127.0.0.1", ui_host="::1") == "::1"

    def test_non_dev_loopback_uses_ipv4(self) -> None:
        """Bundled-UI serve dials the API process on 127.0.0.1."""
        from molexp.cli.workspace.serve import tunnel_local_host

        assert (
            tunnel_local_host(dev=False, api_host="localhost", ui_host="localhost") == "127.0.0.1"
        )


class TestParseDevUiUrl:
    def test_rsbuild_local_line(self) -> None:
        from molexp.cli.workspace.serve import parse_dev_ui_url

        line = "  ➜  Local:    http://localhost:5173/"
        assert parse_dev_ui_url(line) == "http://localhost:5173"

    def test_none_without_match(self) -> None:
        from molexp.cli.workspace.serve import parse_dev_ui_url

        assert parse_dev_ui_url("ready   built in 5.51s") is None

    def test_origin_splits_host_and_port(self) -> None:
        from molexp.cli.workspace.serve import origin_from_dev_ui_url

        assert origin_from_dev_ui_url("http://localhost:5173") == ("localhost", 5173)
        assert origin_from_dev_ui_url("http://[::1]:5173") == ("::1", 5173)


class TestDevUiListenHost:
    def test_prefers_ipv4_when_open(self) -> None:
        from molexp.cli.workspace.serve import dev_ui_listen_host

        def probe(host: str, port: int) -> bool:
            return host == "127.0.0.1" and port == 5173

        assert dev_ui_listen_host(5173, probe=probe, timeout=0.2) == "127.0.0.1"

    def test_falls_back_to_ipv6(self) -> None:
        from molexp.cli.workspace.serve import dev_ui_listen_host

        def probe(host: str, port: int) -> bool:
            return host == "::1" and port == 5173

        assert dev_ui_listen_host(5173, probe=probe, timeout=0.2) == "::1"


class TestAccessBannerLines:
    def test_dev_local_only_lists_localhost_ui_and_api(self) -> None:
        """Local ``--dev`` banner names Dev UI and API with localhost URLs."""
        from molexp.cli.workspace.serve import access_banner_lines

        lines = access_banner_lines(dev=True, host="localhost", api_port=8000, ui_port=5173)
        text = "\n".join(lines)
        assert "Dev UI" in text
        assert "http://localhost:5173" in text
        assert "API" in text
        assert "http://localhost:8000/api" in text

    def test_dev_with_public_url_puts_public_on_dev_ui_and_api_lines(self) -> None:
        """Public origin is the primary URL on both Dev UI and API lines."""
        from molexp.cli.workspace.serve import access_banner_lines

        public = "https://county-thomson-das-corn.trycloudflare.com"
        lines = access_banner_lines(
            dev=True,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            public_url=public,
        )
        assert any("Dev UI" in line and public in line for line in lines)
        assert any("API" in line and f"{public}/api" in line for line in lines)

    def test_dev_with_public_url_still_lists_localhost(self) -> None:
        """Public banner still includes the local Dev UI and API URLs."""
        from molexp.cli.workspace.serve import access_banner_lines

        lines = access_banner_lines(
            dev=True,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            public_url="https://county-thomson-das-corn.trycloudflare.com",
        )
        text = "\n".join(lines)
        assert "http://localhost:5173" in text
        assert "http://127.0.0.1:8000/api" in text

    def test_dev_banner_uses_ui_host(self) -> None:
        """The banner's local Dev UI line uses the advertised Rsbuild host."""
        from molexp.cli.workspace.serve import access_banner_lines

        lines = access_banner_lines(
            dev=True,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            ui_host="localhost",
            public_url="https://example.trycloudflare.com",
        )
        text = "\n".join(lines)
        assert "http://localhost:5173" in text

    def test_non_dev_with_public_url_has_no_dev_ui_line(self) -> None:
        """Bundled-UI serve never prints a Dev UI banner line."""
        from molexp.cli.workspace.serve import access_banner_lines

        lines = access_banner_lines(
            dev=False,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            public_url="https://example.trycloudflare.com",
        )
        assert all("Dev UI" not in line for line in lines)

    def test_non_dev_with_public_url_includes_public_host(self) -> None:
        """Bundled-UI public banner names the tunnel hostname."""
        from molexp.cli.workspace.serve import access_banner_lines

        public = "https://example.trycloudflare.com"
        lines = access_banner_lines(
            dev=False,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            public_url=public,
        )
        text = "\n".join(lines)
        assert "example.trycloudflare.com" in text
        assert public in text or f"{public}/api" in text

    def test_strips_trailing_slash_on_public_url(self) -> None:
        """A trailing slash on ``public_url`` must not produce ``//api``."""
        from molexp.cli.workspace.serve import access_banner_lines

        lines = access_banner_lines(
            dev=True,
            host="127.0.0.1",
            api_port=8000,
            ui_port=5173,
            public_url="https://county-thomson-das-corn.trycloudflare.com/",
        )
        text = "\n".join(lines)
        assert "https://county-thomson-das-corn.trycloudflare.com/api" in text
        assert "https://county-thomson-das-corn.trycloudflare.com//" not in text
