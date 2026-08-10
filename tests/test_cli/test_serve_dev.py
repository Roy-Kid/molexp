"""``molexp serve --dev`` — UI checkout discovery and subprocess wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from molexp.cli.workspace.serve import (
    _DEFAULT_UI_PORT,
    _find_ui_dir,
    _start_ui_dev_server,
    _stop_ui_dev_server,
)


class TestFindUiDir:
    def test_discovers_checkout_ui(self) -> None:
        """Editable install from this repo must resolve ``ui/package.json``."""
        found = _find_ui_dir()
        assert found is not None
        assert (found / "package.json").is_file()
        assert found.name == "ui"

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ui = tmp_path / "custom-ui"
        ui.mkdir()
        (ui / "package.json").write_text('{"name":"x"}')
        monkeypatch.setenv("MOLEXP_UI_DIR", str(ui))
        assert _find_ui_dir() == ui.resolve()

    def test_env_override_missing_package_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MOLEXP_UI_DIR", str(tmp_path))
        assert _find_ui_dir() is None


class TestStartUiDevServer:
    def test_builds_npm_command_with_ports(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        fake_proc = MagicMock()
        with (
            patch("molexp.cli.workspace.serve.shutil.which", return_value="/usr/bin/npm"),
            patch("molexp.cli.workspace.serve.subprocess.Popen", return_value=fake_proc) as popen,
        ):
            proc = _start_ui_dev_server(api_port=9000, ui_port=5173, ui_dir=tmp_path)
        assert proc is fake_proc
        args, kwargs = popen.call_args
        cmd = args[0]
        assert cmd[0] == "/usr/bin/npm"
        assert cmd[1:4] == ["run", "dev:api", "--"]
        assert "--port=5173" in cmd
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
            _start_ui_dev_server(api_port=8000, ui_port=_DEFAULT_UI_PORT, ui_dir=tmp_path)
        assert exc.value.exit_code == 1


class TestStopUiDevServer:
    def test_noop_when_none(self) -> None:
        _stop_ui_dev_server(None)

    def test_noop_when_already_exited(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 0
        _stop_ui_dev_server(proc)
        proc.terminate.assert_not_called()
