"""Operator-config **writer** — ``save_operator_config`` / ``set_operator_values``.

Spec ``vision-loop-03-settings-operator-config``: the services layer gains the
one write path for ``~/.molexp/config.json`` (atomic tmp+rename, the idiom
``cli/config_cmd.py`` already uses), and the CLI's private ``_save_config``
is deleted in favour of delegation — CLI and server share loader **and**
writer. Loader/bridge coverage stays in
``tests/test_server/test_operator_config_bridge.py`` (untouched).

Every test points the writer at a tmp path — the operator's real
``~/.molexp/config.json`` is never read or written.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from molexp.services.operator_config import (
    load_operator_config,
    save_operator_config,
    set_operator_values,
)

_MODEL = "deepseek:deepseek-v4-flash"
_RAW_KEY = "sk-operator-secret-key-9876"


def _boom(*args: object, **kwargs: object) -> None:
    raise OSError("injected rename failure")


class TestSaveOperatorConfig:
    """``save_operator_config(config, path=None)`` — the atomic file writer."""

    def test_save_round_trips_through_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        config = {"agent": {"model": _MODEL, "deepseek_api_key": _RAW_KEY}}

        save_operator_config(config, path=path)

        assert load_operator_config(path) == config

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / ".molexp" / "config.json"

        save_operator_config({"agent": {"model": _MODEL}}, path=path)

        assert load_operator_config(path) == {"agent": {"model": _MODEL}}

    def test_save_sets_owner_only_permissions(self, tmp_path: Path) -> None:
        """The file stores API keys — 0600, same as the CLI idiom it replaces."""
        path = tmp_path / "config.json"

        save_operator_config({"agent": {"deepseek_api_key": _RAW_KEY}}, path=path)

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_save_failure_leaves_original_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Atomicity: a failed rename never corrupts / truncates the target."""
        path = tmp_path / "config.json"
        save_operator_config({"agent": {"model": _MODEL}}, path=path)
        before = path.read_bytes()

        monkeypatch.setattr(os, "replace", _boom)
        monkeypatch.setattr(os, "rename", _boom)
        with pytest.raises(OSError):
            save_operator_config({"agent": {"model": "other:model"}}, path=path)

        assert path.read_bytes() == before
        assert load_operator_config(path) == {"agent": {"model": _MODEL}}


class TestSetOperatorValues:
    """``set_operator_values(updates, *, unset=None, path=None)`` on dotted keys."""

    def test_set_creates_nested_agent_section(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"

        set_operator_values({"agent.model": _MODEL}, path=path)

        assert load_operator_config(path) == {"agent": {"model": _MODEL}}

    def test_set_round_trips_model_and_api_key(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"

        set_operator_values(
            {"agent.model": _MODEL, "agent.deepseek_api_key": _RAW_KEY},
            path=path,
        )

        loaded = load_operator_config(path)
        agent = loaded["agent"]
        assert isinstance(agent, dict)
        assert agent["model"] == _MODEL
        assert agent["deepseek_api_key"] == _RAW_KEY

    def test_set_preserves_unrelated_sections(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        save_operator_config(
            {"defaults": {"shell": "bash"}, "agent": {"instructions": "be brief"}},
            path=path,
        )

        set_operator_values({"agent.model": _MODEL}, path=path)

        loaded = load_operator_config(path)
        assert loaded["defaults"] == {"shell": "bash"}
        assert loaded["agent"] == {"instructions": "be brief", "model": _MODEL}

    def test_set_supports_arbitrary_dotted_sections(self, tmp_path: Path) -> None:
        """The CLI's ``molexp config set defaults.shell bash`` delegates here —
        the writer must accept dotted keys outside the ``agent`` section too."""
        path = tmp_path / "config.json"

        set_operator_values({"defaults.shell": "bash"}, path=path)

        assert load_operator_config(path) == {"defaults": {"shell": "bash"}}

    def test_unset_removes_only_the_named_key(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        set_operator_values(
            {"agent.model": _MODEL, "agent.deepseek_api_key": _RAW_KEY},
            path=path,
        )

        set_operator_values({}, unset=["agent.deepseek_api_key"], path=path)

        loaded = load_operator_config(path)
        agent = loaded["agent"]
        assert isinstance(agent, dict)
        assert agent.get("model") == _MODEL
        assert "deepseek_api_key" not in agent
        assert _RAW_KEY not in path.read_text()

    def test_set_failure_leaves_existing_config_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.json"
        set_operator_values({"agent.model": _MODEL}, path=path)
        before = path.read_bytes()

        monkeypatch.setattr(os, "replace", _boom)
        monkeypatch.setattr(os, "rename", _boom)
        with pytest.raises(OSError):
            set_operator_values({"agent.model": "other:model"}, path=path)

        assert path.read_bytes() == before


class TestCliParity:
    """One writer: ``molexp config set`` and ``set_operator_values`` produce
    byte-identical files — no second serialization path survives the slice."""

    def test_cli_config_set_matches_service_writer_byte_for_byte(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from typer.testing import CliRunner

        from molexp.cli import app, config_cmd
        from molexp.services import operator_config

        cli_path = tmp_path / "cli" / "config.json"
        svc_path = tmp_path / "svc" / "config.json"
        cli_path.parent.mkdir(parents=True)
        svc_path.parent.mkdir(parents=True)

        # The CLI resolves the target through the module seams; point every
        # spelling at the tmp CLI path so the real ~/.molexp is never touched.
        monkeypatch.setattr(operator_config, "OPERATOR_CONFIG_PATH", cli_path)
        monkeypatch.setattr(config_cmd, "_CONFIG_PATH", cli_path, raising=False)

        result = CliRunner().invoke(app, ["config", "set", "agent.model", _MODEL])
        assert result.exit_code == 0, result.output

        set_operator_values({"agent.model": _MODEL}, path=svc_path)

        assert cli_path.read_bytes() == svc_path.read_bytes()
