"""Loader + bridge for ``molexp.services.operator_config`` (owned here).

The services layer owns the config *writer* (``test_services/test_operator_config``);
by that file's own charter the *loader/bridge* coverage lives here. The CLI
documents ``molexp config set agent.model <id>`` (persisted to
``~/.molexp/config.json``) while server routes resolve the model from the
in-code ``molexp.config`` — ``create_app`` bridges the two so a CLI-configured
operator does not hit a 503 from ``/api/agent-tasks``.
"""

from __future__ import annotations

import json

import pytest

import molexp
from molexp.services.operator_config import (
    AGENT_MODEL_KEY,
    LEGACY_AGENT_MODEL_KEY,
    bridge_operator_config,
    configured_agent_model,
    load_operator_config,
)

_BRIDGED_KEYS = (AGENT_MODEL_KEY, LEGACY_AGENT_MODEL_KEY, "deepseek_api_key")


@pytest.fixture(autouse=True)
def _clean_molexp_config():
    """Snapshot/restore the process-global ``molexp.config`` keys we touch."""
    saved = {
        key: molexp.config.get(key) for key in _BRIDGED_KEYS if molexp.config.get(key) is not None
    }
    for key in _BRIDGED_KEYS:
        if molexp.config.get(key) is not None:
            del molexp.config[key]
    yield
    for key in _BRIDGED_KEYS:
        if molexp.config.get(key) is not None:
            del molexp.config[key]
    for key, value in saved.items():
        molexp.config[key] = value


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"model": "deepseek:deepseek-chat"}}))
    return path


class TestLoader:
    def test_missing_file_loads_as_empty(self, tmp_path):
        assert load_operator_config(tmp_path / "nope.json") == {}

    def test_corrupt_json_loads_as_empty(self, tmp_path):
        bad = tmp_path / "config.json"
        bad.write_text("{not json")
        assert load_operator_config(bad) == {}

    def test_configured_agent_model_reads_nested_key(self, config_file):
        cfg = load_operator_config(config_file)
        assert configured_agent_model(cfg) == "deepseek:deepseek-chat"

    def test_configured_agent_model_absent_variants_return_none(self):
        assert configured_agent_model({}) is None
        assert configured_agent_model({"agent": {}}) is None
        assert configured_agent_model({"agent": "oops"}) is None


class TestBridge:
    def test_bridge_populates_molexp_config(self, config_file):
        bridge_operator_config(config_file)
        assert molexp.config.get(AGENT_MODEL_KEY) == "deepseek:deepseek-chat"

    def test_in_code_model_wins_over_file(self, config_file):
        molexp.config[AGENT_MODEL_KEY] = "in-code:model"
        bridge_operator_config(config_file)
        assert molexp.config.get(AGENT_MODEL_KEY) == "in-code:model"

    def test_no_file_is_a_noop(self, tmp_path):
        bridge_operator_config(tmp_path / "absent.json")
        assert molexp.config.get(AGENT_MODEL_KEY) is None


class TestApiKeyBridge:
    """``agent.<provider>_api_key`` entries bridge into ``molexp.config`` —
    the persisted key path for ``molexp plan`` / ``molexp agent`` with
    providers (DeepSeek) whose keys are never read from the environment."""

    @pytest.fixture
    def keyed_config_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "agent": {
                        "model": "deepseek:deepseek-v4-flash",
                        "deepseek_api_key": "sk-from-operator-config",
                        "note": "not a key",
                        "empty_api_key": "",
                    }
                }
            )
        )
        return path

    def test_configured_api_keys_ignores_non_key_and_empty_entries(self, keyed_config_file):
        from molexp.services.operator_config import configured_api_keys

        cfg = load_operator_config(keyed_config_file)
        assert configured_api_keys(cfg) == {"deepseek_api_key": "sk-from-operator-config"}

    def test_bridge_lands_key_in_molexp_config(self, keyed_config_file):
        bridge_operator_config(keyed_config_file)
        assert molexp.config.get("deepseek_api_key") == "sk-from-operator-config"
        assert molexp.config.get(AGENT_MODEL_KEY) == "deepseek:deepseek-v4-flash"

    def test_model_precedence_does_not_block_key_bridging(self, keyed_config_file):
        molexp.config[AGENT_MODEL_KEY] = "in-code:model"
        bridge_operator_config(keyed_config_file)
        assert molexp.config.get(AGENT_MODEL_KEY) == "in-code:model"
        assert molexp.config.get("deepseek_api_key") == "sk-from-operator-config"


class TestRouteResolution:
    def test_agent_route_reads_bridged_model(self, config_file):
        from molexp.server.routes.agent import _configured_model

        bridge_operator_config(config_file)
        assert _configured_model() == "deepseek:deepseek-chat"

    def test_agent_route_falls_back_to_legacy_key(self):
        from molexp.server.routes.agent import _configured_model

        molexp.config[LEGACY_AGENT_MODEL_KEY] = "legacy:model"
        assert _configured_model() == "legacy:model"
