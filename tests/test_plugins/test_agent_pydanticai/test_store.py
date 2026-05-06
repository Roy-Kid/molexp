"""ProviderStore round-trip + update tests."""

from __future__ import annotations

from pathlib import Path

from molexp.plugins.agent_pydanticai.store import (
    DEFAULT_MODELS,
    ProviderStore,
    default_config,
    mask_api_key,
)


def test_default_config_when_missing(tmp_path: Path) -> None:
    config = ProviderStore(tmp_path).load()
    assert config.provider_name == "anthropic"
    assert config.model == DEFAULT_MODELS["anthropic"]
    assert config.api_key is None


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    store = ProviderStore(tmp_path)
    saved = store.save(default_config()._replace_no_op() if False else default_config())
    assert (tmp_path / ".molexp-agent" / "provider.json").exists()
    reloaded = store.load()
    assert reloaded.provider_name == saved.provider_name
    assert reloaded.model == saved.model


def test_update_changes_model_when_provider_switches(tmp_path: Path) -> None:
    store = ProviderStore(tmp_path)
    store.update(provider_name="anthropic", api_key="k1")
    after = store.update(provider_name="openai")
    assert after.provider_name == "openai"
    assert after.model == DEFAULT_MODELS["openai"]
    # api_key carries over
    assert after.api_key == "k1"


def test_update_clears_api_key_with_empty_string(tmp_path: Path) -> None:
    store = ProviderStore(tmp_path)
    store.update(api_key="secret")
    cleared = store.update(api_key="")
    assert cleared.api_key is None


def test_mask_api_key_obfuscates_long_keys() -> None:
    assert mask_api_key("sk-1234567890abcdef") == "sk-...cdef"
    assert mask_api_key("short") == "*****"
    assert mask_api_key("") == ""
    assert mask_api_key(None) == ""
