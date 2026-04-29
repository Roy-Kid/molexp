"""Unit tests for the workspace-scoped provider config store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from molexp.plugins.agent_pydanticai import provider as provider_mod
from molexp.plugins.agent_pydanticai.provider import (
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderConfig,
    ProviderStore,
    check_credentials,
    mask_api_key,
    probe_provider,
    to_public,
)


@pytest.mark.unit
def test_load_returns_default_when_file_missing(tmp_path):
    store = ProviderStore(tmp_path)
    config = store.load()
    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-6"
    assert config.api_key == ""
    assert config.base_url == ""


@pytest.mark.unit
def test_save_round_trips_through_load(tmp_path):
    store = ProviderStore(tmp_path)
    saved = store.save(
        ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-1234567890",
            base_url="https://proxy.example/v1",
        )
    )
    reloaded = ProviderStore(tmp_path).load()
    assert reloaded == saved


@pytest.mark.unit
def test_to_public_redacts_key(tmp_path):
    config = ProviderConfig(
        provider="openai", model="gpt-4o", api_key="sk-test-1234567890"
    )
    public = to_public(config)
    assert public.api_key_set is True
    assert public.api_key_preview == "sk-...7890"
    assert "1234" not in public.api_key_preview
    # The model_dump should not contain the raw key.
    assert "api_key" not in public.model_dump()


@pytest.mark.unit
def test_mask_api_key_short_keys_fully_masked():
    assert mask_api_key("") == ""
    assert mask_api_key("abc") == "***"
    assert mask_api_key("longer-key-1234") == "lon...1234"


@pytest.mark.unit
def test_update_partial_preserves_other_fields(tmp_path):
    store = ProviderStore(tmp_path)
    store.save(
        ProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-secret",
            base_url="",
        )
    )
    updated = store.update(api_key="sk-rotated")
    assert updated.api_key == "sk-rotated"
    assert updated.provider == "anthropic"
    assert updated.model == "claude-sonnet-4-6"


@pytest.mark.unit
def test_update_clears_key_with_empty_string(tmp_path):
    store = ProviderStore(tmp_path)
    store.save(ProviderConfig(api_key="sk-secret"))
    updated = store.update(api_key="")
    assert updated.api_key == ""


@pytest.mark.unit
def test_update_provider_switch_resets_model_to_default(tmp_path):
    store = ProviderStore(tmp_path)
    store.save(ProviderConfig(provider="anthropic", model="claude-sonnet-4-6"))
    updated = store.update(provider="openai")
    assert updated.provider == "openai"
    assert updated.model == DEFAULT_MODELS["openai"]


@pytest.mark.unit
def test_update_provider_switch_with_explicit_model_keeps_it(tmp_path):
    store = ProviderStore(tmp_path)
    store.save(ProviderConfig(provider="anthropic", model="claude-sonnet-4-6"))
    updated = store.update(provider="openai", model="gpt-4o-mini")
    assert updated.provider == "openai"
    assert updated.model == "gpt-4o-mini"


@pytest.mark.unit
def test_pydantic_ai_model_string_round_trip():
    cfg = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6")
    assert cfg.pydantic_ai_model_string() == "anthropic:claude-sonnet-4-6"
    cfg = ProviderConfig(provider="openai-compatible", model="qwen2.5")
    # openai-compatible reuses the openai prefix; only base_url differs.
    assert cfg.pydantic_ai_model_string() == "openai:qwen2.5"


@pytest.mark.unit
def test_supported_providers_contains_default_models():
    for name in SUPPORTED_PROVIDERS:
        assert name in DEFAULT_MODELS


@pytest.mark.unit
def test_deepseek_is_first_class_provider():
    """DeepSeek must appear in the public list with a sensible default."""
    assert "deepseek" in SUPPORTED_PROVIDERS
    assert DEFAULT_MODELS["deepseek"] == "deepseek-chat"


@pytest.mark.unit
def test_build_model_for_deepseek_uses_default_base_url():
    from molexp.plugins.agent_pydanticai._pydantic_ai.runtime import _build_model_from_config
    from molexp.plugins.agent_pydanticai.provider import DEEPSEEK_DEFAULT_BASE_URL

    cfg = ProviderConfig(provider="deepseek", model="deepseek-chat", api_key="sk-x")
    model = _build_model_from_config(cfg)
    # base_url should resolve to the DeepSeek endpoint when the user
    # leaves it blank — that's the whole point of the dedicated branch.
    assert DEEPSEEK_DEFAULT_BASE_URL.rstrip("/") in str(model.base_url).rstrip("/")
    assert model.model_name == "deepseek-chat"


@pytest.mark.unit
def test_build_model_for_deepseek_honors_user_base_url():
    """Explicit ``base_url`` (e.g. a regional mirror) overrides the default."""
    from molexp.plugins.agent_pydanticai._pydantic_ai.runtime import _build_model_from_config

    cfg = ProviderConfig(
        provider="deepseek",
        model="deepseek-reasoner",
        api_key="sk-x",
        base_url="https://my-mirror.example/v1",
    )
    model = _build_model_from_config(cfg)
    assert "my-mirror.example" in str(model.base_url)


@pytest.mark.unit
def test_check_credentials_for_deepseek_uses_correct_env_var(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    status = check_credentials(ProviderConfig(provider="deepseek", api_key=""))
    assert status.ready is False
    assert status.env_var == "DEEPSEEK_API_KEY"
    assert "DEEPSEEK_API_KEY" in status.reason

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    status = check_credentials(ProviderConfig(provider="deepseek", api_key=""))
    assert status.ready is True
    assert status.source == "env"


@pytest.mark.unit
def test_load_corrupt_file_returns_default(tmp_path):
    path = tmp_path / ".agent_provider.json"
    path.write_text("{not json")
    config = ProviderStore(tmp_path).load()
    assert config == ProviderConfig()


# ── Probe ──────────────────────────────────────────────────────────────────


@dataclass
class _FakeAgentResult:
    output: str


class _FakeAgent:
    """Stand-in for pydantic_ai.Agent that records calls and returns canned output."""

    last_kwargs: dict | None = None

    def __init__(self, model):
        self.model = model

    async def run(self, prompt, **kwargs):
        type(self).last_kwargs = {"prompt": prompt, **kwargs}
        return _FakeAgentResult(output="pong")


class _RaisingAgent:
    def __init__(self, model):
        self.model = model

    async def run(self, prompt, **kwargs):
        raise RuntimeError("invalid x-api-key")


def _patch_pydantic_ai(monkeypatch, agent_cls):
    """Stub out the lazy imports inside :func:`probe_provider`.

    We replace the public ``pydantic_ai.Agent`` attribute and the internal
    model builder so the probe path never hits a real LLM. Replacing the
    whole ``pydantic_ai`` module breaks its submodules — keep the real
    package and patch only the two symbols we need.
    """
    import pydantic_ai

    monkeypatch.setattr(pydantic_ai, "Agent", agent_cls)
    monkeypatch.setattr(
        "molexp.plugins.agent_pydanticai._pydantic_ai.runtime._build_model_from_config",
        lambda cfg: f"<fake-model:{cfg.provider}:{cfg.model}>",
    )


@pytest.mark.unit
def test_probe_returns_error_when_no_key():
    config = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="")
    result = asyncio.run(probe_provider(config))
    assert result.ok is False
    assert "No API key" in (result.error or "")


@pytest.mark.unit
def test_probe_success_records_latency_and_reply(monkeypatch):
    _patch_pydantic_ai(monkeypatch, _FakeAgent)
    config = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-real")
    result = asyncio.run(probe_provider(config))
    assert result.ok is True
    assert result.reply == "pong"
    assert result.latency_ms >= 0
    assert _FakeAgent.last_kwargs is not None
    assert _FakeAgent.last_kwargs["model_settings"] == {
        "max_tokens": provider_mod.PROBE_MAX_TOKENS,
    }


@pytest.mark.unit
def test_probe_failure_redacts_key_in_error(monkeypatch):
    """If the SDK leaks the key into the error string, scrub it before returning."""
    api_key = "sk-very-secret-1234567890"

    class LeakyAgent:
        def __init__(self, model):
            pass

        async def run(self, prompt, **kwargs):
            raise RuntimeError(f"401: {api_key} is invalid")

    _patch_pydantic_ai(monkeypatch, LeakyAgent)
    config = ProviderConfig(provider="openai", model="gpt-4o", api_key=api_key)
    result = asyncio.run(probe_provider(config))
    assert result.ok is False
    assert api_key not in (result.error or "")
    assert "RuntimeError" in (result.error or "")


# ── check_credentials ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_check_credentials_prefers_stored_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = ProviderConfig(provider="anthropic", api_key="sk-stored")
    status = check_credentials(config)
    assert status.ready is True
    assert status.source == "stored"
    assert status.reason == ""


@pytest.mark.unit
def test_check_credentials_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    config = ProviderConfig(provider="openai", api_key="")
    status = check_credentials(config)
    assert status.ready is True
    assert status.source == "env"
    assert status.env_var == "OPENAI_API_KEY"


@pytest.mark.unit
def test_check_credentials_reports_none_with_helpful_reason(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    config = ProviderConfig(provider="google", api_key="")
    status = check_credentials(config)
    assert status.ready is False
    assert status.source == "none"
    assert status.env_var == "GOOGLE_API_KEY"
    assert "GOOGLE_API_KEY" in status.reason


@pytest.mark.unit
def test_probe_timeout_returns_friendly_error(monkeypatch):
    class HangingAgent:
        def __init__(self, model):
            pass

        async def run(self, prompt, **kwargs):
            await asyncio.sleep(60)

    _patch_pydantic_ai(monkeypatch, HangingAgent)
    monkeypatch.setattr(provider_mod, "PROBE_TIMEOUT_SECONDS", 0.05)
    config = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-real")
    result = asyncio.run(probe_provider(config))
    assert result.ok is False
    assert "Timeout" in (result.error or "")
