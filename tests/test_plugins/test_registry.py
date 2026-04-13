"""Tests for plugin registry."""

import pytest

from molexp.plugins import (
    Capability,
    CapabilityNotAvailable,
    PluginRegistry,
    discover_ui_plugins,
)


class TestPluginRegistry:
    def test_fresh_registry_agent_probed(self):
        reg = PluginRegistry()
        # Agent availability depends on pydantic-ai being installed
        try:
            import pydantic_ai  # noqa: F401
            assert reg.is_available(Capability.AGENT)
        except ImportError:
            assert not reg.is_available(Capability.AGENT)

    def test_unavailable_raises(self):
        reg = PluginRegistry()
        # Use a fresh registry with cleared caches to test the raise path
        reg.reset()
        # Monkey-patch to make AGENT unavailable
        reg._unavailable.add(Capability.AGENT)
        with pytest.raises(CapabilityNotAvailable):
            reg.get(Capability.AGENT)

    def test_agent_available_if_pydantic_ai_installed(self):
        reg = PluginRegistry()
        try:
            import pydantic_ai  # noqa: F401
            assert reg.is_available(Capability.AGENT)
        except ImportError:
            assert not reg.is_available(Capability.AGENT)

    def test_reset_clears_caches(self):
        reg = PluginRegistry()
        reg.is_available(Capability.AGENT)  # caches result
        reg.reset()
        assert len(reg._cache) == 0
        assert len(reg._unavailable) == 0

    def test_available_capabilities_returns_list(self):
        reg = PluginRegistry()
        caps = reg.available_capabilities()
        assert isinstance(caps, list)

    def test_capability_enum_values(self):
        assert Capability.AGENT == "agent"

    def test_discover_ui_plugins_always_includes_core(self):
        plugins = discover_ui_plugins()
        assert [plugin.id for plugin in plugins][0] == "core"

    def test_discover_ui_plugins_includes_molq_when_supported(self, monkeypatch):
        monkeypatch.setattr("molexp.plugins.supported_schedulers", lambda: ("slurm", "pbs"))

        plugins = discover_ui_plugins()
        molq = next(plugin for plugin in plugins if plugin.id == "molq")

        assert molq.ui_module == "molq"
        assert molq.metadata == {"schedulers": ["slurm", "pbs"]}
