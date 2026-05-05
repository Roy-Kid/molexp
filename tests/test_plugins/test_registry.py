"""Tests for plugin registry."""

import pytest

from molexp.plugins import (
    Capability,
    CapabilityNotAvailable,
    PluginRegistry,
    discover_ui_plugins,
)


class TestPluginRegistry:
    def test_agent_capability_probed_against_pydantic_ai(self):
        reg = PluginRegistry()
        try:
            import pydantic_ai  # noqa: F401

            assert reg.is_available(Capability.AGENT)
        except ImportError:
            assert not reg.is_available(Capability.AGENT)

    def test_unavailable_raises(self):
        reg = PluginRegistry()
        reg.reset()
        reg._unavailable.add(Capability.AGENT)
        with pytest.raises(CapabilityNotAvailable):
            reg.get(Capability.AGENT)

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

    def test_coding_agent_caps_registered(self):
        """Both coding-agent providers (Claude CLI, Codex) are registered as
        independent capability slots, peers of ``Capability.AGENT``."""
        assert Capability.CODING_AGENT_CLAUDE == "coding_agent_claude"
        assert Capability.CODING_AGENT_CODEX == "coding_agent_codex"

        reg = PluginRegistry()
        # Both must be probable. Underlying CLI binaries may not be installed
        # on the dev machine, so we only assert that the capability slot is
        # *known* (probe runs without raising), not that it's available.
        for cap in (Capability.CODING_AGENT_CLAUDE, Capability.CODING_AGENT_CODEX):
            available = reg.is_available(cap)
            assert isinstance(available, bool)

    def test_gh_capability_registered_and_gated(self):
        """``Capability.GH`` is registered; available iff httpx is installed."""
        assert Capability.GH == "gh"
        reg = PluginRegistry()
        try:
            import httpx  # noqa: F401

            assert reg.is_available(Capability.GH)
        except ImportError:
            assert not reg.is_available(Capability.GH)

    def test_discover_ui_plugins_always_includes_core(self):
        plugins = discover_ui_plugins()
        assert [plugin.id for plugin in plugins][0] == "core"

    def test_discover_ui_plugins_includes_metrics(self):
        plugins = discover_ui_plugins()
        metrics = next(plugin for plugin in plugins if plugin.id == "metrics")

        assert metrics.ui_module == "metrics"
        assert metrics.capabilities == ("metrics", "run_metrics")

    def test_discover_ui_plugins_includes_molq_when_supported(self, monkeypatch):
        monkeypatch.setattr("molexp.plugins.supported_schedulers", lambda: ("slurm", "pbs"))

        plugins = discover_ui_plugins()
        molq = next(plugin for plugin in plugins if plugin.id == "molq")

        assert molq.ui_module == "molq"
        assert molq.metadata == {"schedulers": ["slurm", "pbs"]}
