"""Unit tests for the server-owned active-workspace switching machinery in
``molexp.server.dependencies`` — the mutually-exclusive path/descriptor overrides
and the subscriber-drain contract fired on each workspace switch. All behaviors
here are server-layer — no lower-layer domain outcome is asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.server.dependencies import (
    register_workspace_subscriber,
    reset_workspace_cache,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)


@pytest.fixture(autouse=True)
def _isolate_dependency_state():
    """Reset cache + overrides for each test."""
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    yield
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)


@pytest.mark.unit
class TestActiveWorkspaceSwitching:
    """Cache + override invariants of the active-workspace switch."""

    def test_path_and_descriptor_overrides_are_mutually_exclusive(self, tmp_path: Path):
        import molexp.server.dependencies as deps

        set_workspace_path_override(tmp_path / "x")
        assert deps._workspace_path_override is not None
        assert deps._workspace_descriptor_override is None

        set_active_workspace_descriptor("hpc1")
        assert deps._workspace_descriptor_override == "hpc1"
        assert deps._workspace_path_override is None

        set_workspace_path_override(tmp_path / "y")
        assert deps._workspace_descriptor_override is None
        assert deps._workspace_path_override is not None


@pytest.mark.unit
class TestWorkspaceSubscribers:
    """The subscriber-drain contract fired on each workspace switch."""

    def test_drained_before_cache_reset(self, tmp_path: Path):
        """Registered closers must run *before* the workspace cache is reset."""
        order: list[str] = []

        def closer():
            order.append("closer")

        import molexp.server.dependencies as deps

        deps._workspace_cache[("local", "/tmp/sentinel")] = object()  # type: ignore[assignment]
        register_workspace_subscriber(closer)
        set_workspace_path_override(tmp_path / "x")
        assert "closer" in order
        assert ("local", "/tmp/sentinel") not in deps._workspace_cache

    def test_awaitable_closers_are_awaited(self, tmp_path: Path):
        """Closers may return an awaitable; the drain awaits them."""
        order: list[str] = []

        async def async_closer():
            order.append("async-closer")

        register_workspace_subscriber(async_closer)
        set_workspace_path_override(tmp_path / "x")
        assert order == ["async-closer"]

    def test_cleared_after_drain_no_double_fire(self, tmp_path: Path):
        """Drain must clear the subscriber list — switching twice doesn't double-fire."""
        calls: list[int] = []

        def closer():
            calls.append(1)

        register_workspace_subscriber(closer)
        set_workspace_path_override(tmp_path / "a")
        set_workspace_path_override(tmp_path / "b")
        assert calls == [1]
