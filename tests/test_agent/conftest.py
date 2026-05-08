"""Shared fixtures for the agent test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws
