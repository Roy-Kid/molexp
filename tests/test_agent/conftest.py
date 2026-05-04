"""Shared fixtures for the harness test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent import AgentService, Goal


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def agent_service(workspace_path: Path) -> AgentService:
    return AgentService.from_workspace(workspace_path)


@pytest.fixture
def chat_goal() -> Goal:
    return Goal(description="say hello")
