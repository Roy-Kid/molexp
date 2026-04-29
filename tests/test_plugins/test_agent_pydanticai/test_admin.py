"""Tests for native tool catalog introspection.

MCP server tests live in :mod:`test_mcp_store`; this module covers the
``describe_native_tools`` helper used by the settings UI.
"""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai.admin import describe_native_tools
from molexp.plugins.agent_pydanticai.policy import ApprovalPolicy


@pytest.mark.unit
def test_describe_native_tools_lists_known_tools():
    rows = describe_native_tools()
    names = {r["name"] for r in rows}
    assert {"submit_run", "retry_run", "ask_user"} <= names


@pytest.mark.unit
def test_describe_native_tools_no_default_approval():
    """Default policy is friction-free — chat is the consent surface.

    Production deployments opt back in by passing an explicit
    :class:`ApprovalPolicy` to :class:`AgentService`. See the
    :data:`DEFAULT_APPROVAL_TOOLS` docstring for rationale.
    """
    rows = describe_native_tools()
    by_name = {r["name"]: r for r in rows}
    assert by_name["submit_run"]["requires_approval"] is False
    assert by_name["execute_run"]["requires_approval"] is False
    assert by_name["retry_run"]["requires_approval"] is False
    assert by_name["get_run_status"]["requires_approval"] is False


@pytest.mark.unit
def test_describe_native_tools_respects_custom_policy():
    policy = ApprovalPolicy(require_approval_for=["ask_user"])
    rows = describe_native_tools(policy)
    by_name = {r["name"]: r for r in rows}
    assert by_name["ask_user"]["requires_approval"] is True
    assert by_name["submit_run"]["requires_approval"] is False
