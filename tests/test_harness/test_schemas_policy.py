"""Tests for PathPolicy / ToolPolicy / ApprovalPolicy (Phase 6 §7.2-7.4).

Locks the deny-by-default security posture: denied paths/commands ship
non-empty, network is off, and every ApprovalPolicy gate defaults to
requiring approval. Flipping any of these silently would weaken the
sandbox without any other test noticing.
"""

from __future__ import annotations


def test_path_policy_defaults() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    p = PathPolicy(workspace_root="/tmp/wf")
    assert p.workspace_root == "/tmp/wf"
    assert p.allowed_read_paths == []
    assert p.allowed_write_paths == []
    assert p.denied_paths == ["/", "/etc", "/usr", "~/.ssh"]


def test_tool_policy_defaults() -> None:
    from molexp.harness.schemas.policy import ToolPolicy

    p = ToolPolicy()
    assert p.allowed_commands == []
    assert p.denied_commands == ["rm -rf", "sudo", "chmod -R 777"]
    assert p.allow_network is False
    assert p.max_runtime_s == 3600
    assert p.max_output_mb == 1024


def test_approval_policy_all_six_fields_default_true() -> None:
    from molexp.harness.schemas.policy import ApprovalPolicy

    p = ApprovalPolicy()
    assert p.require_for_agent_inferred_scientific_parameters is True
    assert p.require_for_full_execution is True
    assert p.require_for_hpc_submission is True
    assert p.require_for_large_resource_request is True
    assert p.require_for_overwrite is True
    assert p.require_for_final_report is True
