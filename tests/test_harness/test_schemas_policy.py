"""Tests for run-scoped policy schemas (``molexp.harness.schemas.policy``).

Locks the deny-by-default security posture: sensitive paths and destructive
commands ship denied, network is off, and every approval gate defaults to
requiring approval. Flipping any of these silently would weaken the sandbox
without any other test noticing.
"""

from __future__ import annotations

from molexp.harness.schemas.policy import ApprovalPolicy, PathPolicy, ToolPolicy


class TestPathPolicy:
    def test_denies_sensitive_paths_by_default(self) -> None:
        assert PathPolicy(workspace_root="/tmp/wf").denied_paths == ["/", "/etc", "/usr", "~/.ssh"]


class TestToolPolicy:
    def test_denies_destructive_commands_and_network_by_default(self) -> None:
        p = ToolPolicy()
        assert p.denied_commands == ["rm -rf", "sudo", "chmod -R 777"]
        assert p.allow_network is False


class TestApprovalPolicy:
    def test_all_gates_require_approval_by_default(self) -> None:
        p = ApprovalPolicy()
        assert p.require_for_agent_inferred_scientific_parameters is True
        assert p.require_for_full_execution is True
        assert p.require_for_hpc_submission is True
        assert p.require_for_large_resource_request is True
        assert p.require_for_overwrite is True
        assert p.require_for_final_report is True
