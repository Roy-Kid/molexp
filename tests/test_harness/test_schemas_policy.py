"""Tests for PathPolicy / ToolPolicy / ApprovalPolicy (Phase 6 §7.2-7.4).

Locks:
- frozen pydantic round-trip
- documented defaults (denied_paths, denied_commands, allow_network=False, etc.)
- ApprovalPolicy all six require_for_* default True
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ----------------------------------------------------------------- PathPolicy


def test_path_policy_defaults() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    p = PathPolicy(workspace_root="/tmp/wf")
    assert p.workspace_root == "/tmp/wf"
    assert p.allowed_read_paths == []
    assert p.allowed_write_paths == []
    assert p.denied_paths == ["/", "/etc", "/usr", "~/.ssh"]


def test_path_policy_round_trip() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    p = PathPolicy(
        workspace_root="/tmp/wf",
        allowed_read_paths=["/data"],
        allowed_write_paths=["/tmp/wf/out"],
        denied_paths=["/", "~/.aws"],
    )
    dumped = p.model_dump_json()
    rehydrated = PathPolicy.model_validate_json(dumped)
    assert rehydrated == p


def test_path_policy_is_frozen() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    p = PathPolicy(workspace_root="/tmp/wf")
    with pytest.raises(ValidationError):
        p.workspace_root = "/mutated"  # type: ignore[misc]


def test_path_policy_requires_workspace_root() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    with pytest.raises(ValidationError):
        PathPolicy()  # type: ignore[call-arg]


def test_path_policy_default_factories_independent() -> None:
    from molexp.harness.schemas.policy import PathPolicy

    a = PathPolicy(workspace_root="/a")
    b = PathPolicy(workspace_root="/b")
    assert a.denied_paths is not b.denied_paths
    assert a.allowed_read_paths is not b.allowed_read_paths


# ----------------------------------------------------------------- ToolPolicy


def test_tool_policy_defaults() -> None:
    from molexp.harness.schemas.policy import ToolPolicy

    p = ToolPolicy()
    assert p.allowed_commands == []
    assert p.denied_commands == ["rm -rf", "sudo", "chmod -R 777"]
    assert p.allow_network is False
    assert p.max_runtime_s == 3600
    assert p.max_output_mb == 1024


def test_tool_policy_round_trip() -> None:
    from molexp.harness.schemas.policy import ToolPolicy

    p = ToolPolicy(
        allowed_commands=["pytest", "ruff"],
        denied_commands=["rm -rf", "curl"],
        allow_network=True,
        max_runtime_s=7200,
        max_output_mb=2048,
    )
    dumped = p.model_dump_json()
    rehydrated = ToolPolicy.model_validate_json(dumped)
    assert rehydrated == p


def test_tool_policy_is_frozen() -> None:
    from molexp.harness.schemas.policy import ToolPolicy

    p = ToolPolicy()
    with pytest.raises(ValidationError):
        p.allow_network = True  # type: ignore[misc]


def test_tool_policy_default_factories_independent() -> None:
    from molexp.harness.schemas.policy import ToolPolicy

    a = ToolPolicy()
    b = ToolPolicy()
    assert a.denied_commands is not b.denied_commands
    assert a.allowed_commands is not b.allowed_commands


# ---------------------------------------------------------------- ApprovalPolicy


def test_approval_policy_all_six_fields_default_true() -> None:
    from molexp.harness.schemas.policy import ApprovalPolicy

    p = ApprovalPolicy()
    assert p.require_for_agent_inferred_scientific_parameters is True
    assert p.require_for_full_execution is True
    assert p.require_for_hpc_submission is True
    assert p.require_for_large_resource_request is True
    assert p.require_for_overwrite is True
    assert p.require_for_final_report is True


def test_approval_policy_can_disable_individual_flags() -> None:
    from molexp.harness.schemas.policy import ApprovalPolicy

    p = ApprovalPolicy(require_for_hpc_submission=False, require_for_final_report=False)
    assert p.require_for_hpc_submission is False
    assert p.require_for_final_report is False
    # Other flags retain True default.
    assert p.require_for_full_execution is True


def test_approval_policy_round_trip() -> None:
    from molexp.harness.schemas.policy import ApprovalPolicy

    p = ApprovalPolicy(
        require_for_agent_inferred_scientific_parameters=False,
        require_for_full_execution=True,
        require_for_hpc_submission=True,
        require_for_large_resource_request=True,
        require_for_overwrite=False,
        require_for_final_report=True,
    )
    dumped = p.model_dump_json()
    rehydrated = ApprovalPolicy.model_validate_json(dumped)
    assert rehydrated == p


def test_approval_policy_is_frozen() -> None:
    from molexp.harness.schemas.policy import ApprovalPolicy

    p = ApprovalPolicy()
    with pytest.raises(ValidationError):
        p.require_for_full_execution = False  # type: ignore[misc]


# -------------------------------------------------- re-export discipline


def test_three_policies_re_exported_from_schemas_package() -> None:
    from molexp.harness.schemas import (
        ApprovalPolicy as via_pkg_app,
    )
    from molexp.harness.schemas import (
        PathPolicy as via_pkg_path,
    )
    from molexp.harness.schemas import (
        ToolPolicy as via_pkg_tool,
    )
    from molexp.harness.schemas.policy import (
        ApprovalPolicy as via_mod_app,
    )
    from molexp.harness.schemas.policy import (
        PathPolicy as via_mod_path,
    )
    from molexp.harness.schemas.policy import (
        ToolPolicy as via_mod_tool,
    )

    assert via_pkg_path is via_mod_path
    assert via_pkg_tool is via_mod_tool
    assert via_pkg_app is via_mod_app


def test_three_policies_re_exported_from_top_level() -> None:
    from molexp.harness import ApprovalPolicy, PathPolicy, ToolPolicy  # noqa: F401
