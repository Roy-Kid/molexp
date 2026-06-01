"""Run-scoped policy schemas (Phase 6 §7.2-7.4).

Three frozen pydantic models sibling to Phase-3's task-scoped
:class:`ResourcePolicy` on ``BoundWorkflow``:

- :class:`PathPolicy` — filesystem access policy for the run.
- :class:`ToolPolicy` — command-execution policy for the run.
- :class:`ApprovalPolicy` — which run actions require human approval.

Runtime enforcement of :class:`PathPolicy` and :class:`ToolPolicy` lands
when executors arrive (Phase 7+); Phase 6 ships only the typed shape so
upstream callers can declare policies and downstream evaluators (and
future executors) can check them.

:class:`ApprovalPolicy` *is* consumed in Phase 6 by
:func:`molexp.harness.policy.evaluate_approval_policy`, which walks a
``BoundWorkflow`` (and optional ``WorkflowIR``) and emits one
:class:`ApprovalRequest` per triggered ``require_for_*`` clause.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ApprovalPolicy", "PathPolicy", "ToolPolicy"]


class PathPolicy(BaseModel):
    """Filesystem access policy at run scope."""

    model_config = ConfigDict(frozen=True)

    workspace_root: str
    allowed_read_paths: list[str] = Field(default_factory=list)
    allowed_write_paths: list[str] = Field(default_factory=list)
    denied_paths: list[str] = Field(default_factory=lambda: ["/", "/etc", "/usr", "~/.ssh"])


class ToolPolicy(BaseModel):
    """Command-execution policy at run scope.

    ``allowed_commands == []`` is interpreted as **unrestricted** —
    ``denied_commands`` still acts as an explicit blocklist. Strict
    whitelisting (empty list = deny all) is deferred to a future phase.
    """

    model_config = ConfigDict(frozen=True)

    allowed_commands: list[str] = Field(default_factory=list)
    denied_commands: list[str] = Field(default_factory=lambda: ["rm -rf", "sudo", "chmod -R 777"])
    allow_network: bool = False
    max_runtime_s: int = 3600
    max_output_mb: int = 1024


class ApprovalPolicy(BaseModel):
    """Which actions in a harness run require explicit approval.

    All six flags default to ``True`` — the safest stance is to ask. A
    caller who wants a fully-automated run flips the relevant flags off
    after considering the audit implications.
    """

    model_config = ConfigDict(frozen=True)

    require_for_agent_inferred_scientific_parameters: bool = True
    require_for_full_execution: bool = True
    require_for_hpc_submission: bool = True
    require_for_large_resource_request: bool = True
    require_for_overwrite: bool = True
    require_for_final_report: bool = True
