"""RunMode — execute, monitor, and repair the materialized workflow.

The fourth pipeline mode (sub-spec 05). Consumes AuthorMode's
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`,
clears the :data:`~molexp.agent.modes._planning.ApprovalGate.approve_execution`
gate, loads the LLM-authored :class:`molexp.workflow.Workflow` through the
public ``molexp.workflow`` API, binds it to a workspace ``Run``, executes
it, projects per-step progress onto the typed plan, and on unrecoverable
failure emits a structured repair contract.

Public surface — import from this package root:

- :class:`RunMode` / :class:`RunModeConfig` — the mode + its tunables.
- :class:`RunProgress` / :class:`StepProgress` — the typed progress
  projection.
- :class:`RunReport` — the agent-facing run summary.
- :class:`RepairEscalation` — the contract toward AuthorMode.
- :class:`RunFolder` — the plan-anchored persistence ``Folder``.
"""

from molexp.agent.modes.run._mode import RunMode, RunModeConfig
from molexp.agent.modes.run.monitor import RunProgress, StepProgress, StepStatus
from molexp.agent.modes.run.repair import RepairEscalation, RuntimeFailureKind
from molexp.agent.modes.run.run_folder import RunFolder, RunReport

__all__ = [
    "RepairEscalation",
    "RunFolder",
    "RunMode",
    "RunModeConfig",
    "RunProgress",
    "RunReport",
    "RuntimeFailureKind",
    "StepProgress",
    "StepStatus",
]
