"""Public mode surface — concrete :class:`~molexp.agent.mode.AgentMode` subclasses.

Six modes ship today. Five are *declarative* (a static stage DAG):
:class:`ChatMode` (the simple reference mode), :class:`PlanMode` (the
read-only typed planner, sub-spec 03), :class:`AuthorMode` (materialize
an approved plan into a workspace, sub-spec 04), :class:`RunMode`
(execute, monitor, and repair the materialized workflow, sub-spec 05),
and :class:`ReviewMode` (read-only typed review of an existing artefact
against the shared IntentSpec contract, sub-spec 06).

The sixth, :class:`InteractiveMode`, is *emergent*: the LLM drives a
tool-using loop, autonomously deciding when to call a read-only tool or
delegate to the structured PlanMode pipeline. It is the CLI's default
interactive mode and a sibling of the declarative five — composing,
never inheriting, them.
"""

from molexp.agent.modes.author import (
    AuthorMode,
    AuthorModeConfig,
    MaterializedWorkspaceHandoff,
)
from molexp.agent.modes.chat import ChatMode, ChatModeConfig
from molexp.agent.modes.interactive import InteractiveMode, InteractiveModeConfig
from molexp.agent.modes.plan import (
    ApprovedPlanHandoff,
    PlanFolder,
    PlanMode,
    PlanModeConfig,
)
from molexp.agent.modes.review import ReviewMode, ReviewModeConfig
from molexp.agent.modes.run import (
    RepairEscalation,
    RunFolder,
    RunMode,
    RunModeConfig,
    RunProgress,
    RunReport,
    StepProgress,
)

__all__ = [
    "ApprovedPlanHandoff",
    "AuthorMode",
    "AuthorModeConfig",
    "ChatMode",
    "ChatModeConfig",
    "InteractiveMode",
    "InteractiveModeConfig",
    "MaterializedWorkspaceHandoff",
    "PlanFolder",
    "PlanMode",
    "PlanModeConfig",
    "RepairEscalation",
    "ReviewMode",
    "ReviewModeConfig",
    "RunFolder",
    "RunMode",
    "RunModeConfig",
    "RunProgress",
    "RunReport",
    "StepProgress",
]
