"""Shared planning-contracts package for the four-mode agent architecture.

Pure frozen-pydantic data models — the substrate every agent mode
(Plan / Author / Run / Review) reads and writes. No LLM, no disk I/O.

Four concept clusters:

- :mod:`intent` — the formal user-intent contract (``IntentSpec``).
- :mod:`plan_graph` — the typed plan DAG (``PlanGraph`` / ``PlanStep``);
  each ``PlanStep`` carries ``api_refs`` + ``composition_notes`` inline.
- :mod:`lifecycle` — the explicit ``PlanState`` machine.
- :mod:`diff` — the plan-diff repair unit and the approval gates.

Import the full surface from this package root; the cluster modules are
implementation detail.
"""

from __future__ import annotations

from .diff import (
    ApprovalGate,
    DiffOpKind,
    PlanDiff,
    PlanNodeOp,
    _resolve_forward_refs,
)
from .intent import (
    IntentConstraint,
    IntentSpec,
    MissingInfoItem,
    ResourceBudget,
    RiskLevel,
    SuccessCriterion,
)
from .lifecycle import (
    LEGAL_TRANSITIONS,
    IllegalPlanTransitionError,
    PlanState,
    assert_legal_transition,
    is_legal_transition,
    legal_successors,
)
from .plan_graph import (
    IsolatedTestSketch,
    PlanCheck,
    PlanGraph,
    PlanStep,
    PlanStepArtifact,
    PlanStepInput,
    PlanStepIO,
    RetryPolicy,
)

# Resolve the plan_graph <-> diff forward references now that both
# cluster modules are loaded.
_resolve_forward_refs()

__all__ = [
    "LEGAL_TRANSITIONS",
    "ApprovalGate",
    "DiffOpKind",
    "IllegalPlanTransitionError",
    "IntentConstraint",
    "IntentSpec",
    "IsolatedTestSketch",
    "MissingInfoItem",
    "PlanCheck",
    "PlanDiff",
    "PlanGraph",
    "PlanNodeOp",
    "PlanState",
    "PlanStep",
    "PlanStepArtifact",
    "PlanStepIO",
    "PlanStepInput",
    "ResourceBudget",
    "RetryPolicy",
    "RiskLevel",
    "SuccessCriterion",
    "assert_legal_transition",
    "is_legal_transition",
    "legal_successors",
]
