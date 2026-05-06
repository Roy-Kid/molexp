"""Plan-side data contracts: PlanProposal and its frozen-dataclass family.

This module is the **single source of truth** for the structured plan
that a planning agent emits and that the workflow compiler consumes.

The hierarchy::

    PlanProposal
      ├── task_proposals: tuple[TaskProposal, ...]
      ├── sanity_specs:   tuple[SanitySpec, ...]
      ├── parallels:      tuple[ParallelSpec, ...]
      ├── loops:          tuple[LoopSpec, ...]
      ├── branches:       tuple[BranchSpec, ...]
      ├── sweeps:         tuple[SweepSpec, ...]
      ├── intervention_points: tuple[InterventionPoint, ...]
      ├── parent_proposal_id: str | None
      ├── revision: int
      └── proposal_id: str  (computed)

Every dataclass is ``frozen=True`` and uses tuple / Mapping fields so
the in-memory plan is structurally immutable. The ``proposal_id`` is a
content hash computed in ``PlanProposal.__post_init__`` and is
**order-independent** for set-like collections (task ordering,
``depends_on`` ordering, ``Mapping`` key insertion order).

Module-level imports are limited to stdlib so the workflow layer stays
import-pure and decoupled from the agent runtime. Agent-authored code is
referenced by absolute :class:`pathlib.Path` — workspace ``ArtifactAsset``
integration is a downstream Part B concern.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "BranchSpec",
    "InterventionPoint",
    "LoopSpec",
    "ParallelSpec",
    "ParameterizedWorkflowSpec",
    "PlanProposal",
    "SanitySpec",
    "SweepSpec",
    "TaskProposal",
]


# ── Constants ─────────────────────────────────────────────────────────────────


_HASH_HEX_LEN = 16
"""Hex-character length of `proposal_id` and `workflow_id`.

Both ids are sha256 truncated to this many hex chars. Tests assert the
shape; consumers may rely on it. Changing this is a contract break.
"""


# ── Control-flow specs ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InterventionPoint:
    """A named anchor in the plan where the user (or another agent) may step in.

    Carries a JSON schema describing the expected payload at that point. Part
    A.1 stores it but does not yet act on it; the runtime hook lives in Part B.
    """

    name: str
    description: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)


_VALID_ON_FAIL = ("halt", "replan", "continue")


@dataclass(frozen=True)
class SanitySpec:
    """Declarative sanity-check spec, fronting Part A.2's first-class node.

    Part A.1 only stores the spec on a :class:`PlanProposal`; runtime
    wiring (real ``SanityNodeStarted`` / ``Passed`` / ``Failed`` events,
    halt / replan / continue dispatch) lands in Part A.2. ``modifier_ref``
    lets a future replan loop re-run a deterministic, agent-authored
    correction; it is part of the ``proposal_id`` hash now so two plans
    that differ only in their modifier do not collide.
    """

    after: str
    on_fail: Literal["halt", "replan", "continue"]
    predicate_ref: Path | None = None
    modifier_ref: Path | None = None
    retry: int = 0

    def __post_init__(self) -> None:
        if self.on_fail not in _VALID_ON_FAIL:
            raise ValueError(
                f"SanitySpec.on_fail must be one of {_VALID_ON_FAIL!r}; got {self.on_fail!r}"
            )


@dataclass(frozen=True)
class ParallelSpec:
    map_over: str
    body: str
    join: str
    max_concurrency: int


@dataclass(frozen=True)
class LoopSpec:
    body: tuple[str, ...]
    until: str
    max_iters: int
    on_exit: str


@dataclass(frozen=True)
class BranchSpec:
    src: str
    routes: Mapping[str, str]


@dataclass(frozen=True)
class SweepSpec:
    dimension: str
    axes: Mapping[str, tuple[Any, ...]]


# ── TaskProposal ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskProposal:
    """One node in a :class:`PlanProposal`.

    ``kind == "registered"`` references a slug in the workflow's
    :class:`~molexp.workflow.registry.TaskTypeRegistry`; the compiler
    looks it up and instantiates the bound class.

    ``kind == "agent_authored"`` carries an absolute ``Path`` to a
    file the planning agent dropped on disk via the sandbox; the
    compiler verifies the file exists. The whole point of routing
    agent-authored code through a path (instead of inlining the
    source string) is that the proposal stays small and the code is
    discoverable + diffable on disk.
    """

    task_id: str
    kind: Literal["registered", "agent_authored"]
    task_type: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    code_artifact: Path | None = None

    def __post_init__(self) -> None:
        if self.kind == "registered":
            if not self.task_type:
                raise ValueError(
                    f"TaskProposal(kind='registered') requires non-empty task_type "
                    f"(task_id={self.task_id!r})"
                )
        elif self.kind == "agent_authored":
            if self.code_artifact is None:
                raise ValueError(
                    f"TaskProposal(kind='agent_authored') requires code_artifact "
                    f"as a non-None Path (task_id={self.task_id!r})"
                )
        else:
            raise ValueError(
                f"TaskProposal.kind must be 'registered' or 'agent_authored'; got {self.kind!r}"
            )


# ── PlanProposal ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanProposal:
    """A whole experiment plan, ready to be compiled into a runnable spec.

    ``proposal_id`` is a content hash set in ``__post_init__`` — the
    canonical-JSON serialization is **order-independent** for the
    set-like collections (``task_proposals``, ``depends_on``, ``Mapping``
    key order), so two logically-equal plans always hash the same.

    Lineage invariant: ``parent_proposal_id is None`` ⇔ ``revision == 0``.
    The chain of ``parent_proposal_id`` plays the same role as Run's
    ``replanned_from`` — it makes plan revisions traceable when a
    rejected plan is revised.
    """

    name: str
    task_proposals: tuple[TaskProposal, ...] = ()
    sanity_specs: tuple[SanitySpec, ...] = ()
    parallels: tuple[ParallelSpec, ...] = ()
    loops: tuple[LoopSpec, ...] = ()
    branches: tuple[BranchSpec, ...] = ()
    sweeps: tuple[SweepSpec, ...] = ()
    intervention_points: tuple[InterventionPoint, ...] = ()
    parent_proposal_id: str | None = None
    revision: int = 0
    proposal_id: str = field(init=False, default="", compare=False)

    def __post_init__(self) -> None:
        # Lineage validation: parent ⇔ revision > 0
        if self.parent_proposal_id is None and self.revision != 0:
            raise ValueError(
                f"PlanProposal: parent_proposal_id is None implies revision == 0; "
                f"got revision={self.revision}"
            )
        if self.parent_proposal_id is not None and self.revision <= 0:
            raise ValueError(
                f"PlanProposal: parent_proposal_id is not None implies revision >= 1; "
                f"got revision={self.revision}"
            )
        # Compute and freeze proposal_id
        digest = _hash_proposal(self)
        object.__setattr__(self, "proposal_id", digest)


# ── ParameterizedWorkflowSpec (compile output) ────────────────────────────────


@dataclass(frozen=True)
class ParameterizedWorkflowSpec:
    """Output of ``WorkflowCompiler.proposal_to_spec``.

    A thin wrapper carrying the compiled topology plus a deterministic
    ``workflow_id``. Part A.1 does not let it run yet — actually
    executing through ``WorkflowSpec.execute()`` is Part B.
    """

    workflow_id: str
    name: str
    tasks: tuple[Any, ...]
    sanity_specs: tuple[SanitySpec, ...]
    control_flow: Mapping[str, Any]


# ── Canonical hashing ─────────────────────────────────────────────────────────


def _canonical(obj: Any) -> Any:
    """Recursively normalize for canonical JSON.

    - ``Mapping`` → dict with string-sorted keys
    - tuple / list → list with element order preserved (caller sorts
      where order shouldn't matter)
    - frozen dataclass → dict of fields, ``proposal_id`` excluded so a
      ``PlanProposal`` can hash itself without circular reference
    - primitives → unchanged
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name != "proposal_id" and f.init
        }
    raise TypeError(f"_canonical: unsupported type {type(obj).__name__!r}")


def _canonical_task(task: TaskProposal) -> dict[str, Any]:
    """Canonicalize a TaskProposal with order-independent ``depends_on``."""
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "task_type": task.task_type,
        "config": _canonical(dict(task.config)),
        "depends_on": sorted(task.depends_on),
        "code_artifact": _canonical(task.code_artifact),
    }


def _sorted_canonicals(items: tuple[Any, ...]) -> list[Any]:
    """Canonicalize each item then sort by stable JSON repr.

    Used for set-like collections (sanity_specs, parallels, loops,
    branches, sweeps) where order should not affect ``proposal_id``.
    """
    canonicals = [_canonical(it) for it in items]
    canonicals.sort(key=lambda d: json.dumps(d, sort_keys=True))
    return canonicals


def _hash_proposal(plan: PlanProposal) -> str:
    """Compute the 16-hex content hash for ``plan``."""
    body = {
        "name": plan.name,
        "task_proposals": sorted(
            (_canonical_task(t) for t in plan.task_proposals),
            key=lambda d: d["task_id"],
        ),
        "sanity_specs": _sorted_canonicals(plan.sanity_specs),
        "parallels": _sorted_canonicals(plan.parallels),
        "loops": _sorted_canonicals(plan.loops),
        "branches": _sorted_canonicals(plan.branches),
        "sweeps": _sorted_canonicals(plan.sweeps),
        "intervention_points": sorted(
            (_canonical(ip) for ip in plan.intervention_points),
            key=lambda d: d["name"],
        ),
        "parent_proposal_id": plan.parent_proposal_id,
        "revision": plan.revision,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[
        :_HASH_HEX_LEN
    ]
