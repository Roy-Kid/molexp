"""Contract + behavior tests for the built-in run-lifecycle capability catalog.

Feature under test (vision-loop-07, Design §1): ``molexp.harness.capabilities.
lifecycle`` exposes a frozen five-entry ``LIFECYCLE_CAPABILITIES`` tuple +
``lifecycle_capabilities()`` accessor. Every entry is destructive (non-empty
``side_effects`` — the link-03 gate and the NL ChangeProposal branch both key
on it) and names a lazily-resolved dotted ``callable_path``.

Mirrors the curation-catalog drift-guard idioms
(``tests/test_harness/test_curation_capability_catalog.py``): iterate the
catalog and cross-check each entry against the live callable its
``callable_path`` resolves to, rather than pinning per-entry literals.

The behavior section exercises the resolved callables themselves: the execute
family refuses non-local ``metadata.target`` loudly (local-only v1; one shared
``_refuse_non_local`` guard), and the two-phase prune capability actually
deletes failed attempts (catalog → callable → ``workspace.prune`` cores, one
path).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from molexp.harness.capabilities import lifecycle_capabilities
from molexp.harness.capability import resolve_callable
from molexp.harness.schemas import ToolCapability
from molexp.workspace import Workspace
from molexp.workspace.models import ComputeTarget, ExecutionRecord, RunStatus
from molexp.workspace.run import Run
from molexp.workspace.targets import add_target

# ── shared constants ────────────────────────────────────────────────────────

#: The five lifecycle verbs — spec table (vision-loop-07 Design §1).
EXPECTED_IDS: frozenset[str] = frozenset(
    {
        "molexp.lifecycle.run_execute",
        "molexp.lifecycle.run_resume",
        "molexp.lifecycle.run_rerun",
        "molexp.lifecycle.run_cancel",
        "molexp.lifecycle.runs_prune",
    }
)

#: Exact side-effect tokens per entry (spec table) — ALL non-empty, so both
#: gates (pre-dispatch side-effect approvals + the NL ChangeProposal branch)
#: fire automatically.
EXPECTED_SIDE_EFFECTS: dict[str, list[str]] = {
    "molexp.lifecycle.run_execute": ["execute:run"],
    "molexp.lifecycle.run_resume": ["execute:run"],
    "molexp.lifecycle.run_rerun": ["execute:run"],
    "molexp.lifecycle.run_cancel": ["cancel:run"],
    "molexp.lifecycle.runs_prune": ["delete:executions"],
}


# ── module-level helpers ─────────────────────────────────────────────────────


def _by_id(capability_id: str) -> ToolCapability:
    """Return the catalog entry with *capability_id* (asserted present)."""
    matches = [entry for entry in lifecycle_capabilities() if entry.id == capability_id]
    assert len(matches) == 1, f"expected exactly one entry {capability_id!r}"
    return matches[0]


def _resolved(capability_id: str) -> Callable[..., Any]:
    """Resolve an entry's ``callable_path`` through the harness resolver."""
    entry = _by_id(capability_id)
    assert entry.callable_path is not None, capability_id
    fn = resolve_callable(entry.callable_path)
    assert callable(fn), f"{capability_id} -> {entry.callable_path} is not callable"
    return fn


def _signature_properties(fn: Callable[..., Any]) -> tuple[set[str], set[str]]:
    """Derive (properties, required) from a callable's live signature."""
    properties: set[str] = set()
    required: set[str] = set()
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in {"self", "cls"}:
            continue
        properties.add(name)
        if param.default is inspect.Parameter.empty:
            required.add(name)
    return properties, required


def _remote_run(tmp_path: Path, *, status: RunStatus = RunStatus.PENDING) -> Run:
    """A run whose ``metadata.target`` names a registered non-local target."""
    ws = Workspace(root=tmp_path, name="remote-lab")
    add_target(
        ws,
        ComputeTarget(name="hpc1", host="cluster.example.org", scratch_root="/scratch/molexp"),
    )
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1}, target="hpc1")
    run.materialize()
    if status is not RunStatus.PENDING:
        run.update_ops(lambda s: s.model_copy(update={"status": status}))
    return run


# ── catalog contract ──────────────────────────────────────────────────────────


def test_ids_are_exactly_the_five_lifecycle_verbs() -> None:
    assert {entry.id for entry in lifecycle_capabilities()} == EXPECTED_IDS


def test_side_effect_tokens_match_the_spec_table() -> None:
    """ALL five are destructive — the gates key on non-empty ``side_effects``."""
    for entry in lifecycle_capabilities():
        assert entry.side_effects == EXPECTED_SIDE_EFFECTS[entry.id], entry.id


def test_input_schema_tracks_the_live_signature() -> None:
    """Drift guard — every ``callable_path`` resolves and its declared
    ``input_schema`` tracks the live signature."""
    for entry in lifecycle_capabilities():
        assert entry.callable_path is not None, entry.id
        fn = resolve_callable(entry.callable_path)
        expected_props, expected_required = _signature_properties(fn)
        declared_props = set(entry.input_schema["properties"].keys())
        assert declared_props == expected_props, (
            f"{entry.id}: declared properties {declared_props} != signature {expected_props}"
        )
        declared_required = set(entry.input_schema.get("required", []))
        assert declared_required == expected_required, (
            f"{entry.id}: declared required {declared_required} != signature {expected_required}"
        )


# ── behavior: local-only v1 (loud refusal) ────────────────────────────────────


def test_run_execute_refuses_remote_target(tmp_path: Path) -> None:
    """The execute family refuses a non-local ``metadata.target`` loudly,
    before any execution state is created (shared ``_refuse_non_local`` guard
    for execute / resume / rerun)."""
    run = _remote_run(tmp_path, status=RunStatus.PENDING)
    fn = _resolved("molexp.lifecycle.run_execute")
    with pytest.raises(ValueError, match="hpc1"):
        fn(run)
    assert run.status == "pending"
    assert not (Path(str(run.run_dir)) / "executions").exists()


# ── behavior: two-phase prune through the catalog callable ───────────────────


def _seeded_run(tmp_path: Path) -> Run:
    ws = Workspace(root=tmp_path, name="prune-cap-lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1})
    run.materialize()
    history: list[ExecutionRecord] = []
    for i, status in enumerate(("succeeded", "failed"), start=1):
        exec_id = f"exec-{run.id}" if i == 1 else f"exec-{run.id}-{i}"
        (Path(str(run.run_dir)) / "executions" / exec_id).mkdir(parents=True)
        history.append(
            ExecutionRecord(
                execution_id=exec_id,
                started_at=datetime(2026, 7, 1, 10, i),
                finished_at=datetime(2026, 7, 1, 10, i + 1),
                status=status,
            )
        )
    run.update_ops(
        lambda s: s.model_copy(update={"executions": tuple(history), "status": RunStatus.SUCCEEDED})
    )
    return run


def test_prune_capability_deletes_failed_attempts(tmp_path: Path) -> None:
    """``runs_prune``'s callable drives the workspace two-phase core end-to-end."""
    run = _seeded_run(tmp_path)
    fn = _resolved("molexp.lifecycle.runs_prune")
    result = fn(run, statuses=["failed"])
    exec_root = Path(str(run.run_dir)) / "executions"
    assert sorted(p.name for p in exec_root.iterdir()) == [f"exec-{run.id}"]
    assert [rec.execution_id for rec in run.execution_history] == [f"exec-{run.id}"]
    assert result["pruned"] == [f"exec-{run.id}-2"]
