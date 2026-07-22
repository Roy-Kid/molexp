"""Contract + behavior tests for the built-in run-lifecycle capability catalog.

Feature under test (vision-loop-07, Design §1): ``molexp.harness.capabilities.
lifecycle`` exposes ``lifecycle_capabilities()`` — five destructive entries
(non-empty ``side_effects``, so the link-03 gate and the NL ChangeProposal
branch both fire) each naming a lazily-resolved dotted ``callable_path``.

The behavior section drives the resolved handler callables
(``harness.actions.handlers.lifecycle``, no dedicated test file): the execute
family refuses non-local ``metadata.target`` loudly (local-only v1), and the
two-phase prune capability actually deletes failed attempts.
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

#: The five lifecycle verbs + their side-effect tokens (spec table §1). ALL
#: non-empty, so both gates fire automatically.
EXPECTED_SIDE_EFFECTS: dict[str, list[str]] = {
    "molexp.lifecycle.run_execute": ["execute:run"],
    "molexp.lifecycle.run_resume": ["execute:run"],
    "molexp.lifecycle.run_rerun": ["execute:run"],
    "molexp.lifecycle.run_cancel": ["cancel:run"],
    "molexp.lifecycle.runs_prune": ["delete:executions"],
}


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


class TestLifecycleCatalog:
    """The frozen five-entry catalog: identity, destructiveness, schema drift."""

    def test_catalog_is_the_five_destructive_lifecycle_verbs(self) -> None:
        """Exactly the five verbs, every one carrying its non-empty side-effect
        tokens (the gates key on non-empty ``side_effects``)."""
        catalog = {entry.id: entry for entry in lifecycle_capabilities()}
        assert set(catalog) == set(EXPECTED_SIDE_EFFECTS)
        for cap_id, entry in catalog.items():
            assert entry.side_effects == EXPECTED_SIDE_EFFECTS[cap_id], cap_id

    def test_input_schema_tracks_the_live_signature(self) -> None:
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


class TestLifecycleHandlers:
    """The resolved handler callables — local-only guard + two-phase prune."""

    @staticmethod
    def _remote_run(tmp_path: Path) -> Run:
        """A pending run whose ``metadata.target`` names a non-local target."""
        ws = Workspace(root=tmp_path, name="remote-lab")
        add_target(
            ws,
            ComputeTarget(name="hpc1", host="cluster.example.org", scratch_root="/scratch/molexp"),
        )
        exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
        run = exp.add_run(params={"seed": 1}, target="hpc1")
        run.materialize()
        return run

    @staticmethod
    def _seeded_run(tmp_path: Path) -> Run:
        """A succeeded run with one succeeded + one failed execution attempt."""
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
            lambda s: s.model_copy(
                update={"executions": tuple(history), "status": RunStatus.SUCCEEDED}
            )
        )
        return run

    def test_run_execute_refuses_remote_target(self, tmp_path: Path) -> None:
        """The execute family refuses a non-local ``metadata.target`` loudly,
        before any execution state is created (shared ``_refuse_non_local``)."""
        run = self._remote_run(tmp_path)
        fn = _resolved("molexp.lifecycle.run_execute")
        with pytest.raises(ValueError, match="hpc1"):
            fn(run)
        assert run.status == "pending"
        assert not (Path(str(run.run_dir)) / "executions").exists()

    def test_prune_capability_deletes_failed_attempts(self, tmp_path: Path) -> None:
        """``runs_prune``'s callable drives the workspace two-phase core
        end-to-end and returns the pruned execution ids."""
        run = self._seeded_run(tmp_path)
        fn = _resolved("molexp.lifecycle.runs_prune")
        result = fn(run, statuses=["failed"])
        exec_root = Path(str(run.run_dir)) / "executions"
        assert sorted(p.name for p in exec_root.iterdir()) == [f"exec-{run.id}"]
        assert [rec.execution_id for rec in run.execution_history] == [f"exec-{run.id}"]
        assert result["pruned"] == [f"exec-{run.id}-2"]
