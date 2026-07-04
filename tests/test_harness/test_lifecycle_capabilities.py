"""Contract + behavior tests for the built-in run-lifecycle capability catalog.

Feature under test (vision-loop-07, Design §1): ``molexp.harness.capabilities.
lifecycle`` exposes a frozen five-entry ``LIFECYCLE_CAPABILITIES`` tuple +
``lifecycle_capabilities()`` accessor. Every entry is destructive (non-empty
``side_effects`` — the link-03 gate and the NL ChangeProposal branch both key
on it), names a lazily-resolved dotted ``callable_path``, and teaches the LLM
planner its verb domain in ``description``.

Mirrors the curation-catalog drift-guard idioms
(``tests/test_harness/test_curation_capability_catalog.py``): iterate the
catalog and cross-check each entry against the live callable its
``callable_path`` resolves to, rather than pinning per-entry literals.

The behavior section exercises the resolved callables themselves: the execute
family refuses non-local ``metadata.target`` loudly (local-only v1), and the
two-phase prune capability actually deletes failed attempts (catalog →
callable → ``workspace.prune`` cores, one path).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pydantic
import pytest

from molexp.harness.capabilities import lifecycle_capabilities
from molexp.harness.capabilities.lifecycle import LIFECYCLE_CAPABILITIES
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

#: Verb-domain vocabulary each description must teach the LLM planner
#: (lower-cased substring match).
EXPECTED_DOMAIN_WORDS: dict[str, tuple[str, ...]] = {
    "molexp.lifecycle.run_execute": ("pending",),
    "molexp.lifecycle.run_resume": ("failed", "cancelled"),
    "molexp.lifecycle.run_rerun": ("failed", "cancelled"),
    "molexp.lifecycle.run_cancel": ("running",),
    "molexp.lifecycle.runs_prune": ("execution",),
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


# ── catalog shape ─────────────────────────────────────────────────────────────


class TestCatalogShape:
    """The accessor + backing tuple: five frozen ``ToolCapability`` entries."""

    def test_backing_tuple_is_frozen_and_has_five_entries(self) -> None:
        assert isinstance(LIFECYCLE_CAPABILITIES, tuple)
        assert len(LIFECYCLE_CAPABILITIES) == 5
        assert all(isinstance(entry, ToolCapability) for entry in LIFECYCLE_CAPABILITIES)

    def test_accessor_mirrors_the_backing_tuple(self) -> None:
        assert list(lifecycle_capabilities()) == list(LIFECYCLE_CAPABILITIES)

    def test_entries_are_frozen(self) -> None:
        entry = next(iter(lifecycle_capabilities()))
        with pytest.raises(pydantic.ValidationError):
            entry.id = "mutated.id"  # type: ignore[misc]

    def test_ids_are_exactly_the_five_lifecycle_verbs(self) -> None:
        assert {entry.id for entry in lifecycle_capabilities()} == EXPECTED_IDS

    def test_ids_are_unique(self) -> None:
        catalog = list(lifecycle_capabilities())
        assert len({entry.id for entry in catalog}) == len(catalog)

    def test_every_id_is_lifecycle_namespaced(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.id.startswith("molexp.lifecycle."), entry.id

    def test_every_package_is_molexp(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.package == "molexp", entry.id

    def test_id_differs_from_callable_path(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.id != entry.callable_path, entry.id


# ── side-effects contract ─────────────────────────────────────────────────────


class TestSideEffectsContract:
    """ALL five are destructive — the gates key on non-empty ``side_effects``."""

    def test_every_entry_declares_non_empty_side_effects(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.side_effects, f"{entry.id} must be gated (non-empty side_effects)"
            assert all(isinstance(token, str) and token for token in entry.side_effects), entry.id

    def test_side_effect_tokens_match_the_spec_table(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.side_effects == EXPECTED_SIDE_EFFECTS[entry.id], entry.id


# ── callable_path resolvability ───────────────────────────────────────────────


class TestCallablePathResolves:
    """Drift guard — every ``callable_path`` resolves via the harness resolver."""

    def test_each_callable_path_resolves_to_a_callable(self) -> None:
        for entry in lifecycle_capabilities():
            assert entry.callable_path is not None, entry.id
            resolved = resolve_callable(entry.callable_path)
            assert callable(resolved), f"{entry.id} -> {entry.callable_path} is not callable"

    def test_input_schema_tracks_the_live_signature(self) -> None:
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
                f"{entry.id}: declared required {declared_required} != "
                f"signature {expected_required}"
            )


# ── verb-domain text ──────────────────────────────────────────────────────────


class TestVerbDomainDescriptions:
    """Descriptions teach the LLM planner each verb's disjoint domain."""

    def test_each_description_names_its_verb_domain(self) -> None:
        for entry in lifecycle_capabilities():
            description = entry.description.lower()
            for word in EXPECTED_DOMAIN_WORDS[entry.id]:
                assert word in description, (
                    f"{entry.id}: description must name its verb domain "
                    f"(missing {word!r}): {entry.description!r}"
                )

    def test_each_description_is_substantive(self) -> None:
        for entry in lifecycle_capabilities():
            assert len(entry.description.strip()) >= 20, entry.id


# ── behavior: local-only v1 (loud refusal) ────────────────────────────────────


class TestNonLocalTargetRefusal:
    """The execute family refuses a non-local ``metadata.target`` loudly.

    Each verb gets a run inside its own status domain, so only the
    target refusal can fire — proving the refusal precedes execution.
    """

    def test_run_execute_refuses_remote_target(self, tmp_path: Path) -> None:
        run = _remote_run(tmp_path, status=RunStatus.PENDING)
        fn = _resolved("molexp.lifecycle.run_execute")
        with pytest.raises(ValueError, match="hpc1"):
            fn(run)
        assert run.status == "pending"
        assert not (Path(str(run.run_dir)) / "executions").exists()

    def test_run_resume_refuses_remote_target(self, tmp_path: Path) -> None:
        run = _remote_run(tmp_path, status=RunStatus.FAILED)
        fn = _resolved("molexp.lifecycle.run_resume")
        with pytest.raises(ValueError, match="hpc1"):
            fn(run)
        assert run.status == "failed"

    def test_run_rerun_refuses_remote_target(self, tmp_path: Path) -> None:
        run = _remote_run(tmp_path, status=RunStatus.FAILED)
        fn = _resolved("molexp.lifecycle.run_rerun")
        with pytest.raises(ValueError, match="hpc1"):
            fn(run)
        assert run.status == "failed"

    def test_refusal_names_the_sanctioned_dispatch_surface(self, tmp_path: Path) -> None:
        run = _remote_run(tmp_path, status=RunStatus.PENDING)
        fn = _resolved("molexp.lifecycle.run_execute")
        with pytest.raises(ValueError, match="molexp run"):
            fn(run)


# ── behavior: two-phase prune through the catalog callable ───────────────────


class TestPruneCapabilityBehavior:
    """``runs_prune``'s callable drives the workspace two-phase core end-to-end."""

    @staticmethod
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
            lambda s: s.model_copy(
                update={"executions": tuple(history), "status": RunStatus.SUCCEEDED}
            )
        )
        return run

    def test_prune_capability_deletes_failed_attempts(self, tmp_path: Path) -> None:
        run = self._seeded_run(tmp_path)
        fn = _resolved("molexp.lifecycle.runs_prune")
        result = fn(run, statuses=["failed"])
        exec_root = Path(str(run.run_dir)) / "executions"
        assert sorted(p.name for p in exec_root.iterdir()) == [f"exec-{run.id}"]
        assert [rec.execution_id for rec in run.execution_history] == [f"exec-{run.id}"]
        assert result["pruned"] == [f"exec-{run.id}-2"]

    def test_prune_capability_leaves_unselected_attempts_alone(self, tmp_path: Path) -> None:
        run = self._seeded_run(tmp_path)
        fn = _resolved("molexp.lifecycle.runs_prune")
        fn(run, statuses=["cancelled"])  # nothing matches
        exec_root = Path(str(run.run_dir)) / "executions"
        assert len(list(exec_root.iterdir())) == 2
        assert len(run.execution_history) == 2
