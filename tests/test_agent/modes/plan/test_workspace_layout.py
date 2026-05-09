"""Contract tests for the PlanMode experiment-workspace layout helper.

Covers acceptance criteria ac-001..ac-009 for sub-spec
``planmode-workspace-pipeline-03-experiment-workspace-layout``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molexp.agent.modes.plan import (
    AGENT_PLAN_EXPERIMENTS_KIND,
    CheckResult,
    PlanManifest,
    PlanWorkspaceHandle,
    ValidationReport,
)
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws")


# ── PlanWorkspaceHandle.materialize / attach (ac-002) ──────────────────────


def test_materialize_uses_agent_plan_experiments_subsystem(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace)
    assert isinstance(handle.plan_id, str)
    assert handle.plan_id != ""

    expected_root = workspace.subsystem_store(AGENT_PLAN_EXPERIMENTS_KIND).dir() / handle.plan_id
    assert handle.root() == expected_root


def test_materialize_twice_yields_distinct_plan_ids(workspace: Workspace) -> None:
    h1 = PlanWorkspaceHandle.materialize(workspace)
    h2 = PlanWorkspaceHandle.materialize(workspace)
    assert h1.plan_id != h2.plan_id


def test_attach_reuses_existing_plan_id(workspace: Workspace) -> None:
    h1 = PlanWorkspaceHandle.materialize(workspace, plan_id="plan_fixed")
    h2 = PlanWorkspaceHandle.attach(workspace, plan_id="plan_fixed")
    assert h1.plan_id == h2.plan_id == "plan_fixed"
    assert h1.root() == h2.root()


def test_materialize_with_explicit_plan_id_uses_it(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="my_plan")
    assert handle.plan_id == "my_plan"


def test_construction_is_side_effect_free_until_first_dir_call(
    workspace: Workspace,
) -> None:
    PlanWorkspaceHandle.materialize(workspace, plan_id="lazy_plan")
    # The subsystem root may have been created by Workspace's eager
    # init, but the per-plan directory should not exist yet.
    expected_per_plan = workspace.subsystem_store(AGENT_PLAN_EXPERIMENTS_KIND).dir() / "lazy_plan"
    assert not expected_per_plan.exists()


# ── *_dir() lazy creation (ac-003) ─────────────────────────────────────────


_DIR_METHOD_TO_REL_PATH: list[tuple[str, tuple[str, ...]]] = [
    ("report_dir", ("report",)),
    ("plan_dir", ("plan",)),
    ("ir_dir", ("ir",)),
    ("tasks_ir_dir", ("ir", "tasks")),
    ("src_dir", ("src",)),
    ("experiment_pkg_dir", ("src", "experiment")),
    ("tasks_pkg_dir", ("src", "experiment", "tasks")),
    ("tests_dir", ("tests",)),
    ("configs_dir", ("configs",)),
    ("runs_dir", ("runs",)),
    ("results_dir", ("results",)),
]


@pytest.mark.parametrize(("method_name", "rel_segments"), _DIR_METHOD_TO_REL_PATH)
def test_dir_method_creates_documented_path_lazily(
    workspace: Workspace,
    method_name: str,
    rel_segments: tuple[str, ...],
) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id=f"plan_{method_name}")
    expected = handle.root() / Path(*rel_segments) if rel_segments else handle.root()
    # Pre-condition: the deepest segment does not exist yet (root() in
    # the previous line created the plan root, which is fine — the
    # contract is that the *_dir target itself is created on first call).
    if rel_segments:
        assert not expected.exists()

    method = getattr(handle, method_name)
    result = method()

    assert result == expected
    assert result.exists()
    assert result.is_dir()


def test_dir_methods_are_idempotent(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="idem")
    p1 = handle.report_dir()
    p2 = handle.report_dir()
    assert p1 == p2
    assert p1.exists()


def test_deep_dir_cascades_parents(workspace: Workspace) -> None:
    """tasks_pkg_dir() works without first calling experiment_pkg_dir()."""
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="cascade")
    target = handle.tasks_pkg_dir()
    assert target.is_dir()
    assert target.parent.name == "experiment"
    assert target.parent.parent.name == "src"


# ── Path-only helpers (ac-004) ─────────────────────────────────────────────


def test_manifest_path_does_not_create_file(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="m")
    path = handle.manifest_path()
    assert path.name == "manifest.yaml"
    assert not path.exists()


def test_validation_report_path_does_not_create_file(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="v")
    path = handle.validation_report_path()
    assert path.name == "validation_report.md"
    assert not path.exists()


# ── PlanManifest (ac-005) ──────────────────────────────────────────────────


def _sample_manifest(plan_id: str = "p") -> PlanManifest:
    return PlanManifest(
        plan_id=plan_id,
        created_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        report_source="report/original.md",
        workflow_ir_path=Path("ir/workflow.yaml"),
        task_ir_paths=(Path("ir/tasks/a.yaml"), Path("ir/tasks/b.yaml")),
    )


def test_plan_manifest_is_frozen() -> None:
    m = _sample_manifest()
    assert m.model_config["frozen"] is True
    with pytest.raises(ValidationError):
        m.plan_id = "other"  # type: ignore[misc]


def test_plan_manifest_status_default_is_draft() -> None:
    m = _sample_manifest()
    assert m.status == "draft"


def test_plan_manifest_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        PlanManifest(
            plan_id="p",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
            report_source="r",
            workflow_ir_path=Path("ir/workflow.yaml"),
            status="invalid",  # type: ignore[arg-type]
        )


def test_plan_manifest_accepts_documented_status_values() -> None:
    for status in ("draft", "validated", "approved"):
        m = PlanManifest(
            plan_id="p",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
            report_source="r",
            workflow_ir_path=Path("ir/workflow.yaml"),
            status=status,  # type: ignore[arg-type]
        )
        assert m.status == status


def test_plan_manifest_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanManifest(
            plan_id="p",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
            report_source="r",
            workflow_ir_path=Path("ir/workflow.yaml"),
            stray="x",  # type: ignore[call-arg]
        )


def test_plan_manifest_model_policy_snapshot_default_none() -> None:
    m = _sample_manifest()
    assert m.model_policy_snapshot is None


# ── write_manifest YAML round-trip (ac-006) ────────────────────────────────


def test_write_manifest_round_trips_via_safe_load(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="rt_manifest")
    manifest = _sample_manifest(plan_id="rt_manifest")
    written_path = handle.write_manifest(manifest)
    assert written_path == handle.manifest_path()
    assert written_path.exists()

    loaded = yaml.safe_load(written_path.read_text())
    expected = manifest.model_dump(mode="json")
    assert loaded == expected


# ── ValidationReport.to_markdown + write_validation_report (ac-007) ────────


def test_validation_report_to_markdown_contains_required_text() -> None:
    report = ValidationReport(
        passed=False,
        checks=(CheckResult(name="x", passed=False, severity="error", detail="y"),),
        summary="s",
    )
    md = report.to_markdown()
    assert "x" in md
    assert "error" in md
    assert "s" in md
    # Header line must be present.
    assert md.startswith("# ")


def test_validation_report_to_markdown_handles_empty_checks() -> None:
    report = ValidationReport(passed=True, summary="all green")
    md = report.to_markdown()
    assert "all green" in md
    assert md.startswith("# Validation report — passed")


def test_validation_report_to_markdown_passed_state_in_header() -> None:
    failed = ValidationReport(passed=False, summary="x")
    assert "failed" in failed.to_markdown().split("\n", 1)[0]

    passed = ValidationReport(passed=True, summary="x")
    assert "passed" in passed.to_markdown().split("\n", 1)[0]


def test_write_validation_report_persists_atomically(workspace: Workspace) -> None:
    handle = PlanWorkspaceHandle.materialize(workspace, plan_id="rt_report")
    report = ValidationReport(
        passed=False,
        checks=(
            CheckResult(name="ir_parseable", passed=True, severity="info"),
            CheckResult(
                name="impl_present",
                passed=False,
                severity="error",
                detail="task_a missing",
            ),
        ),
        summary="1 of 2 passed",
    )
    written_path = handle.write_validation_report(report)
    assert written_path == handle.validation_report_path()
    assert written_path.exists()
    text = written_path.read_text()
    assert "ir_parseable" in text
    assert "impl_present" in text
    assert "task_a missing" in text
    assert "1 of 2 passed" in text


# ── Re-exports (ac-009) ────────────────────────────────────────────────────


def test_public_names_reachable_from_modes_plan() -> None:
    import molexp.agent.modes.plan as plan_pkg

    for name in (
        "PlanWorkspaceHandle",
        "PlanManifest",
        "ValidationReport",
        "CheckResult",
    ):
        assert hasattr(plan_pkg, name)
        assert name in plan_pkg.__all__


# ── CheckResult ────────────────────────────────────────────────────────────


def test_check_result_default_detail_is_empty_string() -> None:
    cr = CheckResult(name="x", passed=True, severity="info")
    assert cr.detail == ""


def test_check_result_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        CheckResult(
            name="x",
            passed=False,
            severity="critical",  # type: ignore[arg-type]
        )
