"""Contract tests for the PlanMode plan-folder layout helper.

:class:`PlanFolder` is the agent-layer :class:`molexp.workspace.Folder`
subclass that owns the on-disk layout of a single PlanMode plan. It
replaces the legacy ``PlanWorkspaceHandle`` (and its
``.subsystems/agent.plan-experiments/<id>/`` storage path).

Covers the lazy directory semantics, atomic writers, and YAML
round-trips that the rest of the plan-mode pipeline relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from molexp.agent.modes.plan import PlanFolder
from molexp.agent.modes.plan.plan_folder import (
    AGENT_PLAN_KIND,
    CheckResult,
    PlanManifest,
    ValidationReport,
)
from molexp.agent.modes.plan.capability import (
    CapabilityEvidence,
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
    MissingCapability,
)
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws")


def _mount(workspace: Workspace, plan_id: str | None = None) -> PlanFolder:
    """Mount a fresh :class:`PlanFolder` on ``workspace`` and return it."""
    return cast(PlanFolder, workspace.add_folder(PlanFolder(name=plan_id)))


# ── PlanFolder construction / mounting ─────────────────────────────────────


def test_mount_uses_plans_subdir(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="my-plan")
    assert plan_folder.plan_id == "my-plan"
    assert plan_folder.kind == AGENT_PLAN_KIND
    assert plan_folder.root() == workspace.root / "plans" / "my-plan"


def test_mount_without_name_auto_generates_plan_id(workspace: Workspace) -> None:
    plan_folder = _mount(workspace)
    assert isinstance(plan_folder.plan_id, str) and plan_folder.plan_id


def test_mount_twice_is_idempotent(workspace: Workspace) -> None:
    """``add_folder`` returns the cached instance on slug collision."""
    p1 = _mount(workspace, plan_id="dup")
    p2 = _mount(workspace, plan_id="dup")
    assert p1 is p2


# ── *_dir() lazy creation ──────────────────────────────────────────────────


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
]


@pytest.mark.parametrize(("method_name", "rel_segments"), _DIR_METHOD_TO_REL_PATH)
def test_dir_method_creates_documented_path_lazily(
    workspace: Workspace,
    method_name: str,
    rel_segments: tuple[str, ...],
) -> None:
    plan_folder = _mount(workspace, plan_id=f"plan-{method_name.replace('_', '-')}")
    expected = plan_folder.root() / Path(*rel_segments) if rel_segments else plan_folder.root()
    if rel_segments:
        assert not expected.exists()

    method = getattr(plan_folder, method_name)
    result = method()

    assert result == expected
    assert result.exists()
    assert result.is_dir()


def test_dir_methods_are_idempotent(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="idem")
    p1 = plan_folder.report_dir()
    p2 = plan_folder.report_dir()
    assert p1 == p2
    assert p1.exists()


def test_deep_dir_cascades_parents(workspace: Workspace) -> None:
    """tasks_pkg_dir() works without first calling experiment_pkg_dir()."""
    plan_folder = _mount(workspace, plan_id="cascade")
    target = plan_folder.tasks_pkg_dir()
    assert target.is_dir()
    assert target.parent.name == "experiment"
    assert target.parent.parent.name == "src"


# ── Path-only helpers ──────────────────────────────────────────────────────


def test_manifest_path_does_not_create_file(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="m")
    path = plan_folder.manifest_path()
    assert path.name == "manifest.yaml"
    assert not path.exists()


def test_validation_report_path_does_not_create_file(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="v")
    path = plan_folder.validation_report_path()
    assert path.name == "validation_report.md"
    assert not path.exists()


# ── PlanManifest ──────────────────────────────────────────────────────────


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
    for status in (
        "draft",
        "validated",
        "validation_failed",
        "ready_for_review",
        "approved",
        "approved_with_override",
        "ready_for_run",
        "pending_review",
    ):
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


# ── write_manifest YAML round-trip ────────────────────────────────────────


def test_write_manifest_round_trips_via_safe_load(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="rt-manifest")
    manifest = _sample_manifest(plan_id="rt-manifest")
    written_path = plan_folder.write_manifest(manifest)
    assert written_path == plan_folder.manifest_path()
    assert written_path.exists()

    loaded = yaml.safe_load(written_path.read_text())
    expected = manifest.model_dump(mode="json")
    assert loaded == expected


# ── ValidationReport.to_markdown + write_validation_report ─────────────────


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
    plan_folder = _mount(workspace, plan_id="rt-report")
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
    written_path = plan_folder.write_validation_report(report)
    assert written_path == plan_folder.validation_report_path()
    assert written_path.exists()
    data_path = plan_folder.validation_report_data_path()
    assert data_path.exists()
    text = written_path.read_text()
    assert "ir_parseable" in text
    assert "impl_present" in text
    assert "task_a missing" in text
    assert "1 of 2 passed" in text
    data = yaml.safe_load(data_path.read_text())
    assert data["passed"] is False
    assert data["checks"][1]["name"] == "impl_present"


# ── Re-exports ────────────────────────────────────────────────────────────


def test_public_names_reachable_from_modes_plan() -> None:
    import molexp.agent.modes.plan as plan_pkg

    for name in (
        "PlanFolder",
        "PlanMode",
        "PlanModeConfig",
        "PlanResult",
        "PlanRunHandoff",
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


# ── repairs_dir / latest_decision_path / archive ──────────────────────────


def test_repairs_dir_creates_isolated_iteration_directories(workspace: Workspace) -> None:
    """``repairs_dir(n)`` creates ``<plan>/repairs/iter-<n>/`` lazily."""
    plan_folder = _mount(workspace, plan_id="rep")
    iter0 = plan_folder.repairs_dir(0)
    iter1 = plan_folder.repairs_dir(1)
    assert iter0 != iter1
    assert iter0.exists() and iter0.is_dir()
    assert iter1.exists() and iter1.is_dir()
    assert iter0.name == "iter-0"
    assert iter1.name == "iter-1"
    assert iter0.parent == iter1.parent
    assert iter0.parent.name == "repairs"
    assert iter0.parent.parent == plan_folder.root()


def test_latest_decision_path_does_not_create_file(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="ld")
    p = plan_folder.latest_decision_path()
    assert p == plan_folder.root() / "repairs" / "latest_decision.yaml"
    assert not p.exists()


def test_archive_artifacts_for_repair_copies_five_subtrees(workspace: Workspace) -> None:
    """Live ``report/ plan/ ir/ src/ tests/`` snapshotted under ``repairs/iter-<n>/``."""
    plan_folder = _mount(workspace, plan_id="arch")
    for subdir, fname in (
        (plan_folder.report_dir(), "digest.md"),
        (plan_folder.plan_dir(), "implementation_plan.md"),
        (plan_folder.ir_dir(), "workflow.yaml"),
        (plan_folder.src_dir(), "marker.py"),
        (plan_folder.tests_dir(), "marker.py"),
    ):
        (subdir / fname).write_text("v1")

    plan_folder.archive_artifacts_for_repair(0)
    iter0 = plan_folder.repairs_dir(0)
    for sub, fname in (
        ("report", "digest.md"),
        ("plan", "implementation_plan.md"),
        ("ir", "workflow.yaml"),
        ("src", "marker.py"),
        ("tests", "marker.py"),
    ):
        archived = iter0 / sub / fname
        assert archived.exists(), f"missing archived file {archived}"
        assert archived.read_text() == "v1"


def test_archive_isolates_live_overwrites(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="iso")
    (plan_folder.report_dir() / "digest.md").write_text("v1")
    plan_folder.archive_artifacts_for_repair(0)
    (plan_folder.report_dir() / "digest.md").write_text("v2")
    archived = plan_folder.repairs_dir(0) / "report" / "digest.md"
    assert archived.read_text() == "v1"


def test_plan_manifest_repair_iterations_default_zero() -> None:
    manifest = PlanManifest(
        plan_id="x",
        created_at=datetime(2026, 5, 10, tzinfo=UTC),
        report_source="report.md",
        workflow_ir_path=Path("ir/workflow.yaml"),
    )
    assert manifest.repair_iterations == 0
    assert manifest.repair_history == ()


def test_plan_manifest_repair_history_round_trips_through_yaml(workspace: Workspace) -> None:
    from molexp.agent.modes.plan.schemas import RepairIterationRecord

    plan_folder = _mount(workspace, plan_id="hist")
    record = RepairIterationRecord(
        iteration=0,
        target_steps=("DraftImplementationPlan",),
        target_task_ids=("prepare",),
        cascade_downstream=True,
        archived_at=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
        feedback="rework equilibration",
    )
    manifest = PlanManifest(
        plan_id="hist",
        created_at=datetime(2026, 5, 10, tzinfo=UTC),
        report_source="r",
        workflow_ir_path=Path("ir/workflow.yaml"),
        repair_iterations=1,
        repair_history=(record,),
    )
    written = plan_folder.write_manifest(manifest)
    loaded = yaml.safe_load(written.read_text())
    assert loaded["repair_iterations"] == 1
    assert loaded["repair_history"][0]["target_steps"] == ["DraftImplementationPlan"]
    assert loaded["repair_history"][0]["target_task_ids"] == ["prepare"]
    assert loaded["repair_history"][0]["cascade_downstream"] is True


def test_repairs_dir_and_manifest_iteration(workspace: Workspace) -> None:
    """End-to-end: two repair rounds populate two distinct iter-<n>/ dirs."""
    from molexp.agent.modes.plan.schemas import RepairIterationRecord

    plan_folder = _mount(workspace, plan_id="acc")
    (plan_folder.report_dir() / "digest.md").write_text("round-0")
    plan_folder.archive_artifacts_for_repair(0)

    (plan_folder.report_dir() / "digest.md").write_text("round-1")
    plan_folder.archive_artifacts_for_repair(1)

    iter0_text = (plan_folder.repairs_dir(0) / "report" / "digest.md").read_text()
    iter1_text = (plan_folder.repairs_dir(1) / "report" / "digest.md").read_text()
    assert iter0_text == "round-0"
    assert iter1_text == "round-1"

    manifest = PlanManifest(
        plan_id="acc",
        created_at=datetime(2026, 5, 10, tzinfo=UTC),
        report_source="r",
        workflow_ir_path=Path("ir/workflow.yaml"),
        repair_iterations=2,
        repair_history=(
            RepairIterationRecord(
                iteration=0,
                target_steps=("DraftImplementationPlan",),
                archived_at=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
            ),
            RepairIterationRecord(
                iteration=1,
                target_task_ids=("prepare",),
                archived_at=datetime(2026, 5, 10, 12, 5, tzinfo=UTC),
            ),
        ),
    )
    written = plan_folder.write_manifest(manifest)
    loaded = yaml.safe_load(written.read_text())
    assert loaded["repair_iterations"] == 2
    assert len(loaded["repair_history"]) == 2
    assert loaded["repair_history"][0]["iteration"] == 0
    assert loaded["repair_history"][1]["iteration"] == 1


# ── Capability writers ─────────────────────────────────────────────────────


def test_capability_dir_created_lazily(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="cap-plan")
    capability = plan_folder.capability_dir()
    assert capability.name == "capability"
    assert capability.is_dir()
    assert capability.parent == plan_folder.root()


def test_write_capability_needs_round_trips_through_yaml(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="cap-plan")
    report = CapabilityNeedReport(
        discovery_required=True,
        needs=(
            CapabilityNeed(
                task_id="prepare",
                capability="construct a peptide",
                rationale="needs builder",
                expected_kind="class",
                query_hints=("peptide", "builder"),
            ),
        ),
        rationale_summary="prepare task needs a peptide builder",
    )
    path = plan_folder.write_capability_needs(report)
    assert path.name == "needs.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["discovery_required"] is True
    assert loaded["needs"][0]["task_id"] == "prepare"
    assert loaded["needs"][0]["query_hints"] == ["peptide", "builder"]


def test_write_capability_evidence_round_trips_through_yaml(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="cap-plan")
    batch = CapabilityEvidenceBatch(
        evidence=(
            CapabilityEvidence(
                need_fingerprint="prepare:construct a peptide",
                source="molmcp",
                package="molpy",
                module="molpy.builders.peptide",
                symbol="PeptideBuilder",
                kind="class",
                signature="class PeptideBuilder:",
                doc_summary="Build a peptide from amino-acid codes.",
                api_ref="molpy.builders.peptide.PeptideBuilder",
                confidence=0.95,
            ),
        ),
        missing=(),
        discovery_skipped=False,
    )
    path = plan_folder.write_capability_evidence(batch)
    assert path.name == "evidence.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["discovery_skipped"] is False
    assert loaded["evidence"][0]["api_ref"] == "molpy.builders.peptide.PeptideBuilder"


def test_write_capability_missing_renders_markdown_table(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="cap-plan")
    misses = (
        MissingCapability(
            need=CapabilityNeed(
                task_id="prepare",
                capability="construct a peptide",
            ),
            reason="mcp_no_match",
            detail="no matching symbol in molpy",
            repairable=False,
        ),
        MissingCapability(
            need=None,
            reason="unevidenced_in_code",
            detail="molpy.foo not in evidence batch",
            repairable=True,
        ),
    )
    path = plan_folder.write_capability_missing(misses)
    assert path.name == "missing.md"
    body = path.read_text()
    assert "# Missing capabilities" in body
    assert "mcp_no_match" in body
    assert "unevidenced_in_code" in body
    assert "prepare: construct a peptide" in body
    assert "_(no need)_" in body


def test_write_capability_missing_handles_empty_input(workspace: Workspace) -> None:
    plan_folder = _mount(workspace, plan_id="cap-plan")
    path = plan_folder.write_capability_missing(())
    body = path.read_text()
    assert "# Missing capabilities" in body
    assert "_(none)_" in body


# ── ac-003: PlanManifest.completed_nodes ──────────────────────────────────


def test_plan_manifest_completed_nodes_default() -> None:
    """ac-003: completed_nodes defaults to empty tuple."""
    manifest = PlanManifest(
        plan_id="test",
        created_at=datetime.now(tz=UTC),
        report_source="test",
        workflow_ir_path=Path("ir/workflow.yaml"),
    )
    assert manifest.completed_nodes == ()


def test_plan_manifest_completed_nodes_roundtrip() -> None:
    """ac-003: completed_nodes survives model_dump + reconstruct round-trip."""
    manifest = PlanManifest(
        plan_id="test",
        created_at=datetime.now(tz=UTC),
        report_source="test",
        workflow_ir_path=Path("ir/workflow.yaml"),
        completed_nodes=("IngestReport", "DraftReportDigest"),
    )
    data = manifest.model_dump(mode="json")
    reloaded = PlanManifest(**data)
    assert reloaded.completed_nodes == ("IngestReport", "DraftReportDigest")


# ── ac-004: PlanFolder.write_node_result ──────────────────────────────────


def test_write_node_result_creates_yaml(workspace: Workspace) -> None:
    """ac-004: write_node_result writes result as YAML in results/<name>.yaml."""
    from molexp.agent.modes.plan.schemas import IngestReportResult

    plan_folder = _mount(workspace, plan_id="resume-plan")
    result = IngestReportResult(report_path=Path("/tmp/report.md"), report_hash="abc123")
    path = plan_folder.write_node_result("IngestReport", result)

    assert path.name == "IngestReport.yaml"
    assert path.parent.name == "results"
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert loaded["report_hash"] == "abc123"


def test_write_node_result_overwrite(workspace: Workspace) -> None:
    """ac-004: repeated write_node_result overwrites previous file."""
    from molexp.agent.modes.plan.schemas import IngestReportResult

    plan_folder = _mount(workspace, plan_id="resume-plan")
    r1 = IngestReportResult(report_path=Path("/tmp/a.md"), report_hash="aaa")
    r2 = IngestReportResult(report_path=Path("/tmp/b.md"), report_hash="bbb")

    plan_folder.write_node_result("IngestReport", r1)
    plan_folder.write_node_result("IngestReport", r2)

    path = plan_folder.results_dir() / "IngestReport.yaml"
    loaded = yaml.safe_load(path.read_text())
    assert loaded["report_hash"] == "bbb"


# ── ac-005: PlanFolder.load_node_result ───────────────────────────────────


def test_load_node_result_deserializes_correct_type(workspace: Workspace) -> None:
    """ac-005: load_node_result returns the correct BaseModel subclass."""
    from molexp.agent.modes.plan.schemas import IngestReportResult

    plan_folder = _mount(workspace, plan_id="resume-plan")
    original = IngestReportResult(report_path=Path("/tmp/r.md"), report_hash="def456")
    plan_folder.write_node_result("IngestReport", original)

    loaded = plan_folder.load_node_result("IngestReport")
    assert isinstance(loaded, IngestReportResult)
    assert loaded.report_hash == "def456"


def test_load_node_result_raises_on_unknown_node(workspace: Workspace) -> None:
    """ac-005: load_node_result raises for unknown node names."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    with pytest.raises(KeyError, match="Unknown node"):
        plan_folder.load_node_result("NonExistentNode")


# ── ac-006: PlanFolder.checkpoint ─────────────────────────────────────────


def test_checkpoint_appends_to_completed_nodes(workspace: Workspace) -> None:
    """ac-006: checkpoint appends node name to manifest completed_nodes."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    plan_folder.checkpoint("IngestReport")
    plan_folder.checkpoint("DraftReportDigest")

    manifest = plan_folder.load_manifest()
    assert manifest.completed_nodes == ("IngestReport", "DraftReportDigest")

    # Reload from disk to verify persistence
    manifest2 = plan_folder.load_manifest()
    assert manifest2.completed_nodes == ("IngestReport", "DraftReportDigest")


def test_checkpoint_without_manifest_creates_stub(workspace: Workspace) -> None:
    """ac-006: checkpoint creates a minimal manifest stub when none exists."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    # No manifest written yet — checkpoint should create a stub
    plan_folder.checkpoint("IngestReport")
    manifest = plan_folder.load_manifest()
    assert "IngestReport" in manifest.completed_nodes


# ── ac-007: PlanFolder.reset_completed_nodes ──────────────────────────────


def test_reset_completed_nodes_clears(workspace: Workspace) -> None:
    """ac-007: reset_completed_nodes clears completed_nodes."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    plan_folder.checkpoint("IngestReport")
    assert len(plan_folder.load_manifest().completed_nodes) == 1

    plan_folder.reset_completed_nodes()
    assert plan_folder.load_manifest().completed_nodes == ()


# ── ac-008: PlanFolder.load_seed_outputs ───────────────────────────────────


def test_load_seed_outputs_returns_completed_results_map(workspace: Workspace) -> None:
    """ac-008: load_seed_outputs returns {node_name: result} for all completed nodes."""
    from molexp.agent.modes.plan.schemas import IngestReportResult

    plan_folder = _mount(workspace, plan_id="resume-plan")
    r1 = IngestReportResult(report_path=Path("/tmp/r.md"), report_hash="h1")
    plan_folder.write_node_result("IngestReport", r1)
    plan_folder.checkpoint("IngestReport")

    seed_outputs = plan_folder.load_seed_outputs()
    assert "IngestReport" in seed_outputs
    assert isinstance(seed_outputs["IngestReport"], IngestReportResult)
    assert seed_outputs["IngestReport"].report_hash == "h1"


def test_load_seed_outputs_empty_with_no_completed_nodes(workspace: Workspace) -> None:
    """ac-008: load_seed_outputs returns empty dict when nothing completed."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    assert plan_folder.load_seed_outputs() == {}


# ── ac-009: PlanFolder.load_manifest ──────────────────────────────────────


def test_load_manifest_raises_on_missing(workspace: Workspace) -> None:
    """ac-009: load_manifest raises FileNotFoundError when no manifest exists."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    with pytest.raises(FileNotFoundError, match="No manifest found"):
        plan_folder.load_manifest()


def test_load_manifest_returns_plan_manifest(workspace: Workspace) -> None:
    """ac-009: load_manifest returns a PlanManifest instance when present."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    manifest = PlanManifest(
        plan_id="resume-plan",
        created_at=datetime.now(tz=UTC),
        report_source="test",
        workflow_ir_path=Path("ir/wf.yaml"),
        completed_nodes=("IngestReport",),
    )
    plan_folder.write_manifest(manifest)

    loaded = plan_folder.load_manifest()
    assert isinstance(loaded, PlanManifest)
    assert loaded.plan_id == "resume-plan"
    assert loaded.completed_nodes == ("IngestReport",)


# ── results_dir ───────────────────────────────────────────────────────────


def test_results_dir_is_created_lazily(workspace: Workspace) -> None:
    """results_dir() returns the results/ subdirectory, creating it on demand."""
    plan_folder = _mount(workspace, plan_id="resume-plan")
    results = plan_folder.results_dir()
    assert results.name == "results"
    assert results.exists()
    assert results.is_dir()
