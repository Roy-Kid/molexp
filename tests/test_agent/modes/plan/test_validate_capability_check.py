"""Phase 6 tests for ``ValidateWorkspace.capability_evidence_check``.

Covers acceptance criterion ``PYDA-19``: the dual-signal validator
runs *both*:

* ``declared_refs`` — extracted from each generated module's
  ``__capability_evidence__`` literal;
* ``ast_refs`` — extracted via :func:`ast.walk` filtered by
  :data:`MOLCRAFTS_NAMESPACES`;

and diffs each against the on-disk ``capability/evidence.yaml``'s
``api_ref`` set. A non-empty diff in *either* signal flips the
relevant :class:`CheckResult` to ``passed=False`` with severity
``error``. ``discovery_skipped=True`` short-circuits both signals to
``passed=True`` with detail ``"discovery_skipped"``. Missing
``evidence.yaml`` is treated as an info-level pass (the probe is null
or the pipeline aborted).

Three branches MUST be covered:

1. **declared-only-missing** — source declares a ref absent from
   evidence; AST diff is clean.
2. **ast-only-missing** — source's AST references a ref the declared
   block doesn't list and which isn't in evidence either.
3. **both-missing** — same ref absent from declared block, AST, and
   evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from molexp.agent.modes.plan.capability import (
    CapabilityEvidence,
    CapabilityEvidenceBatch,
)
from molexp.agent.modes.plan.tasks import _capability_evidence_checks
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.workspace import Workspace


@pytest.fixture
def ws_handle(tmp_path: Path) -> PlanFolder:
    return Workspace(tmp_path / "ws").add_folder(PlanFolder(name="vc_check"))


def _evidence(api_ref: str) -> CapabilityEvidence:
    module, _, symbol = api_ref.rpartition(".")
    return CapabilityEvidence(
        need_fingerprint=f"task:{api_ref}",
        source="molmcp",
        package=module.split(".", 1)[0] if module else "",
        module=module,
        symbol=symbol,
        kind="class",
        signature=f"class {symbol}:",
        doc_summary="",
        api_ref=api_ref,
        confidence=1.0,
    )


def _write_evidence(handle: PlanFolder, batch: CapabilityEvidenceBatch) -> None:
    handle.write_capability_evidence(batch)


def _write_task_module(handle: PlanFolder, name: str, source: str) -> None:
    path = handle.tasks_pkg_dir() / f"{name}.py"
    path.write_text(source, encoding="utf-8")


def _write_test_module(handle: PlanFolder, name: str, source: str) -> None:
    path = handle.tests_dir() / f"test_{name}.py"
    path.write_text(source, encoding="utf-8")


def _check_named(checks, name: str):
    matches = [c for c in checks if c.name == name]
    assert len(matches) == 1, (
        f"expected exactly one CheckResult named {name!r}, found {len(matches)}: {checks}"
    )
    return matches[0]


# ── Boundary cases ─────────────────────────────────────────────────────────


def test_no_evidence_yaml_yields_info_pass(ws_handle: PlanFolder) -> None:
    """When evidence.yaml is missing, the validator emits a single info pass."""
    checks = _capability_evidence_checks(ws_handle)
    assert len(checks) == 1
    assert checks[0].name == "capability_evidence_present"
    assert checks[0].passed is True
    assert checks[0].severity == "info"


def test_discovery_skipped_passes_both_signals(ws_handle: PlanFolder) -> None:
    """``discovery_skipped=True`` short-circuits both signals to passed=True."""
    _write_evidence(ws_handle, CapabilityEvidenceBatch(discovery_skipped=True))
    checks = _capability_evidence_checks(ws_handle)
    assert len(checks) == 2
    declared = _check_named(checks, "capability_evidence_check[declared]")
    ast_check = _check_named(checks, "capability_evidence_check[ast]")
    assert declared.passed is True
    assert declared.detail == "discovery_skipped"
    assert ast_check.passed is True
    assert ast_check.detail == "discovery_skipped"


def test_evidence_yaml_corrupt_yields_error_check(ws_handle: PlanFolder) -> None:
    """Unparseable evidence.yaml flags a single error check."""
    cap_dir = ws_handle.capability_dir()
    (cap_dir / "evidence.yaml").write_text("not: [a valid: yaml")  # malformed
    checks = _capability_evidence_checks(ws_handle)
    assert any(
        c.name == "capability_evidence_parseable" and not c.passed and c.severity == "error"
        for c in checks
    )


# ── PYDA-19 — three required diff branches ────────────────────────────────


def test_declared_only_missing(ws_handle: PlanFolder) -> None:
    """Declared block names a ref absent from evidence; AST is clean."""
    _write_evidence(
        ws_handle,
        CapabilityEvidenceBatch(
            evidence=(_evidence("molexp.workflow.Task"),),
            discovery_skipped=False,
        ),
    )
    # Source declares Task + Actor; uses only Task. Actor is in declared but
    # NOT in evidence → declared-signal miss. AST signal sees only Task,
    # which IS in evidence → AST signal clean.
    _write_task_module(
        ws_handle,
        "prepare",
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        '    "molexp.workflow.Actor",\n'
        ")\n"
        "from molexp.workflow import Task\n"
        "x = Task\n",
    )
    checks = _capability_evidence_checks(ws_handle)
    declared = _check_named(checks, "capability_evidence_check[declared]")
    ast_check = _check_named(checks, "capability_evidence_check[ast]")
    assert declared.passed is False
    assert "molexp.workflow.Actor" in declared.detail
    assert ast_check.passed is True


def test_ast_only_missing(ws_handle: PlanFolder) -> None:
    """AST uses a ref absent from evidence; declared block omits it too."""
    _write_evidence(
        ws_handle,
        CapabilityEvidenceBatch(
            evidence=(_evidence("molexp.workflow.Task"),),
            discovery_skipped=False,
        ),
    )
    # Source uses Task + Actor; declares only Task. Actor is in AST but NOT
    # in declared → undeclared-in-code (caught by Phase 5 gate, but here
    # also triggers an AST-signal miss because Actor is not in evidence).
    # Declared signal: declared={Task}, evidence={Task}, diff={} → passes.
    _write_task_module(
        ws_handle,
        "prepare",
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        ")\n"
        "from molexp.workflow import Task, Actor\n"
        "x = Task; y = Actor\n",
    )
    checks = _capability_evidence_checks(ws_handle)
    declared = _check_named(checks, "capability_evidence_check[declared]")
    ast_check = _check_named(checks, "capability_evidence_check[ast]")
    assert declared.passed is True
    assert ast_check.passed is False
    assert "molexp.workflow.Actor" in ast_check.detail


def test_both_missing(ws_handle: PlanFolder) -> None:
    """A ref absent from evidence AND declared by the block AND used in AST."""
    _write_evidence(
        ws_handle,
        CapabilityEvidenceBatch(
            evidence=(_evidence("molexp.workflow.Task"),),
            discovery_skipped=False,
        ),
    )
    _write_test_module(
        ws_handle,
        "prepare",
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        '    "molexp.workflow.Actor",\n'
        ")\n"
        "from molexp.workflow import Task, Actor\n"
        "def test_x():\n"
        "    assert Task is not None\n"
        "    assert Actor is not None\n",
    )
    checks = _capability_evidence_checks(ws_handle)
    declared = _check_named(checks, "capability_evidence_check[declared]")
    ast_check = _check_named(checks, "capability_evidence_check[ast]")
    assert declared.passed is False
    assert ast_check.passed is False
    # Both signals point at the same offending ref.
    assert "molexp.workflow.Actor" in declared.detail
    assert "molexp.workflow.Actor" in ast_check.detail


def test_clean_workspace_passes_both_signals(ws_handle: PlanFolder) -> None:
    """Source uses + declares only refs that are in evidence → both signals pass."""
    _write_evidence(
        ws_handle,
        CapabilityEvidenceBatch(
            evidence=(
                _evidence("molexp.workflow.Task"),
                _evidence("molexp.workflow.Actor"),
            ),
            discovery_skipped=False,
        ),
    )
    _write_task_module(
        ws_handle,
        "prepare",
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        ")\n"
        "from molexp.workflow import Task\n"
        "x = Task\n",
    )
    checks = _capability_evidence_checks(ws_handle)
    declared = _check_named(checks, "capability_evidence_check[declared]")
    ast_check = _check_named(checks, "capability_evidence_check[ast]")
    assert declared.passed is True
    assert ast_check.passed is True


def test_evidence_yaml_round_trips_through_pydantic(ws_handle: PlanFolder) -> None:
    """Sanity: write_capability_evidence + reload via the validator preserves shape."""
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molexp.workflow.Task"),),
        discovery_skipped=False,
    )
    _write_evidence(ws_handle, batch)
    loaded = yaml.safe_load((ws_handle.capability_dir() / "evidence.yaml").read_text())
    rebuilt = CapabilityEvidenceBatch.model_validate(loaded)
    assert rebuilt.evidence[0].api_ref == "molexp.workflow.Task"
