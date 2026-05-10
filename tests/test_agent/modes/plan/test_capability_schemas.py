"""Unit tests for the Phase 3 capability schemas + validator.

Covers acceptance criterion PYDA-08:

- The 5 frozen pydantic models (``CapabilityNeed`` /
  ``CapabilityNeedReport`` / ``CapabilityEvidence`` /
  ``CapabilityEvidenceBatch`` / ``MissingCapability``) reject mutation
  and unknown fields.
- ``MOLCRAFTS_NAMESPACES`` constant is exported and contains the
  expected namespace tuple.
- ``validate_codegen_evidence`` honors the four-way diff described in
  the spec's *Schemas* section: ``unevidenced_in_code``,
  ``undeclared_in_code``, ``declared_but_unused``, plus the
  ``discovery_skipped`` short-circuit.

The validator is exercised against representative AST shapes — bare
``from X import Y`` imports, dotted ``X.Y.Z(...)`` attribute chains,
non-Molcrafts namespaces (which must be ignored), and the
``__capability_evidence__: tuple[str, ...] = (...)`` declared-block
literal extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from molexp.agent.modes.plan.capability import (
    MOLCRAFTS_NAMESPACES,
    CapabilityEvidence,
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
    MissingCapability,
    validate_codegen_evidence,
)

# ── Frozen + extra-forbid contracts ────────────────────────────────────────


def _need() -> CapabilityNeed:
    return CapabilityNeed(
        task_id="prepare",
        capability="construct peptide",
        rationale="needs a builder for amino-acid chains",
        expected_kind="class",
        query_hints=("peptide", "builder"),
    )


def _evidence(api_ref: str = "molpy.builders.peptide.PeptideBuilder") -> CapabilityEvidence:
    module, _, symbol = api_ref.rpartition(".")
    return CapabilityEvidence(
        need_fingerprint="prepare:construct peptide",
        source="molmcp",
        package="molpy",
        module=module,
        symbol=symbol,
        kind="class",
        signature=f"class {symbol}:",
        doc_summary="Build a peptide from amino-acid codes.",
        api_ref=api_ref,
        confidence=0.95,
    )


@pytest.mark.parametrize(
    "make, field, new_value",
    [
        (_need, "task_id", "different"),
        (
            lambda: CapabilityNeedReport(
                discovery_required=True,
                needs=(_need(),),
                rationale_summary="prepare task needs a peptide builder",
            ),
            "discovery_required",
            False,
        ),
        (_evidence, "confidence", 0.5),
        (
            lambda: CapabilityEvidenceBatch(
                evidence=(_evidence(),),
                missing=(),
                discovery_skipped=False,
            ),
            "discovery_skipped",
            True,
        ),
        (
            lambda: MissingCapability(
                need=_need(),
                reason="mcp_no_match",
                detail="no matching symbol found in molpy",
                repairable=True,
            ),
            "reason",
            "mcp_timeout",
        ),
    ],
    ids=[
        "CapabilityNeed",
        "CapabilityNeedReport",
        "CapabilityEvidence",
        "CapabilityEvidenceBatch",
        "MissingCapability",
    ],
)
def test_models_are_frozen(make: Callable[[], BaseModel], field: str, new_value: object) -> None:
    instance = make()
    with pytest.raises(ValidationError):
        setattr(instance, field, new_value)


@pytest.mark.parametrize(
    "model_cls, base_kwargs",
    [
        (
            CapabilityNeed,
            {
                "task_id": "prepare",
                "capability": "x",
                "rationale": "y",
                "expected_kind": "callable",
                "query_hints": (),
            },
        ),
        (
            CapabilityNeedReport,
            {
                "discovery_required": False,
                "needs": (),
                "rationale_summary": "no needs",
            },
        ),
        (
            CapabilityEvidence,
            {
                "need_fingerprint": "x",
                "source": "molmcp",
                "package": "molpy",
                "module": "molpy",
                "symbol": "foo",
                "kind": "callable",
                "signature": "def foo(): ...",
                "doc_summary": "Foo.",
                "api_ref": "molpy.foo",
                "confidence": 0.9,
            },
        ),
        (
            CapabilityEvidenceBatch,
            {"evidence": (), "missing": (), "discovery_skipped": False},
        ),
        (
            MissingCapability,
            {
                "need": None,
                "reason": "unevidenced_in_code",
                "detail": "x",
                "repairable": True,
            },
        ),
    ],
)
def test_models_forbid_extra_fields(
    model_cls: type[BaseModel], base_kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        model_cls(**base_kwargs, surprise_extra="boom")


def test_missing_capability_reason_must_be_known() -> None:
    """MissingCapability.reason is constrained to the six documented values."""
    with pytest.raises(ValidationError):
        MissingCapability(
            need=None,
            reason="not_a_real_reason",  # type: ignore[arg-type]
            detail="",
            repairable=True,
        )


def test_molcrafts_namespaces_constant() -> None:
    assert isinstance(MOLCRAFTS_NAMESPACES, tuple)
    assert all(isinstance(ns, str) for ns in MOLCRAFTS_NAMESPACES)
    # The spec pins these eight namespaces; tests guard against accidental
    # drift since validate_codegen_evidence scopes scans by this prefix.
    assert set(MOLCRAFTS_NAMESPACES) == {
        "molpy",
        "molexp",
        "molvis",
        "molpack",
        "molnex",
        "molq",
        "mollog",
        "molcfg",
    }


# ── validate_codegen_evidence — happy paths ───────────────────────────────


def test_discovery_skipped_short_circuits() -> None:
    """``discovery_skipped=True`` makes the validator return ``()`` regardless."""
    source = (
        "__capability_evidence__: tuple[str, ...] = ()\n"
        "from molpy.builders.peptide import PeptideBuilder\n"
    )
    batch = CapabilityEvidenceBatch(evidence=(), missing=(), discovery_skipped=True)
    assert validate_codegen_evidence(source, batch) == ()


def test_all_refs_evidenced_returns_empty() -> None:
    source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molpy.builders.peptide.PeptideBuilder",\n'
        ")\n"
        "from molpy.builders.peptide import PeptideBuilder\n"
    )
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molpy.builders.peptide.PeptideBuilder"),),
        missing=(),
        discovery_skipped=False,
    )
    assert validate_codegen_evidence(source, batch) == ()


def test_attribute_chain_counts_as_ref() -> None:
    """``import molpy`` + ``molpy.builders.peptide.PeptideBuilder`` reads as the full path."""
    source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molpy.builders.peptide.PeptideBuilder",\n'
        ")\n"
        "import molpy\n"
        "PeptideBuilder = molpy.builders.peptide.PeptideBuilder\n"
    )
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molpy.builders.peptide.PeptideBuilder"),),
        missing=(),
        discovery_skipped=False,
    )
    assert validate_codegen_evidence(source, batch) == ()


def test_non_molcrafts_refs_are_ignored() -> None:
    source = (
        "__capability_evidence__: tuple[str, ...] = ()\n"
        "import os\n"
        "from collections import OrderedDict\n"
        "_ = os.path.join('a', 'b')\n"
    )
    batch = CapabilityEvidenceBatch(evidence=(), missing=(), discovery_skipped=False)
    assert validate_codegen_evidence(source, batch) == ()


# ── validate_codegen_evidence — diff branches ─────────────────────────────


def test_ast_ref_not_in_evidence_flagged_unevidenced() -> None:
    """Code uses ``molpy.foo`` that evidence doesn't carry → ``unevidenced_in_code``.

    The same ref is missing from ``__capability_evidence__`` so it
    additionally surfaces as ``undeclared_in_code`` (declared vs AST
    diff). The validator MUST report both branches; the test asserts
    presence of both.
    """
    source = "__capability_evidence__: tuple[str, ...] = ()\nfrom molpy import foo\n"
    batch = CapabilityEvidenceBatch(evidence=(), missing=(), discovery_skipped=False)
    misses = validate_codegen_evidence(source, batch)

    reasons = {m.reason for m in misses}
    assert "unevidenced_in_code" in reasons
    assert "undeclared_in_code" in reasons


def test_declared_ref_not_in_evidence_flagged_unevidenced() -> None:
    """Block lists a ref evidence doesn't carry → ``unevidenced_in_code``.

    Block declares ``molpy.foo`` while the AST never touches it, so the
    declared-vs-AST diff additionally flags ``declared_but_unused``.
    """
    source = '__capability_evidence__: tuple[str, ...] = (\n    "molpy.foo",\n)\n'
    batch = CapabilityEvidenceBatch(evidence=(), missing=(), discovery_skipped=False)
    misses = validate_codegen_evidence(source, batch)

    reasons = {m.reason for m in misses}
    assert "unevidenced_in_code" in reasons
    assert "declared_but_unused" in reasons


def test_undeclared_in_code_reported() -> None:
    """AST uses ``molpy.foo`` but block omits it → ``undeclared_in_code``."""
    source = "__capability_evidence__: tuple[str, ...] = ()\nfrom molpy import foo\n"
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molpy.foo"),),
        missing=(),
        discovery_skipped=False,
    )
    misses = validate_codegen_evidence(source, batch)
    reasons = {m.reason for m in misses}
    assert "undeclared_in_code" in reasons
    assert "unevidenced_in_code" not in reasons


def test_declared_but_unused_reported() -> None:
    """Block declares ``molpy.foo`` but AST never references it."""
    source = '__capability_evidence__: tuple[str, ...] = (\n    "molpy.foo",\n)\n'
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molpy.foo"),),
        missing=(),
        discovery_skipped=False,
    )
    misses = validate_codegen_evidence(source, batch)
    reasons = {m.reason for m in misses}
    assert "declared_but_unused" in reasons
    assert "unevidenced_in_code" not in reasons


def test_missing_declared_block_treats_declared_as_empty() -> None:
    """No ``__capability_evidence__`` block → declared_refs is the empty set.

    With evidence covering the AST ref the only remaining diff is
    declared-vs-AST: ``ast - declared = {molpy.foo}`` → ``undeclared_in_code``.
    """
    source = "from molpy import foo\n"
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molpy.foo"),),
        missing=(),
        discovery_skipped=False,
    )
    misses = validate_codegen_evidence(source, batch)
    reasons = {m.reason for m in misses}
    assert reasons == {"undeclared_in_code"}


def test_validator_detail_carries_ref() -> None:
    """The ``detail`` of each emitted MissingCapability must name the offending ref."""
    source = "__capability_evidence__: tuple[str, ...] = ()\nfrom molpy import foo\n"
    batch = CapabilityEvidenceBatch(evidence=(), missing=(), discovery_skipped=False)
    misses = validate_codegen_evidence(source, batch)
    assert all("molpy.foo" in m.detail for m in misses)
