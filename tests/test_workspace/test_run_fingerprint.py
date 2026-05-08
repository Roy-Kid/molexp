"""Tests for `RunFingerprint` and `Run.fingerprint`.

Trace: ac-012.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from molexp.workspace.run import RunFingerprint

# ── RunFingerprint model — deterministic content hash ───────────────────────


def test_fingerprint_is_frozen_pydantic_model() -> None:
    assert issubclass(RunFingerprint, BaseModel)
    assert RunFingerprint.model_config["frozen"] is True
    assert RunFingerprint.model_config.get("extra") == "forbid"


def test_fingerprint_id_is_deterministic_and_16_hex() -> None:
    fp = RunFingerprint(
        workflow_spec_id="wf-aaa",
        parameters_hash="p-aaa",
        inputs_hash="i-aaa",
        environment_hash="e-aaa",
    )
    assert isinstance(fp.fingerprint_id, str)
    assert len(fp.fingerprint_id) == 16
    assert all(c in "0123456789abcdef" for c in fp.fingerprint_id)
    fp2 = RunFingerprint(
        workflow_spec_id="wf-aaa",
        parameters_hash="p-aaa",
        inputs_hash="i-aaa",
        environment_hash="e-aaa",
    )
    assert fp.fingerprint_id == fp2.fingerprint_id


@pytest.mark.parametrize(
    "field,changed",
    [
        ("workflow_spec_id", "wf-DIFFERENT"),
        ("parameters_hash", "p-DIFFERENT"),
        ("inputs_hash", "i-DIFFERENT"),
        ("environment_hash", "e-DIFFERENT"),
    ],
)
def test_fingerprint_changes_when_any_input_changes(field: str, changed: str) -> None:
    base = {
        "workflow_spec_id": "wf",
        "parameters_hash": "p",
        "inputs_hash": "i",
        "environment_hash": "e",
    }
    fp_base = RunFingerprint(**base)
    fp_mut = RunFingerprint(**{**base, field: changed})
    assert fp_base.fingerprint_id != fp_mut.fingerprint_id


# ── Run.fingerprint property — exposes fingerprint alongside Run.id ─────────


def test_run_id_is_unaffected_by_fingerprint(run) -> None:
    """ac-012: existing run.id (UUID) is unaffected."""
    # The UUID id and the content-addressed fingerprint are independent.
    assert isinstance(run.id, str)
    fp = run.fingerprint
    assert isinstance(fp, RunFingerprint)
    # Two reads of the same run yield the same fingerprint.
    assert run.fingerprint.fingerprint_id == fp.fingerprint_id


def test_two_runs_with_same_inputs_have_same_fingerprint(experiment) -> None:
    run_a = experiment.run(parameters={"lr": 1e-4})
    run_b = experiment.run(parameters={"lr": 1e-4})
    # Same parameters + same workflow_snapshot + same environment ⇒ same fingerprint.
    assert run_a.fingerprint.fingerprint_id == run_b.fingerprint.fingerprint_id
    # But the UUIDs are independent.
    assert run_a.id != run_b.id


def test_runs_with_different_parameters_have_different_fingerprints(
    experiment,
) -> None:
    run_a = experiment.run(parameters={"lr": 1e-4})
    run_b = experiment.run(parameters={"lr": 9e-3})
    assert run_a.fingerprint.fingerprint_id != run_b.fingerprint.fingerprint_id
