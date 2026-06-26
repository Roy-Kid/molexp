"""Tests for ExecutionReport — the descriptive "where & how" hand-off (step 9).

Locks the wire format: a resolved compute target (machine/account/scheduler)
plus the bound workflow's resource policy + environment. Descriptive only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _report_kwargs() -> dict:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment, ResourcePolicy

    return {
        "id": "er-1",
        "bound_workflow_id": "bw-1",
        "target_name": "hpc1",
        "scheduler": "slurm",
        "host": "me@cluster.example.org",
        "scratch_root": "/scratch/me/molexp",
        "account": "proj-1234",
        "queue": "normal",
        "partition": "gpu",
        "total_runs": 6,
        "resource_policy": ResourcePolicy(backend="slurm", max_runtime_s=3600),
        "environment": ExecutionEnvironment(python_version="3.12", platform="linux"),
    }


def test_execution_report_full_round_trip() -> None:
    from molexp.harness.schemas.execution_report import ExecutionReport

    report = ExecutionReport(**_report_kwargs())
    assert ExecutionReport.model_validate_json(report.model_dump_json()) == report
    assert report.account == "proj-1234"
    assert report.scheduler == "slurm"


def test_execution_report_local_defaults() -> None:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment, ResourcePolicy
    from molexp.harness.schemas.execution_report import ExecutionReport

    report = ExecutionReport(
        id="er-1",
        bound_workflow_id="bw-1",
        target_name="laptop",
        resource_policy=ResourcePolicy(backend="local", max_runtime_s=60),
        environment=ExecutionEnvironment(),
    )
    assert report.scheduler == "local"
    assert report.host is None
    assert report.account is None
    assert report.total_runs == 1
    assert report.notes == []


def test_execution_report_is_frozen() -> None:
    from molexp.harness.schemas.execution_report import ExecutionReport

    report = ExecutionReport(**_report_kwargs())
    with pytest.raises(ValidationError):
        report.account = "other"  # type: ignore[misc]


def test_execution_report_rejects_unknown_scheduler() -> None:
    from molexp.harness.schemas.bound_workflow import ExecutionEnvironment, ResourcePolicy
    from molexp.harness.schemas.execution_report import ExecutionReport

    with pytest.raises(ValidationError):
        ExecutionReport(
            id="er-1",
            bound_workflow_id="bw-1",
            target_name="x",
            scheduler="condor",  # type: ignore[arg-type]
            resource_policy=ResourcePolicy(backend="local", max_runtime_s=60),
            environment=ExecutionEnvironment(),
        )
