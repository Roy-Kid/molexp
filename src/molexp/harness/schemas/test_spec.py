"""``TestSpec`` + ``TestResult`` — first-class structured test artifacts.

Per ``.claude/notes/harness-goal.md`` §4.8: tests are not natural-language
appendices but typed, validatable artifacts. Phase 5 ships only the
**shape** + structural validators; actual test execution and TestResult
production land in later phases.

A ``TestSpec`` carries enough metadata for a future runner to:
- decide whether to run (``kind``, ``required``)
- locate its target (``target_task_id`` or ``target_workflow_id``)
- supply inputs (``inputs: dict[str, ParameterValue]``)
- optionally invoke a CLI (``command: list[str] | None``)
- assert what should exist after (``expected_artifacts``)
- assert numerical bounds (``expected_metrics`` + ``tolerance``)

A ``TestResult`` records the outcome with full provenance (status,
metrics, produced artifacts, stdout/stderr refs, optional human-readable
reason).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactRef
from molexp.harness.schemas.parameter import ParameterValue

__all__ = ["TestKind", "TestResult", "TestSpec", "TestSpecBundle", "TestStatus"]


TestKind = Literal[
    "schema_test",
    "unit_test",
    "dry_run_test",
    "integration_test",
    "regression_test",
    "numerical_tolerance_test",
    "artifact_existence_test",
    "provenance_test",
    "resource_policy_test",
]


TestStatus = Literal["passed", "failed", "skipped", "error"]


class TestSpec(BaseModel):
    """Structural description of one harness test."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    kind: TestKind
    target_task_id: str | None = None
    target_workflow_id: str | None = None
    description: str
    inputs: dict[str, ParameterValue] = Field(default_factory=dict)
    command: list[str] | None = None
    expected_artifacts: list[str] = Field(default_factory=list)
    expected_metrics: dict[str, ParameterValue] = Field(default_factory=dict)
    tolerance: dict[str, float] = Field(default_factory=dict)
    required: bool = True


class TestSpecBundle(BaseModel):
    """One ``test_spec`` artifact carrying a per-task list of :class:`TestSpec`.

    The ``GenerateTestSpec`` stage fans test generation out to **one
    :class:`TestSpec` per ``BoundTask``** of the bound workflow; rather than
    persist N separate ``test_spec`` artifacts (which would break the
    single-latest-artifact contract both ``ValidateTestSpec`` and
    ``GenerateTestCode`` rely on) the specs ride inside this bundle. The
    artifact *kind* stays ``"test_spec"``; only its JSON shape widens from a
    bare ``TestSpec`` to this wrapper. Each member's ``target_task_id`` names
    the ``BoundTask`` it covers.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    bound_workflow_id: str
    specs: list[TestSpec] = Field(default_factory=list)

    @classmethod
    def from_artifact(cls, raw: bytes | str) -> TestSpecBundle:
        """Parse a ``test_spec`` artifact body as a bundle.

        Accepts the current bundle shape; falls back to a bare
        :class:`TestSpec` (wrapped as a one-element bundle) so single-spec
        artifacts written before the per-task fan-out still load. Raises if
        the bytes are neither a bundle nor a TestSpec.
        """
        try:
            return cls.model_validate_json(raw)
        except ValueError:
            spec = TestSpec.model_validate_json(raw)
            return cls(id=spec.id, bound_workflow_id=spec.target_workflow_id or "", specs=[spec])


class TestResult(BaseModel):
    """Outcome of running one :class:`TestSpec`."""

    model_config = ConfigDict(frozen=True)

    id: str
    test_spec_id: str
    status: TestStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    produced_artifacts: list[ArtifactRef] = Field(default_factory=list)
    stdout: ArtifactRef | None = None
    stderr: ArtifactRef | None = None
    reason: str | None = None
