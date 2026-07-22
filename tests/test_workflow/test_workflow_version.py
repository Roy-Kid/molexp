"""Tests for workflow versioning — pure-data shape.

Spec: workflow-rectification (criterion `workflow-version-pure-data`).

After the rectification, `WorkflowVersion` is a pure data type with no
filesystem persistence helpers; `CompiledWorkflow.version` returns the record
derived from the DAG, and `version_label` (a plain string) is a separate
attribute from that record.
"""

from __future__ import annotations

from molexp.workflow import WorkflowCompiler
from molexp.workflow.version import (
    TaskTopologyEntry,
    WorkflowVersion,
)


def _make_two_task_workflow(version: str = "1.0.0") -> WorkflowCompiler:
    wf = WorkflowCompiler(name="pipeline", version=version)

    @wf.task
    async def fetch(ctx):
        return 1

    @wf.task(depends_on=["fetch"])
    async def transform(ctx):
        return 2

    return wf


class TestWorkflowVersion:
    def test_compile_derives_version_record(self):
        spec = _make_two_task_workflow(version="1.0.0").compile()
        record = spec.version

        assert isinstance(record, WorkflowVersion)
        assert record.workflow_id == spec.workflow_id
        assert record.version == "1.0.0"
        assert record.name == "pipeline"
        # The version *label* (string) and the version *record* (WorkflowVersion)
        # are two distinct attributes; both carry the same label value.
        assert spec.version_label == "1.0.0"

    def test_version_records_task_topology(self):
        record = _make_two_task_workflow(version="1.0.0").compile().version

        assert len(record.topology) == 2
        assert all(isinstance(t, TaskTopologyEntry) for t in record.topology)
        assert record.topology[0].name == "fetch"
        assert record.topology[1].name == "transform"
        assert record.topology[1].depends_on == ("fetch",)
