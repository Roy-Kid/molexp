"""Tests for workflow versioning — pure-data shape.

Spec: workflow-rectification (criterion `workflow-version-pure-data`).

After the rectification, `WorkflowVersion` is a pure data type with no
filesystem persistence helpers. `WorkflowSpec.version()` returns the
record; `WorkflowSpec.register(workspace)` and on-disk write/load
helpers (`write_record` / `load_record` / `_versions_dir` /
`_record_path`) are gone.
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


class TestWorkflowSpecVersionMethod:
    def test_version_returns_workflow_version_record(self):
        spec = _make_two_task_workflow(version="1.0.0").compile()
        record = spec.version

        assert isinstance(record, WorkflowVersion)
        assert record.workflow_id == spec.workflow_id
        assert record.version == "1.0.0"
        assert record.name == "pipeline"

    def test_version_topology_shape(self):
        spec = _make_two_task_workflow(version="1.0.0").compile()
        record = spec.version

        assert len(record.topology) == 2
        assert all(isinstance(t, TaskTopologyEntry) for t in record.topology)
        assert record.topology[0].name == "fetch"
        assert record.topology[1].name == "transform"
        assert record.topology[1].depends_on == ("fetch",)

    def test_version_label_is_separate_attribute(self):
        spec = _make_two_task_workflow(version="3.1.4").compile()
        # The version *label* (string) and the version *record* (WorkflowVersion)
        # are two different things; the record carries the label.
        assert spec.version_label == "3.1.4"
        assert spec.version.version == "3.1.4"
