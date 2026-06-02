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
    WorkflowVersionConflictError,
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


class TestWorkflowVersionConflictErrorIsRuntimeError:
    def test_conflict_error_is_runtime_error_subclass(self):
        assert issubclass(WorkflowVersionConflictError, RuntimeError)


class TestNoFilesystemHelpers:
    def test_no_register_method_on_spec(self):
        spec = _make_two_task_workflow().compile()
        assert not hasattr(spec, "register"), (
            "WorkflowSpec.register(workspace) must be removed; persistence is "
            "no longer the workflow layer's responsibility."
        )

    def test_no_to_workflow_version_method(self):
        spec = _make_two_task_workflow().compile()
        assert not hasattr(spec, "to_workflow_version"), (
            "to_workflow_version() must be renamed to version()."
        )

    def test_no_persistence_helpers_in_version_module(self):
        from molexp.workflow import version as version_mod

        for name in ("write_record", "load_record", "_versions_dir", "_record_path"):
            assert not hasattr(version_mod, name), (
                f"Persistence helper {name!r} must be removed from molexp.workflow.version."
            )
