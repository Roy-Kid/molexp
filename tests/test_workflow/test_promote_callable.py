"""``promote_callable`` covers the moved-from-workspace helper.

Phase 1 of the rectification spec moved ``_promote_to_workflow``,
``_resolve_callable_entrypoint``, and ``_resolve_spec_entrypoint``
out of ``workspace/experiment.py`` (where they forced a downward leak)
into ``workflow/promote.py`` and re-exported them publicly as
``promote_callable`` / ``resolve_callable_entrypoint`` /
``resolve_spec_entrypoint``.

Tests here exercise:

- promotion turns a bare ``fn(RunContext)`` into a runnable
  ``WorkflowSpec`` whose only task wraps that callable;
- the entrypoint resolver returns ``"<file>:<qualname>"`` for module-
  level callables and rejects unimportable shapes (lambdas, locals);
- the spec entrypoint resolver locates a ``WorkflowSpec`` bound at
  module scope and rejects unbound specs.
"""

from __future__ import annotations

from molexp.workflow import (
    Task,
    TaskContext,
    Workflow,
    WorkflowSpec,
    promote_callable,
    resolve_callable_entrypoint,
    resolve_spec_entrypoint,
)


# Module-level callables for the resolver tests.


async def _async_fn(ctx) -> None:  # noqa: ANN001 - duck-typed RunContext-like
    return None


def _sync_fn(ctx) -> None:  # noqa: ANN001 - duck-typed RunContext-like
    return None


# Module-level user task — gives ``resolve_spec_entrypoint`` a first
# task whose module is *this* test file (the resolver walks the first
# task's module to find the spec's module-level binding). Promoted
# callables route through the in-package ``_EntryTask``, which lives
# in ``promote.py`` and would mislead the resolver — that path is
# meant for ``resolve_callable_entrypoint`` instead.
class _ProbeTask(Task):
    async def execute(self, ctx: TaskContext) -> None:
        return None


# Module-level spec for the spec-entrypoint resolver test.
_FIXTURE_SPEC: WorkflowSpec = Workflow(name="probe").add(_ProbeTask(), name="step").build()


def test_promote_async_callable_yields_runnable_spec() -> None:
    spec = promote_callable(_async_fn, "p1")
    assert isinstance(spec, WorkflowSpec)
    assert spec.name == "p1"


def test_promote_sync_callable_yields_runnable_spec() -> None:
    spec = promote_callable(_sync_fn, "p2")
    assert isinstance(spec, WorkflowSpec)
    assert spec.name == "p2"


def test_resolve_callable_entrypoint_for_module_level_function() -> None:
    ep = resolve_callable_entrypoint(_async_fn)
    assert ":" in ep, "entrypoint must be <file>:<qualname>"
    file_part, _, qual_part = ep.partition(":")
    assert file_part.endswith("test_promote_callable.py"), (
        f"expected this file as the source path, got {file_part}"
    )
    assert qual_part == "_async_fn"


def test_resolve_callable_entrypoint_rejects_lambda() -> None:
    import pytest

    with pytest.raises(ValueError, match="entrypoint"):
        resolve_callable_entrypoint(lambda ctx: None)


def test_resolve_callable_entrypoint_rejects_local_function() -> None:
    def _local_fn(ctx) -> None:  # noqa: ANN001
        return None

    import pytest

    with pytest.raises(ValueError, match="entrypoint"):
        resolve_callable_entrypoint(_local_fn)


def test_resolve_spec_entrypoint_finds_module_level_binding() -> None:
    ep = resolve_spec_entrypoint(_FIXTURE_SPEC)
    file_part, _, var_part = ep.partition(":")
    assert file_part.endswith("test_promote_callable.py")
    assert var_part == "_FIXTURE_SPEC"


def test_resolve_spec_entrypoint_rejects_unbound_spec() -> None:
    spec = promote_callable(_sync_fn, "ephemeral")
    # Not assigned to any module-level variable in this test — just
    # held in a local. Resolver should refuse.
    import pytest

    with pytest.raises(ValueError, match="bound to a module-level variable"):
        resolve_spec_entrypoint(spec)
