"""Entrypoint resolvers in ``molexp.workflow.promote``.

``resolve_callable_entrypoint`` / ``resolve_spec_entrypoint`` return
``"<file>:<qualname|varname>"`` refs so a cluster worker can re-import a
promoted body or a module-bound spec. Both reject shapes that cannot be
re-imported (lambdas / locals; specs not bound at module scope).

The promotion + tracked-execution behavior of ``promote_callable`` /
``_EntryTask`` is owned by ``test_promote_tracked_execution``.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    CompiledWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    promote_callable,
    resolve_callable_entrypoint,
    resolve_spec_entrypoint,
)

# Module-level callables for the resolver tests.


async def _async_fn(ctx) -> None:
    return None


def _sync_fn(ctx) -> None:
    return None


# Module-level user task — gives ``resolve_spec_entrypoint`` a first task
# whose module is *this* test file (the resolver walks the first task's module
# to find the spec's module-level binding).
class _ProbeTask(Task):
    async def execute(self, ctx: TaskContext) -> None:
        return None


# Module-level spec for the spec-entrypoint resolver test.
_FIXTURE_SPEC: CompiledWorkflow = (
    WorkflowCompiler(name="probe").add(_ProbeTask(), name="step").compile()
)


class TestResolveCallableEntrypoint:
    def test_returns_file_and_qualname_for_module_level_function(self) -> None:
        ep = resolve_callable_entrypoint(_async_fn)
        assert ":" in ep, "entrypoint must be <file>:<qualname>"
        file_part, _, qual_part = ep.partition(":")
        assert file_part.endswith("test_promote_callable.py"), (
            f"expected this file as the source path, got {file_part}"
        )
        assert qual_part == "_async_fn"

    def test_rejects_lambda(self) -> None:
        with pytest.raises(ValueError, match="entrypoint"):
            resolve_callable_entrypoint(lambda ctx: None)  # noqa: ARG005


class TestResolveSpecEntrypoint:
    def test_finds_module_level_binding(self) -> None:
        ep = resolve_spec_entrypoint(_FIXTURE_SPEC)
        file_part, _, var_part = ep.partition(":")
        assert file_part.endswith("test_promote_callable.py")
        assert var_part == "_FIXTURE_SPEC"

    def test_rejects_unbound_spec(self) -> None:
        # Held in a local, not assigned to any module-level variable — the
        # resolver must refuse rather than guess.
        spec = promote_callable(_sync_fn, "ephemeral")
        with pytest.raises(ValueError, match="bound to a module-level variable"):
            resolve_spec_entrypoint(spec)
