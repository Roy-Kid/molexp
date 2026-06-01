"""``Executor`` Protocol — the contract every executor backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.harness.schemas import CommandResult, CommandSpec
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["Executor"]


@runtime_checkable
class Executor(Protocol):
    """Structural type for any executor backend."""

    async def execute(
        self,
        spec: CommandSpec,
        *,
        artifact_store: ArtifactStore,
    ) -> CommandResult: ...
