"""``CapabilityRegistry`` Protocol — the typed contract for capability catalogs.

Every backend (in-memory test impl, future per-package adapters loading from
setuptools entrypoints, MCP-tools-list-backed registries, …) implements
this Protocol. The harness validator and Phase-5+ stages program against
the Protocol, never against a concrete class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.harness.schemas import ToolCapability

__all__ = ["CapabilityRegistry"]


@runtime_checkable
class CapabilityRegistry(Protocol):
    """Structural type for any capability catalog."""

    def list_capabilities(self) -> list[ToolCapability]:
        """Insertion-ordered snapshot of every registered capability."""
        ...

    def get(self, capability_id: str) -> ToolCapability:
        """Return the capability with the given id.

        Raises:
            CapabilityNotFoundError: if no capability with that id exists.
        """
        ...

    def has(self, capability_id: str) -> bool:
        """``True`` iff a capability with ``capability_id`` is registered."""
        ...

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
    ) -> list[ToolCapability]:
        """Return capabilities whose ``id`` / ``name`` / ``description``
        case-insensitively contain ``query``. When ``tags`` is provided,
        every requested tag must additionally appear in the capability's
        ``tags`` list (conjunctive AND across requested tags). When ``tags``
        is empty / None, no tag filter is applied.
        """
        ...

    def validate_call(
        self,
        capability_id: str,
        parameters: dict[str, object],
    ) -> None:
        """Shallow key-level validation against the capability's input_schema.

        Checks that:
        - ``capability_id`` exists in the registry.
        - ``parameters.keys() ⊆ input_schema["properties"].keys()``.
        - ``input_schema["required"] ⊆ parameters.keys()``.

        Does NOT validate value types against the schema (Phase 5+).

        Raises:
            CapabilityCallValidationError: on any mismatch.
        """
        ...
