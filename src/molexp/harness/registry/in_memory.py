"""In-memory implementation of :class:`CapabilityRegistry`.

Backed by a single ``dict`` keyed on ``capability.id`` plus a parallel list
that preserves insertion order. Suitable for tests and for callers who want
to assemble a registry by hand from a list of :class:`ToolCapability`
instances; production per-package adapters typically build a registry from
their own catalog and hand it to the harness via ``register``.
"""

from __future__ import annotations

from molexp.harness.errors import (
    CapabilityAlreadyRegisteredError,
    CapabilityCallValidationError,
    CapabilityNotFoundError,
)
from molexp.harness.schemas import ToolCapability

__all__ = ["InMemoryCapabilityRegistry"]


class InMemoryCapabilityRegistry:
    """In-memory capability catalog."""

    def __init__(self, capabilities: list[ToolCapability] | None = None) -> None:
        self._by_id: dict[str, ToolCapability] = {}
        self._order: list[str] = []
        for cap in capabilities or []:
            self.register(cap)

    # ---------------------------------------------------------- mutation

    def register(self, capability: ToolCapability) -> None:
        if capability.id in self._by_id:
            raise CapabilityAlreadyRegisteredError(
                f"capability id {capability.id!r} is already registered"
            )
        self._by_id[capability.id] = capability
        self._order.append(capability.id)

    # ---------------------------------------------------------- queries

    def list_capabilities(self) -> list[ToolCapability]:
        return [self._by_id[cid] for cid in self._order]

    def get(self, capability_id: str) -> ToolCapability:
        try:
            return self._by_id[capability_id]
        except KeyError as exc:
            raise CapabilityNotFoundError(
                f"capability id {capability_id!r} not registered"
            ) from exc

    def has(self, capability_id: str) -> bool:
        return capability_id in self._by_id

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
    ) -> list[ToolCapability]:
        needle = query.lower()
        results: list[ToolCapability] = []
        for cid in self._order:
            cap = self._by_id[cid]
            haystack = f"{cap.id} {cap.name} {cap.description}".lower()
            if needle not in haystack:
                continue
            if tags and not set(tags).issubset(set(cap.tags)):
                continue
            results.append(cap)
        return results

    def validate_call(
        self,
        capability_id: str,
        parameters: dict[str, object],
    ) -> None:
        if capability_id not in self._by_id:
            raise CapabilityCallValidationError(
                f"validate_call: capability id {capability_id!r} not registered"
            )
        cap = self._by_id[capability_id]
        schema = cap.input_schema

        # A schema with no ``"properties"`` key is treated as "any input
        # allowed" — this is the JSON-Schema default ("no restriction"
        # is materially different from "no allowed keys"). The previous
        # implementation rejected every parameter against an
        # unrestricted schema because ``properties`` defaulted to an
        # empty set, turning a wildcard contract into a closed one.
        provided = set(parameters.keys())
        if "properties" in schema:
            properties = set(schema["properties"].keys())
            extra = provided - properties
            if extra:
                raise CapabilityCallValidationError(
                    f"validate_call({capability_id!r}): unexpected parameter keys: {sorted(extra)}"
                )
        required = set(schema.get("required", []))
        missing = required - provided
        if missing:
            raise CapabilityCallValidationError(
                f"validate_call({capability_id!r}): missing required parameter keys: "
                f"{sorted(missing)}"
            )
