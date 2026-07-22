"""Tests for ``InMemoryCapabilityRegistry`` (harness capability registry).

Locks: the ``CapabilityRegistry`` runtime_checkable Protocol conformance;
register / get / has / list_capabilities / search / validate_call; the typed
errors (``CapabilityNotFoundError`` / ``CapabilityAlreadyRegisteredError`` /
``CapabilityCallValidationError``); and the shallow ``validate_call`` contract
(closed-schema key checking, no value-type checking, unrestricted-schema pass).
"""

from __future__ import annotations

import pytest


def _make_capability(
    *,
    id_: str = "molpy.builder.X",
    name: str = "X",
    description: str = "A builder",
    package: str = "molpy",
    supported_backends: list[str] | None = None,
    tags: list[str] | None = None,
    side_effects: list[str] | None = None,
    required: list[str] | None = None,
    properties: dict | None = None,
):
    from molexp.harness.schemas.capability import ToolCapability

    if properties is None:
        properties = {"n_chains": {"type": "integer"}}
    if required is None:
        required = ["n_chains"]
    return ToolCapability(
        id=id_,
        package=package,
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties, "required": required},
        output_schema={"type": "object", "properties": {"structure": {"type": "string"}}},
        supported_backends=supported_backends if supported_backends is not None else ["local"],
        tags=tags or [],
        side_effects=side_effects or [],
    )


class TestInMemoryCapabilityRegistry:
    # ---------------------------------------------------------------- Protocol
    def test_satisfies_runtime_checkable_capability_registry_protocol(self) -> None:
        from molexp.harness.registry.capability_registry import CapabilityRegistry
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        assert isinstance(InMemoryCapabilityRegistry(), CapabilityRegistry)

    # ------------------------------------------------------------- register/get
    def test_register_then_get_roundtrips(self) -> None:
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        cap = _make_capability()
        reg.register(cap)
        assert reg.get(cap.id) == cap

    def test_register_duplicate_raises_already_registered(self) -> None:
        from molexp.harness.errors import CapabilityAlreadyRegisteredError
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        cap = _make_capability()
        reg.register(cap)
        with pytest.raises(CapabilityAlreadyRegisteredError):
            reg.register(cap)

    def test_get_missing_raises_not_found(self) -> None:
        from molexp.harness.errors import CapabilityNotFoundError
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        with pytest.raises(CapabilityNotFoundError):
            reg.get("ghost")

    def test_constructor_seeds_capabilities_queryable_via_has(self) -> None:
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        a = _make_capability(id_="cap.a")
        b = _make_capability(id_="cap.b")
        reg = InMemoryCapabilityRegistry(capabilities=[a, b])
        assert reg.has("cap.a")
        assert reg.has("cap.b")

    # --------------------------------------------------------- list / search
    def test_list_capabilities_preserves_insertion_order(self) -> None:
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        ids = ["cap.a", "cap.b", "cap.c"]
        for i in ids:
            reg.register(_make_capability(id_=i))
        assert [c.id for c in reg.list_capabilities()] == ids

    def test_search_by_name_substring_is_case_insensitive(self) -> None:
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(_make_capability(id_="cap.a", name="GBigSmilesCompiler"))
        results = reg.search("gbigsmiles")
        assert [c.id for c in results] == ["cap.a"]

    def test_search_tag_filter_is_conjunctive(self) -> None:
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(_make_capability(id_="cap.a", tags=["builder", "polymer"]))
        reg.register(_make_capability(id_="cap.b", tags=["builder"]))
        reg.register(_make_capability(id_="cap.c", tags=["polymer"]))
        # query="" → matches all; tag filter requires BOTH tags.
        results = reg.search("", tags=["builder", "polymer"])
        assert [c.id for c in results] == ["cap.a"]

    def test_search_empty_tag_filter_is_a_no_op(self) -> None:
        """tags=[] must NOT filter out capabilities (no requirement to satisfy)."""
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(_make_capability(id_="cap.a", description="x"))
        results = reg.search("", tags=[])
        assert [c.id for c in results] == ["cap.a"]

    # --------------------------------------------------------- validate_call
    def test_validate_call_unknown_capability_raises(self) -> None:
        from molexp.harness.errors import CapabilityCallValidationError
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        with pytest.raises(CapabilityCallValidationError):
            reg.validate_call("ghost", {})

    def test_validate_call_missing_required_raises(self) -> None:
        from molexp.harness.errors import CapabilityCallValidationError
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(
            _make_capability(required=["n_chains"], properties={"n_chains": {"type": "integer"}})
        )
        with pytest.raises(CapabilityCallValidationError):
            reg.validate_call("molpy.builder.X", {})

    def test_validate_call_extra_key_raises(self) -> None:
        from molexp.harness.errors import CapabilityCallValidationError
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(_make_capability(properties={"n_chains": {"type": "integer"}}, required=[]))
        with pytest.raises(CapabilityCallValidationError):
            reg.validate_call("molpy.builder.X", {"n_chains": 100, "stray": 1})

    def test_validate_call_is_shallow_accepts_wrong_value_type(self) -> None:
        """Shallow contract: no value-type checking. String passed for int → OK."""
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

        reg = InMemoryCapabilityRegistry()
        reg.register(
            _make_capability(properties={"n_chains": {"type": "integer"}}, required=["n_chains"])
        )
        reg.validate_call("molpy.builder.X", {"n_chains": "not an int"})  # MUST NOT raise

    def test_validate_call_unrestricted_schema_accepts_any_keys(self) -> None:
        """A schema with NO ``properties`` key is unrestricted: any parameters accepted.

        Regression: the previous implementation treated absent ``properties`` as
        an empty closed set, rejecting every call to a capability whose author
        had left the schema permissive.
        """
        from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
        from molexp.harness.schemas.capability import ToolCapability

        reg = InMemoryCapabilityRegistry()
        cap = ToolCapability(
            id="any.cap",
            package="example",
            name="cap",
            description="unrestricted capability",
            input_schema={},  # NO properties key → unrestricted by JSON-Schema convention
            output_schema={},
        )
        reg.register(cap)
        reg.validate_call("any.cap", {"foo": 1, "bar": 2})  # MUST NOT raise
