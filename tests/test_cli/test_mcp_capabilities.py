"""Unit tests for the molmcp → harness capability mapping (pure functions).

The async ``fetch_molmcp_capabilities`` path needs a live molmcp stdio server and
is exercised separately; here we pin the deterministic, transport-free core:
signature parsing, ``ToolCapability`` mapping, dedup, and that the resulting
``InMemoryCapabilityRegistry`` validates calls without false rejections.

The payloads embedded below are trimmed copies of real
``molmcp_find_capability`` responses (pkg:molpy), so the mapping is tested
against the actual on-the-wire shape.
"""

from __future__ import annotations

import pytest

from molexp.cli.mcp_capabilities import (
    capabilities_from_payloads,
    capability_from_node,
    parse_signature_params,
    synthesize_input_schema,
)
from molexp.harness import InMemoryCapabilityRegistry
from molexp.harness.errors import CapabilityCallValidationError


class TestParseSignatureParams:
    def test_function_with_default_marks_only_non_default_required(self) -> None:
        names, required, accepts_extra = parse_signature_params(
            "write_lammps_data(file: PathLike, frame: Any, atom_style: str='full') -> None"
        )
        assert names == ["file", "frame", "atom_style"]
        assert required == ["file", "frame"]  # atom_style has a default
        assert accepts_extra is False

    def test_method_drops_self_receiver(self) -> None:
        names, required, accepts_extra = parse_signature_params("write(self, frame: Frame) -> None")
        assert names == ["frame"]
        assert required == ["frame"]
        assert accepts_extra is False

    def test_commas_inside_brackets_do_not_split_and_varkw_flags_extra(self) -> None:
        names, required, accepts_extra = parse_signature_params(
            "f(a: dict[str, int], b: int = 3, *args, **kw) -> None"
        )
        assert names == ["a", "b"]  # *args dropped, **kw not a named param
        assert required == ["a"]
        assert accepts_extra is True  # **kw → arbitrary keyword keys allowed

    def test_null_or_empty_signature_is_unparseable(self) -> None:
        assert parse_signature_params(None) is None
        assert parse_signature_params("") is None
        assert parse_signature_params("ClassNameOnly") is None


class TestSynthesizeInputSchema:
    def test_parsed_signature_yields_properties_and_required(self) -> None:
        schema = synthesize_input_schema("f(a: int, b: int = 1) -> None")
        assert schema == {
            "type": "object",
            "properties": {"a": {}, "b": {}},
            "required": ["a"],
        }

    def test_null_signature_is_wildcard_without_properties(self) -> None:
        # No ``properties`` key is the validator's "any input allowed" — class
        # constructor nodes (signature == null) must not false-reject.
        assert synthesize_input_schema(None) == {"type": "object"}

    def test_varkw_signature_is_wildcard_but_keeps_required(self) -> None:
        # ``**kwargs`` → accept arbitrary extra keys (no ``properties``) while
        # still enforcing the named required parameter.
        schema = synthesize_input_schema("emit(self, frame: Frame, *, prefix: str, **opts) -> None")
        assert "properties" not in schema
        assert schema["required"] == ["frame", "prefix"]


# Real ``molmcp_find_capability`` matches (trimmed), pkg:molpy.
_CG_PAYLOAD: dict[str, object] = {
    "query": "build a coarse-grained polymer chain from charged beads and bonds",
    "matches": [
        {
            "rank": 1,
            "node": {
                "qualname": "molpy.core.cg.CGBond",
                "name": "CGBond",
                "kind": "class",
                "file": "molpy/core/cg.py",
                "signature": None,
                "summary": "Coarse-grained bond between two beads.",
            },
        },
        {
            "rank": 2,
            "node": {
                "qualname": "molpy.core.cg.CoarseGrain",
                "name": "CoarseGrain",
                "kind": "class",
                "file": "molpy/core/cg.py",
                "signature": None,
                "summary": "Coarse-grained molecular structure backed by molrs.",
            },
        },
    ],
    "snapshot": {"spec": "pkg:molpy", "commit": "7fa1b1a", "freshness": "fresh"},
}

_IO_PAYLOAD: dict[str, object] = {
    "query": "write a LAMMPS data file",
    "matches": [
        {
            "rank": 1,
            "node": {
                "qualname": "molpy.io.writers.write_lammps_data",
                "name": "write_lammps_data",
                "kind": "function",
                "file": "molpy/io/writers.py",
                "signature": "write_lammps_data(file: PathLike, frame: Any, atom_style: str='full') -> None",
                "summary": "Write a Frame object to a LAMMPS data file.",
            },
        },
        # A duplicate of CGBond appears again across queries — must dedup by id.
        {
            "rank": 2,
            "node": {
                "qualname": "molpy.core.cg.CGBond",
                "name": "CGBond",
                "kind": "class",
                "signature": None,
                "summary": "Coarse-grained bond between two beads.",
            },
        },
    ],
    "snapshot": {"spec": "pkg:molpy", "commit": "7fa1b1a"},
}


class TestCapabilityFromNode:
    def test_function_node_maps_all_fields(self) -> None:
        node = _IO_PAYLOAD["matches"][0]["node"]  # type: ignore[index]
        cap = capability_from_node(node, snapshot_commit="7fa1b1a")
        assert cap is not None
        assert cap.id == "molpy.io.writers.write_lammps_data"
        assert cap.package == "molpy"
        assert cap.callable_path == "molpy.io.writers.write_lammps_data"
        assert cap.description.startswith("Write a Frame")
        assert cap.supported_backends == ["local"]
        assert cap.tags == ["function"]
        assert cap.version == "7fa1b1a"
        assert cap.input_schema["properties"] == {"file": {}, "frame": {}, "atom_style": {}}
        assert cap.input_schema["required"] == ["file", "frame"]

    def test_class_node_gets_wildcard_schema(self) -> None:
        node = _CG_PAYLOAD["matches"][0]["node"]  # type: ignore[index]
        cap = capability_from_node(node)
        assert cap is not None
        assert cap.id == "molpy.core.cg.CGBond"
        assert "properties" not in cap.input_schema  # wildcard

    def test_node_without_qualname_is_skipped(self) -> None:
        assert capability_from_node({"name": "x", "kind": "class"}) is None


class TestCapabilitiesFromPayloads:
    def test_union_is_deduped_by_id(self) -> None:
        caps = capabilities_from_payloads([_CG_PAYLOAD, _IO_PAYLOAD])
        ids = [c.id for c in caps]
        assert ids.count("molpy.core.cg.CGBond") == 1  # deduped across payloads
        assert set(ids) == {
            "molpy.core.cg.CGBond",
            "molpy.core.cg.CoarseGrain",
            "molpy.io.writers.write_lammps_data",
        }

    def test_registry_validates_real_and_class_calls(self) -> None:
        caps = capabilities_from_payloads([_CG_PAYLOAD, _IO_PAYLOAD])
        registry = InMemoryCapabilityRegistry(caps)

        # Known good call against the parsed function schema.
        registry.validate_call(
            "molpy.io.writers.write_lammps_data",
            {"file": "out.data", "frame": object()},
        )
        # Class-constructor capability (wildcard schema) accepts any kwargs.
        registry.validate_call("molpy.core.cg.CGBond", {"a": 1, "b": 2, "type": "fene"})

        # Missing a required key is rejected.
        with pytest.raises(CapabilityCallValidationError):
            registry.validate_call("molpy.io.writers.write_lammps_data", {"file": "out.data"})
        # An unexpected key against a restricted schema is rejected.
        with pytest.raises(CapabilityCallValidationError):
            registry.validate_call(
                "molpy.io.writers.write_lammps_data",
                {"file": "o", "frame": object(), "bogus": 1},
            )

    def test_unknown_capability_is_absent_from_registry(self) -> None:
        registry = InMemoryCapabilityRegistry(capabilities_from_payloads([_CG_PAYLOAD]))
        assert not registry.has("molpy.does.not.Exist")
