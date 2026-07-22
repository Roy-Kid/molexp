"""Bindable capability gate — keep notes/tests/UI out of the plan catalog.

Mirrors :mod:`molexp.harness.registry.bindable`: the single gate deciding which
discovery hits may become plan-binding capabilities.
"""

from __future__ import annotations

from molexp.harness.registry.bindable import (
    is_bindable_capability,
    is_bindable_capability_id,
    is_bindable_kind,
)
from molexp.harness.schemas import ToolCapability


def _cap(cid: str, *, kind: str = "function") -> ToolCapability:
    return ToolCapability(
        id=cid,
        package=cid.split(".", 1)[0] if "." in cid else cid,
        name=cid.rsplit(".", 1)[-1],
        description="",
        input_schema={"type": "object"},
        output_schema={},
        callable_path=cid,
        supported_backends=["local"],
        tags=[kind] if kind else [],
    )


class TestBindable:
    def test_science_qualname_and_capability_are_bindable(self) -> None:
        assert is_bindable_capability_id("molpy.io.writers.write_lammps_data")
        assert is_bindable_capability(_cap("molpy.io.writers.write_lammps_data"))

    def test_notes_tests_ui_and_section_anchors_are_rejected(self) -> None:
        junk = (
            ".claude/notes/harness-goal::molcrafts::example-27",
            "workspace.foo.Bar",
            "tests.test_foo.test_bar",
            "ui.mocks.handlers.workspace.buildNestedTree",
            "mollog.get_logger",
            "molmcp.collection.index.CollectionIndex.explore",
            "molexp.harness.stages.resolve_capabilities",
            "molpy.core.cg.CoarseGrain._private",
        )
        for cid in junk:
            assert not is_bindable_capability_id(cid), cid

    def test_only_function_class_method_kinds_are_bindable(self) -> None:
        assert is_bindable_kind("function")
        assert is_bindable_kind("class")
        assert not is_bindable_kind("example")
        assert not is_bindable_kind("section")
        assert not is_bindable_kind("module")
        assert not is_bindable_kind("package")
        assert not is_bindable_kind("test")

    def test_only_molexp_curation_and_lifecycle_builtins_are_bindable(self) -> None:
        assert is_bindable_capability_id("molexp.curation.reorganize")
        assert is_bindable_capability_id("molexp.lifecycle.run_cancel")
        assert not is_bindable_capability_id("molexp.server.routes.plan")
