"""Bindable capability gate — keep notes/tests/UI out of the plan catalog."""

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


def test_science_function_is_bindable() -> None:
    assert is_bindable_capability_id("molpy.io.writers.write_lammps_data")
    assert is_bindable_capability(_cap("molpy.io.writers.write_lammps_data"))


def test_notes_and_section_anchors_rejected() -> None:
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


def test_kinds_gate() -> None:
    assert is_bindable_kind("function")
    assert is_bindable_kind("class")
    assert not is_bindable_kind("example")
    assert not is_bindable_kind("section")
    assert not is_bindable_kind("module")
    assert not is_bindable_kind("package")
    assert not is_bindable_kind("test")


def test_molexp_curation_built_ins_allowed() -> None:
    assert is_bindable_capability_id("molexp.curation.reorganize")
    assert is_bindable_capability_id("molexp.lifecycle.run_cancel")
    assert not is_bindable_capability_id("molexp.server.routes.plan")
