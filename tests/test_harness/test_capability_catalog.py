"""Unit tests for the binder capability-catalog renderer."""

from __future__ import annotations

from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.schemas import ToolCapability


def _cap(
    cid: str,
    *,
    kind: str = "function",
    properties: list[str] | None = None,
    required: list[str] | None = None,
    desc: str = "",
) -> ToolCapability:
    schema: dict[str, object] = {"type": "object"}
    if properties is not None:
        schema["properties"] = {p: {} for p in properties}
        schema["required"] = required or []
    return ToolCapability(
        id=cid,
        package=cid.split(".")[0],
        name=cid.rsplit(".", 1)[-1],
        description=desc,
        input_schema=schema,
        output_schema={},
        callable_path=cid,
        supported_backends=["local"],
        tags=[kind],
    )


def test_renders_sorted_bindable_with_required_markers() -> None:
    caps = [
        _cap(
            "molpy.io.writers.write_lammps_data",
            properties=["file", "frame", "atom_style"],
            required=["file", "frame"],
            desc="Write a Frame object to a LAMMPS data file.",
        ),
        _cap("molpy.core.cg.CoarseGrain", kind="class", desc="CG structure."),  # wildcard schema
    ]
    out = render_capability_catalog(caps)

    assert "## Available molcrafts capabilities" in out
    # Deterministic order: sorted by id, so CoarseGrain precedes write_lammps_data.
    assert out.index("molpy.core.cg.CoarseGrain") < out.index("molpy.io.writers.write_lammps_data")
    # Required params carry a trailing `*`; defaulted ones do not.
    assert "- molpy.io.writers.write_lammps_data(file*, frame*, atom_style)" in out
    assert "Write a Frame object to a LAMMPS data file." in out
    # A wildcard schema (no properties) renders as (…).
    assert "- molpy.core.cg.CoarseGrain(…) — CG structure." in out


def test_drops_private_and_module_entries() -> None:
    caps = [
        _cap(
            "molpy.core.cg.CoarseGrain._resolve_bead_atoms",
            kind="method",
            properties=["bead_handle"],
            required=["bead_handle"],
        ),
        _cap("molpy.core.cg", kind="module"),
        _cap("molpy.core.cg.Bead", kind="class"),
    ]
    out = render_capability_catalog(caps)

    assert "- molpy.core.cg.Bead(…)" in out
    assert "_resolve_bead_atoms" not in out  # private leaf dropped
    assert "- molpy.core.cg(…)" not in out  # module-kind entry dropped


def test_long_description_is_truncated() -> None:
    out = render_capability_catalog([_cap("p.x.Y", kind="class", desc="d" * 300)])
    assert "…" in out
    # The single rendered bullet line stays bounded.
    bullet = next(line for line in out.splitlines() if line.startswith("- p.x.Y"))
    assert len(bullet) < 200
