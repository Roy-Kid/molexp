"""Unit tests for the binder capability-catalog renderers.

Mirrors :mod:`molexp.harness.prompts.capability_catalog`
(``render_capability_catalog`` / ``render_selected_capability_catalog``). What
counts as bindable is owned by ``test_bindable.py`` — here we only assert the
renderers *apply* that gate and shape the catalog text.
"""

from __future__ import annotations

from molexp.harness.prompts.capability_catalog import (
    render_capability_catalog,
    render_selected_capability_catalog,
)
from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry
from molexp.harness.schemas import CapabilitySelection, ToolCapability
from molexp.harness.schemas.capability_selection import SelectedCapability


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


class TestRenderCapabilityCatalog:
    def test_renders_sorted_with_required_markers_and_wildcard(self) -> None:
        caps = [
            _cap(
                "molpy.io.writers.write_lammps_data",
                properties=["file", "frame", "atom_style"],
                required=["file", "frame"],
                desc="Write a Frame object to a LAMMPS data file.",
            ),
            _cap("molpy.core.cg.CoarseGrain", kind="class", desc="CG structure."),
        ]
        out = render_capability_catalog(caps)

        assert "## Available molcrafts capabilities" in out
        # Deterministic order: sorted by id, so CoarseGrain precedes write_lammps_data.
        assert out.index("molpy.core.cg.CoarseGrain") < out.index(
            "molpy.io.writers.write_lammps_data"
        )
        # Required params carry a trailing `*`; defaulted ones do not.
        assert "- molpy.io.writers.write_lammps_data(file*, frame*, atom_style)" in out
        assert "Write a Frame object to a LAMMPS data file." in out
        # A wildcard schema (no properties) renders as (…).
        assert "- molpy.core.cg.CoarseGrain(…) — CG structure." in out

    def test_applies_bindable_gate_dropping_noise(self) -> None:
        caps = [
            _cap(
                ".claude/notes/harness-goal::example-27",
                kind="example",
                desc="molpy.build_polymer(...)",
            ),
            _cap("mollog.get_logger", kind="function", desc="logger"),
            _cap("molpy.core.cg.CoarseGrain", kind="class", desc="CG structure."),
        ]
        out = render_capability_catalog(caps)
        bullets = [ln for ln in out.splitlines() if ln.startswith("- ")]
        assert bullets == ["- molpy.core.cg.CoarseGrain(…) — CG structure."]
        assert not any("mollog" in ln for ln in bullets)
        assert not any(".claude" in ln for ln in bullets)
        assert not any("build_polymer" in ln for ln in bullets)

    def test_long_description_is_truncated(self) -> None:
        out = render_capability_catalog([_cap("molpy.x.Y", kind="class", desc="d" * 300)])
        assert "…" in out
        bullet = next(line for line in out.splitlines() if line.startswith("- molpy.x.Y"))
        assert len(bullet) < 200


class TestRenderSelectedCapabilityCatalog:
    def test_renders_only_selected_with_reasons(self) -> None:
        registry = InMemoryCapabilityRegistry(
            [
                _cap("molpy.core.cg.CoarseGrain", kind="class", desc="CG structure."),
                _cap("molpy.io.writers.write_gromacs", desc="Write a GROMACS topology."),
            ]
        )
        selection = CapabilitySelection(
            selected=[
                SelectedCapability(id="molpy.core.cg.CoarseGrain", reason="builds the CG beads")
            ]
        )
        out = render_selected_capability_catalog(registry, selection)

        assert "## Selected molcrafts capabilities" in out
        assert "- molpy.core.cg.CoarseGrain(…) — CG structure." in out
        assert "↳ needed: builds the CG beads" in out
        assert "write_gromacs" not in out  # un-selected capability is omitted

    def test_drops_unknown_selected_ids_with_note(self) -> None:
        registry = InMemoryCapabilityRegistry(
            [_cap("molpy.core.cg.CoarseGrain", kind="class", desc="CG structure.")]
        )
        selection = CapabilitySelection(
            selected=[
                SelectedCapability(id="molpy.core.cg.CoarseGrain", reason="real"),
                SelectedCapability(id="made.up.Capability", reason="hallucinated"),
            ]
        )
        out = render_selected_capability_catalog(registry, selection)

        assert "- molpy.core.cg.CoarseGrain(…) — CG structure." in out
        assert "Dropped (not in the grounded toolchain): made.up.Capability" in out
