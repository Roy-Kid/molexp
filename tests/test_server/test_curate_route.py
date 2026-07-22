"""The deterministic proposal builder ``build_curation_proposal`` (curate-unify-03).

Mirrors the ``build_curation_proposal`` builder (its typed-kwarg dispatch +
arg validation — the per-op proposal *shape* is locked in
``test_curate_proposal_flow``).
"""

from __future__ import annotations

import pytest

from molexp.harness.schemas.change_proposal import ObjectRef
from molexp.services.curate_runtime import build_curation_proposal


class TestBuildCurationProposal:
    def test_rehome_asset_builds_asset_move_with_typed_scope_refs(self) -> None:
        """The build-only ``rehome_asset`` op dispatches on typed scope dicts."""
        p = build_curation_proposal(
            "rehome_asset",
            asset="a1",
            source={"kind": "experiment", "id": "e1"},
            target={"kind": "experiment", "id": "e2"},
            action="copy",
        )
        assert p.proposed_change.op == "asset_move"
        assert p.proposed_change.payload["curation_op"] == "rehome_asset"
        assert p.proposed_change.payload["action"] == "copy"
        assert ObjectRef(kind="data_asset", id="a1") in p.affected_objects
        assert ObjectRef(kind="experiment", id="e1") in p.affected_objects
        assert ObjectRef(kind="experiment", id="e2") in p.affected_objects

    def test_missing_required_arg_raises(self) -> None:
        """A required arg missing for the chosen op raises ValueError."""
        with pytest.raises(ValueError):
            build_curation_proposal("move_run", run="r1")
