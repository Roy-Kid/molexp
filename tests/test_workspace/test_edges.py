"""EdgeRole vocabulary + label-channel encode/parse/validate (typed-provenance-edge P0.1).

RED-first: ``molexp.workspace.edges`` does not exist yet, so this module fails at
collection until the slice is implemented.

References:
- spec:       ``.claude/specs/typed-provenance-edge.md``
- acceptance: ``.claude/specs/typed-provenance-edge.acceptance.md`` (ac-003 + primitive)
"""

from __future__ import annotations

import pytest

from molexp.workspace.edges import (
    DEFAULT_EDGE_ROLE,
    encode_label,
    parse_role,
)

ROLES = ("derived_from", "cites", "supersedes", "records", "references")


@pytest.mark.parametrize("role", ROLES)
def test_encode_parse_round_trip(role: str) -> None:
    encoded = encode_label(role, "my-target")
    assert parse_role(encoded) == (role, "my-target")


def test_default_role_encodes_to_bare_label() -> None:
    # byte-identical to the pre-role output → on-disk back-compat
    assert encode_label("references", "my-target") == "my-target"


def test_parse_unrecognized_sigil_is_defaulted_not_dropped() -> None:
    # an unknown sigil token → default role, original label preserved verbatim
    assert parse_role("@mention someone") == (DEFAULT_EDGE_ROLE, "@mention someone")
