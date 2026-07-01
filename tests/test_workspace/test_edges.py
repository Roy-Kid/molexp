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
    Edge,
    encode_label,
    parse_role,
    validate_role,
)

ROLES = ("derived_from", "cites", "supersedes", "records", "references")


def test_default_role_is_references() -> None:
    assert DEFAULT_EDGE_ROLE == "references"


def test_validate_role_accepts_every_known_role() -> None:
    for role in ROLES:
        assert validate_role(role) == role


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError):
        validate_role("not_a_role")


def test_encode_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        encode_label("bogus", "x")


@pytest.mark.parametrize("role", ROLES)
def test_encode_parse_round_trip(role: str) -> None:
    encoded = encode_label(role, "my-target")
    assert parse_role(encoded) == (role, "my-target")


def test_default_role_encodes_to_bare_label() -> None:
    # byte-identical to the pre-role output → on-disk back-compat
    assert encode_label("references", "my-target") == "my-target"


def test_non_default_role_prefixes_a_sigil() -> None:
    assert encode_label("derived_from", "x").startswith("@derived_from")


def test_parse_untyped_label_defaults_to_references() -> None:
    assert parse_role("just a label") == (DEFAULT_EDGE_ROLE, "just a label")


def test_parse_unrecognized_sigil_is_defaulted_not_dropped() -> None:
    # an unknown sigil token → default role, original label preserved verbatim
    assert parse_role("@mention someone") == (DEFAULT_EDGE_ROLE, "@mention someone")


def test_edge_namedtuple_carries_target_and_role() -> None:
    edge = Edge(target="/a/b", role="cites")
    assert edge.target == "/a/b"
    assert edge.role == "cites"
