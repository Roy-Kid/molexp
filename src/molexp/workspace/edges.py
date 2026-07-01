"""Typed edge roles for the OKF markdown-link knowledge graph.

The Open Knowledge Format graph lives as markdown links in each Folder's
``index.md`` (see :mod:`molexp.workspace.folder`). This module gives every edge a
typed *role* — a declared relation (``derived_from`` / ``cites`` / ``supersedes``
/ ``records`` / ``references``) — encoded in the link's ``[label]`` channel, so
the writer (:func:`molexp.workspace.folder.append_link`) and the reader
(:meth:`molexp.workspace.folder.Folder.links`) share one format and can never
drift.

Design notes:

- The vocabulary mirrors the *shape* of ``workflow/ir.py``'s ``EdgeKind`` (a
  plain lowercase-snake :data:`typing.Literal`, never an ``Enum``) **without
  importing it** — the layer DAG forbids ``workspace → workflow``.
- ``EdgeRole`` is a relation *label*, **not** a concept *type*, so it is
  deliberately absent from ``knowledge/types.py``.
- :data:`DEFAULT_EDGE_ROLE` (``"references"``) is the role assigned to any
  legacy/untyped on-disk link — such a link is **defaulted, never dropped**
  (the project "no silent fallback" rule).
- The default role encodes to the bare label — byte-identical to the pre-role
  output — so existing ``index.md`` files stay valid with zero migration. A
  non-default role prefixes the ``@<role> `` sigil. A label that itself begins
  with a literal ``@<known-role> `` is therefore reinterpreted on read; concept
  names (validated slugs) never do this, and it is the only collision the
  collision-resistant sigil admits.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

EdgeRole = Literal["derived_from", "cites", "supersedes", "records", "references"]
"""A declared relation on an OKF knowledge-graph edge."""

DEFAULT_EDGE_ROLE: EdgeRole = "references"
"""Role assigned to any legacy/untyped on-disk link — defaulted, never dropped."""

# Kept in lockstep with the ``EdgeRole`` Literal above (a Literal cannot be built
# from a variable, so the two spellings sit adjacent for easy synchronization).
_EDGE_ROLES: frozenset[str] = frozenset(
    {"derived_from", "cites", "supersedes", "records", "references"}
)

_ROLE_SIGIL = "@"


class Edge(NamedTuple):
    """A resolved typed out-edge: an in-tree *target* path plus its *role*.

    Attributes:
        target: The absolute in-tree path the edge resolves to (same value the
            path-only :meth:`Folder.out_edges` reports).
        role: The declared :data:`EdgeRole` of the edge.
    """

    target: str
    role: EdgeRole


def validate_role(role: str) -> EdgeRole:
    """Return *role* if it is a known :data:`EdgeRole`, else raise ``ValueError``.

    Fails loudly on any token outside the vocabulary — in the spirit of
    ``folder._validate_kind`` — so a corrupt role is never written to disk.

    Args:
        role: The candidate role token.

    Returns:
        The validated role (narrowed to :data:`EdgeRole`).

    Raises:
        ValueError: If *role* is not one of the known roles.
    """
    if role not in _EDGE_ROLES:
        raise ValueError(f"invalid edge role {role!r}: must be one of {sorted(_EDGE_ROLES)}")
    return cast("EdgeRole", role)


def encode_label(role: EdgeRole, label: str) -> str:
    """Encode *role* + human *label* into the markdown link's label channel.

    The default role encodes to the bare *label* (byte-identical to the pre-role
    output → on-disk back-compat); a non-default role prefixes the ``@<role> ``
    sigil.

    Args:
        role: The edge role to encode.
        label: The human-readable link label.

    Returns:
        The text to place in the markdown ``[...]`` label channel.

    Raises:
        ValueError: If *role* is not a known role.
    """
    validate_role(role)
    if role == DEFAULT_EDGE_ROLE:
        return label
    return f"{_ROLE_SIGIL}{role} {label}" if label else f"{_ROLE_SIGIL}{role}"


def parse_role(label: str) -> tuple[EdgeRole, str]:
    """Recover ``(role, human_label)`` from a markdown link's label channel.

    An absent or unrecognized sigil yields ``(DEFAULT_EDGE_ROLE, label)`` — the
    legacy link is never dropped, only defaulted.

    Args:
        label: The raw text from a markdown link's ``[...]`` label channel.

    Returns:
        The recovered ``(role, human_label)`` pair.
    """
    if label.startswith(_ROLE_SIGIL):
        token, _, rest = label[len(_ROLE_SIGIL) :].partition(" ")
        if token in _EDGE_ROLES:
            return (cast("EdgeRole", token), rest)
    return (DEFAULT_EDGE_ROLE, label)


__all__ = [
    "DEFAULT_EDGE_ROLE",
    "Edge",
    "EdgeRole",
    "encode_label",
    "parse_role",
    "validate_role",
]
