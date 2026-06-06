"""Asset lineage traversal.

Each :class:`~molexp.workspace.assets.base.Asset` carries an optional
:class:`~molexp.workspace.assets.base.Producer` whose
:attr:`Producer.inputs` lists the upstream ``asset_id``s consumed to
build it. Together they form a directed acyclic graph spanning the
entire workspace; this module exposes two BFS walkers over it.

Example::

    from molexp.workspace.assets import lineage

    upstream = lineage.ancestors(workspace, leaf_asset.asset_id)
    downstream = lineage.descendants(workspace, raw_input.asset_id)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace import Workspace


def ancestors(workspace: Workspace, asset_id: str) -> set[str]:
    """Return every ``asset_id`` reachable upstream of *asset_id*.

    Walks :attr:`Producer.inputs` edges in reverse breadth-first order.
    The starting ``asset_id`` itself is **not** included; defensive
    self-loops (``producer.inputs`` contains the asset's own id) are
    silently skipped.

    Args:
        workspace: Workspace whose catalog hosts the asset graph.
        asset_id: Leaf to walk back from.

    Returns:
        Set of upstream ``asset_id``s. Empty when the leaf has no
        producer or no inputs.
    """
    visited: set[str] = set()
    frontier: list[str] = [asset_id]
    while frontier:
        cur = frontier.pop()
        asset = workspace.catalog.get(cur)
        if asset is None or asset.producer is None:
            continue
        for upstream in asset.producer.inputs:
            if upstream == asset_id or upstream in visited:
                continue
            visited.add(upstream)
            frontier.append(upstream)
    return visited


def descendants(workspace: Workspace, asset_id: str) -> set[str]:
    """Return every ``asset_id`` reachable downstream of *asset_id*.

    Inverts the :attr:`Producer.inputs` index across the workspace
    catalog, then walks forward breadth-first. The starting
    ``asset_id`` is excluded from the result; self-loops terminate.

    Args:
        workspace: Workspace whose catalog hosts the asset graph.
        asset_id: Source to walk forward from.

    Returns:
        Set of downstream ``asset_id``s. Empty when no asset records
        *asset_id* in its inputs.
    """
    children_of: dict[str, list[str]] = {}
    for asset in workspace.catalog.query_assets():
        if asset.producer is None:
            continue
        for inp in asset.producer.inputs:
            children_of.setdefault(inp, []).append(asset.asset_id)

    visited: set[str] = set()
    frontier: list[str] = [asset_id]
    while frontier:
        cur = frontier.pop()
        for child in children_of.get(cur, ()):
            if child == asset_id or child in visited:
                continue
            visited.add(child)
            frontier.append(child)
    return visited
