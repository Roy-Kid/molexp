"""Consolidate workflow source across runs by content hash.

``dedupe_workflow_source`` groups run ids by the content hash of their
``run_dir/source`` snapshot; ``consolidate_workflow_source`` maps each duplicate
run id to the canonical (first-sorted) id of its group. Both are report-only —
they mutate nothing — and compose ``compute_content_hash`` rather than
re-implementing directory hashing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.ids import compute_content_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

    from molexp.workspace.run import Run

__all__ = ["consolidate_workflow_source", "dedupe_workflow_source"]


def dedupe_workflow_source(runs: Iterable[Run]) -> dict[str, list[str]]:
    """Group run ids by the content hash of their ``source/`` snapshot.

    A run whose ``run_dir/source`` directory does not exist is skipped (it has
    no captured source to compare).

    Args:
        runs: The runs to group.

    Returns:
        A ``{content_hash: [run_id, ...]}`` mapping over the runs that have a
        ``source/`` snapshot.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        source_dir = Path(str(run.run_dir)) / "source"
        if not source_dir.is_dir():
            continue
        groups[compute_content_hash(source_dir)].append(run.id)
    return dict(groups)


def consolidate_workflow_source(runs: Iterable[Run]) -> dict[str, str]:
    """Map each duplicate run id to the canonical id of its source group.

    For every :func:`dedupe_workflow_source` group with more than one member,
    the canonical id is the first when sorted; the remaining ids map to it.
    Single-member groups are excluded. Report-only — nothing is mutated.

    Args:
        runs: The runs to consolidate.

    Returns:
        A ``{duplicate_run_id: canonical_run_id}`` mapping; empty when there are
        no duplicate source groups.
    """
    mapping: dict[str, str] = {}
    for members in dedupe_workflow_source(runs).values():
        if len(members) < 2:
            continue
        canonical, *duplicates = sorted(members)
        for duplicate in duplicates:
            mapping[duplicate] = canonical
    return mapping
