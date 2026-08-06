"""TensorBoard scalars → MolRec metric records.

Delegates tfevents parsing to :mod:`molexp.plugins.tensorboard`, which already
owns that surface; this module is only the mapping to metric records.

Unlike LAMMPS thermo, tfevents points **do** carry a real wall-clock, so it is
passed through rather than stamped at ingest. Tag names (``train/loss``) are
already the MolRec ``key`` convention and are used verbatim.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from molexp._typing import JSONValue


def scalar_records(
    run_dir: Path | str,
    *,
    tags: tuple[str, ...] | None = None,
    extra_tags: dict[str, JSONValue] | None = None,
) -> Iterator[dict[str, JSONValue]]:
    """Yield one metric record per scalar point under *run_dir*.

    Args:
        run_dir: Directory to search for tfevents logdirs.
        tags: Restrict to these tfevents tag names; ``None`` reads all.
        extra_tags: Tags merged into every emitted record.

    Yields:
        Compact-key metric records ready for
        :meth:`molexp.workspace.metrics.MetricsWriter.log_many`.

    Raises:
        ImportError: When the optional ``tensorboard`` dependency is missing.
    """
    from molexp.plugins.tensorboard import (
        discover_logdirs,
        read_scalars,
        require_tensorboard,
    )

    require_tensorboard()
    root = Path(run_dir)

    for logdir in discover_logdirs(root):
        for series in read_scalars(logdir, tags=tags, relative_to=root):
            record_tags: dict[str, JSONValue] = {"logdir": series.logdir}
            if extra_tags:
                record_tags.update(extra_tags)
            for point in series.points:
                yield {
                    "t": "scalar",
                    "k": series.tag,
                    "s": point.step,
                    "w": datetime.fromtimestamp(point.wall_time, UTC).isoformat(),
                    "v": point.value,
                    "tags": record_tags,
                }
