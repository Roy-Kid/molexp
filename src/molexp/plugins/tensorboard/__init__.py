"""TensorBoard tfevents parsing — optional, behind ``molexp[tensorboard]``.

Public surface:
    - :func:`discover_logdirs` walks a run directory and returns the set
      of directories containing ``events.out.tfevents.*`` files.
    - :func:`read_scalars` parses scalar series from one logdir into a
      list of :class:`ScalarSeries`.
    - :func:`require_tensorboard` raises a friendly ``ImportError`` when
      the optional dep is missing; route handlers translate this to a
      503 response.

The parser is intentionally thin — tfevents → typed Python — so the
HTTP layer can stay free of tensorboard-specific imports until a
request that actually needs them lands.
"""

from molexp.plugins.tensorboard.parser import (
    ScalarPoint,
    ScalarSeries,
    discover_logdirs,
    read_scalars,
    require_tensorboard,
)

__all__ = [
    "ScalarPoint",
    "ScalarSeries",
    "discover_logdirs",
    "read_scalars",
    "require_tensorboard",
]
