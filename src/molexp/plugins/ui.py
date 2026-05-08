"""UI plugin directory discovery for third-party molexp packages.

A third-party package contributes a UI bundle by declaring an entry
point::

    [project.entry-points."molexp.ui_plugins"]
    myplugin = "myplugin.ui:bundle_dir"

The referenced symbol is either a :class:`pathlib.Path` directly, or a
zero-argument callable returning a ``Path``. The directory it points
to is mounted by the FastAPI server under ``/api/plugins/{id}/`` so
the browser can fetch ``manifest.json`` + ``index.js`` from there.

This module is **deliberately tiny** — it does not define a Python
``UiPlugin`` dataclass, does not carry an ``api_version`` field, and
does not parse ``manifest.json``. All UI semantics (id / name /
version / api_version / entry / capabilities) live in the bundle's
``manifest.json``, which is parsed by the browser-side loader. Python
is the distribution shim only.

Failures at every step (entry-point load, callable invocation, path
validation) are isolated with a warning so a single broken plugin
cannot prevent the server from starting.

Security note: discovered plugin bundles are mounted as static assets
served from the host molexp process. Treat them as you would any
pip-installed dependency — only install ones you trust.
"""

from __future__ import annotations

import functools
import importlib.metadata as importlib_metadata
from pathlib import Path

from mollog import get_logger

ENTRY_POINT_GROUP = "molexp.ui_plugins"

logger = get_logger(__name__)


@functools.cache
def _discover_ui_uncached() -> dict[str, Path]:
    """Walk the ``molexp.ui_plugins`` entry-point group exactly once.

    Cached at module level. Tests reset via ``cache_clear()``.
    """
    eps = importlib_metadata.entry_points()
    group_iter = eps.select(group=ENTRY_POINT_GROUP)

    discovered: dict[str, Path] = {}

    for ep in group_iter:
        ep_name = getattr(ep, "name", "<unknown>")
        path = _safe_resolve(ep)
        if path is None:
            continue
        if ep_name in discovered:
            logger.warning(f"duplicate UI plugin id '{ep_name}' — keeping first")
            continue
        discovered[ep_name] = path

    return discovered


def _safe_resolve(ep) -> Path | None:
    """Resolve a single entry point to a directory, ``None`` on error."""
    name = getattr(ep, "name", "<unknown>")
    try:
        obj = ep.load()
    except Exception as exc:
        logger.warning(f"failed to load UI plugin entry point '{name}': {exc}")
        return None

    if isinstance(obj, Path):
        path: Path | None = obj
    elif callable(obj):
        try:
            path = obj()
        except Exception as exc:
            logger.warning(f"UI plugin '{name}' callable raised: {exc}")
            return None
    else:
        logger.warning(
            f"UI plugin entry point '{name}' must be a Path or zero-arg "
            f"callable returning Path (got {type(obj).__name__}); skipping"
        )
        return None

    if path is None or not isinstance(path, Path):
        logger.warning(f"UI plugin '{name}' resolved to {path!r} (not a Path); skipping")
        return None

    if not path.is_dir():
        logger.warning(f"UI plugin '{name}' path {path} is not a directory; skipping")
        return None

    return path


def discover_ui_plugin_dirs() -> dict[str, Path]:
    """Return ``{plugin_id: bundle_dir}`` for every installed UI bundle.

    ``plugin_id`` is the entry-point name. Cached after the first call.
    Failures during entry-point load are swallowed and logged — the
    caller always receives a dict, possibly empty, never an exception.
    """
    return _discover_ui_uncached()


__all__ = [
    "ENTRY_POINT_GROUP",
    "discover_ui_plugin_dirs",
]
