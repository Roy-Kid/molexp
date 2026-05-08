"""CLI plugin contract and entry-point discovery for third-party molexp packages.

A third-party Python package depending on ``molexp`` declares a CLI
entry point in its ``pyproject.toml``::

    [project.entry-points."molexp.cli_plugins"]
    myplugin = "myplugin.cli:plugin"

The referenced symbol must be a :class:`CliPlugin` instance. molexp's CLI
boot sequence calls :func:`discover_cli_plugins` and invokes each
plugin's ``register(app)`` to attach subcommands or sub-Typers.

The contract is intentionally distinct from the UI plugin channel
(:mod:`molexp.plugins.ui`) — CLI runs in the Python process at startup,
UI runs in the browser via dynamic ESM import. Their lifecycles,
dependencies, and version cadence are independent, so they each have
their own entry-point group and their own API version constant.

Security note: discovered plugins run in the host molexp process with
its full privileges. Treat plugin packages the same way you would treat
any pip-installed dependency — only install ones you trust.
"""

from __future__ import annotations

import functools
import importlib.metadata as importlib_metadata
from collections.abc import Callable
from typing import TYPE_CHECKING

from mollog import get_logger

if TYPE_CHECKING:
    import typer


CLI_PLUGIN_API_VERSION = "1"
"""Current CLI plugin API version. Plugins whose ``api_version`` does
not match this value are skipped at discovery time."""


ENTRY_POINT_GROUP = "molexp.cli_plugins"

logger = get_logger(__name__)


class CliPlugin:
    """Public descriptor for a third-party molexp CLI extension.

    Plain Python class with explicit ``__init__`` (not a stdlib dataclass,
    not a pydantic ``BaseModel``) because it carries a live ``register``
    callable. Treated as logically immutable by convention; ``__slots__``
    blocks attribute assignment after construction.

    Attributes:
        id: Globally unique short identifier (kebab-case recommended).
        name: Human-readable display name.
        version: Plugin's own semver string.
        register: Required callable receiving the molexp Typer app and
            attaching subcommands or sub-Typers to it. Unlike the
            legacy ``register_cli`` field, this has no ``None`` default
            — a plugin that doesn't register CLI commands has no
            business in the ``molexp.cli_plugins`` group.
        api_version: The CLI plugin contract version targeted. Currently
            only ``"1"`` is accepted; mismatches are skipped.
    """

    __slots__ = ("api_version", "id", "name", "register", "version")

    # Class-level annotations let static type-checkers see the slots —
    # the runtime values are populated via ``object.__setattr__`` because
    # the class is treated as logically immutable after construction.
    id: str
    name: str
    version: str
    register: Callable[[typer.Typer], None]
    api_version: str

    def __init__(
        self,
        *,
        id: str,
        name: str,
        version: str,
        register: Callable[[typer.Typer], None],
        api_version: str = CLI_PLUGIN_API_VERSION,
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "register", register)
        object.__setattr__(self, "api_version", api_version)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot assign to {name!r}")


@functools.cache
def _discover_cli_uncached() -> tuple[CliPlugin, ...]:
    """Walk the ``molexp.cli_plugins`` entry-point group exactly once.

    Cached at module level. Tests reset via ``cache_clear()``.
    """
    eps = importlib_metadata.entry_points()
    group_iter = eps.select(group=ENTRY_POINT_GROUP)

    seen: set[str] = set()
    discovered: list[CliPlugin] = []

    for ep in group_iter:
        plugin = _safe_load(ep)
        if plugin is None:
            continue
        if plugin.id in seen:
            ep_name = getattr(ep, "name", "<unknown>")
            logger.warning(
                f"duplicate plugin id '{plugin.id}' from entry point '{ep_name}' — keeping first"
            )
            continue
        seen.add(plugin.id)
        discovered.append(plugin)

    return tuple(discovered)


def _safe_load(ep) -> CliPlugin | None:
    """Load a single entry point, returning ``None`` on any error."""
    name = getattr(ep, "name", "<unknown>")
    try:
        obj = ep.load()
    except Exception as exc:
        logger.warning(f"failed to load CLI plugin entry point '{name}': {exc}")
        return None

    if not isinstance(obj, CliPlugin):
        logger.warning(
            f"entry point '{name}' did not return a CliPlugin instance "
            f"(got {type(obj).__name__}); skipping"
        )
        return None

    if obj.api_version != CLI_PLUGIN_API_VERSION:
        logger.warning(
            f"plugin '{obj.id}' targets api_version='{obj.api_version}' but "
            f"molexp expects '{CLI_PLUGIN_API_VERSION}'; skipping"
        )
        return None

    return obj


def discover_cli_plugins() -> tuple[CliPlugin, ...]:
    """Return all discoverable third-party :class:`CliPlugin` instances.

    Cached after the first call. Failures during entry-point load are
    swallowed and logged — the caller always receives a tuple, possibly
    empty, never an exception.
    """
    return _discover_cli_uncached()


__all__ = [
    "CLI_PLUGIN_API_VERSION",
    "ENTRY_POINT_GROUP",
    "CliPlugin",
    "discover_cli_plugins",
]
