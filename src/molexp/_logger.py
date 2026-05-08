"""molexp logger — subclasses mollog.Logger to add an agent-trace verb.

This module is the recommended extension pattern for mollog plugins:

* Subclass :class:`mollog.Logger` rather than monkey-patching it.
* Provide a package-local ``get_logger`` factory that returns the subclass.
* Reuse mollog's :class:`LoggerManager` for the root handler and parent
  chain so records propagate into mollog's configured sinks.

Users who want the molexp verbs (``.ice()``) import from molexp::

    from molexp import get_logger

    log = get_logger(__name__)
    log.info("classic mollog verb")
    log.ice("agent step", agent_id=a.id, step=3)

Users who only need mollog's standard verbs continue to import from mollog
unchanged. ``import molexp`` does **not** mutate :class:`mollog.Logger`.
"""

from __future__ import annotations

import threading

import mollog

from molexp._typing import JSONValue


class Logger(mollog.Logger):
    """molexp-aware logger.

    Inherits every standard verb from :class:`mollog.Logger`
    (``trace`` / ``debug`` / ``info`` / ``warning`` / ``error`` /
    ``critical`` / ``exception``) and adds an agent-trace verb.
    """

    def ice(self, message: str, **fields: JSONValue) -> None:
        """Emit an agent-trace event tagged ``verb="ice"``.

        Records flow through the same dispatch path as :meth:`info`, so
        any handler attached to the logger (or any ancestor) receives
        them. The ``verb`` extra field lets formatters / filters
        distinguish molexp agent events from generic info logs.
        """
        extra = {**fields, "verb": "ice"} if fields else {"verb": "ice"}
        self._log(mollog.Level.INFO, message, extra)


_lock = threading.RLock()
_cache: dict[str, Logger] = {}


def get_logger(name: str = "") -> Logger:
    """Return a cached :class:`Logger` parented to mollog's root.

    Mirrors :func:`mollog.get_logger` but constructs molexp's subclass so
    callers see ``.ice()`` in addition to the standard verbs. The parent
    chain is set up via mollog's :class:`LoggerManager` so configured
    handlers on the root (or intermediate ancestors) still receive the
    records.
    """
    with _lock:
        cached = _cache.get(name)
        if cached is not None:
            return cached

        manager = mollog.LoggerManager()
        manager.ensure_default_handler()

        logger = Logger(name)
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            logger.parent = manager.get_logger(parent_name)
        else:
            logger.parent = manager.root

        _cache[name] = logger
        return logger


def _reset_cache() -> None:
    """Clear the molexp logger cache (test-only)."""
    with _lock:
        _cache.clear()
