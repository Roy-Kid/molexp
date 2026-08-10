"""molexp.services — the application-service layer shared by CLI and server.

One backend code path per user-facing operation ("Python operation ≡ UI
operation"): the CLI commands (``molexp plan`` / ``molexp curate`` /
``molexp config``) and the server routes both delegate here instead of
duplicating logic or importing each other.

Contents:

- :mod:`molexp.services.plan_runtime` — PlanMode gateway builder + preflight,
  background-task registry, and the post-run record/persist/materialize steps.
- :mod:`molexp.services.curate_runtime` — the shared curation flow
  (discover → plan → gate → invoke) + proposal backend.
- :mod:`molexp.services.operator_config` — the ``~/.molexp/config.json``
  loader + the ``molexp.config`` bridge (one source of truth for the path,
  the parsing, and the ``agent.model`` key).
- :mod:`molexp.services.agent_task_store` — on-disk metadata/events for
  user-facing agent tasks (consumed by the Agents-hub routes and the plan
  recorder).
- :mod:`molexp.services.auth` — filesystem users + sessions for
  ``molexp serve`` HTTP auth (CLI ``molexp auth`` and the server share it).

Layer rules: services may import ``molexp.harness`` / ``molexp.agent`` /
``molexp.workflow`` / ``molexp.workspace`` (and cross-layer primitives);
it MUST NOT import ``molexp.server`` or ``molexp.cli`` — those application
shells sit *above* services and import it, never the reverse. Enforced by
``tests/test_services/test_import_guard.py``.

No eager re-exports: importing :mod:`molexp.services` stays light; consumers
import the specific submodule they need. (``build_mount_context`` /
``resolve_scope_dir`` re-export lazily via ``__getattr__`` — the one
package-level convenience, spelled out in vision-loop-11.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.services.agent_context import build_mount_context, resolve_scope_dir

__all__ = ["build_mount_context", "resolve_scope_dir"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from molexp.services import agent_context

        return getattr(agent_context, name)
    raise AttributeError(f"module 'molexp.services' has no attribute {name!r}")
