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

Layer rules: services may import ``molexp.harness`` / ``molexp.agent`` /
``molexp.workflow`` / ``molexp.workspace`` (and cross-layer primitives);
it MUST NOT import ``molexp.server`` or ``molexp.cli`` — those application
shells sit *above* services and import it, never the reverse. Enforced by
``tests/test_services/test_import_guard.py``.

No eager re-exports: importing :mod:`molexp.services` stays light; consumers
import the specific submodule they need.
"""

from __future__ import annotations
