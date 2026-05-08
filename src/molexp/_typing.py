"""Cross-layer type aliases.

Lives at the package root so that **every** molexp layer (workflow,
workspace, agent, server, plugins) can import these names without
introducing layer-cycle violations. Keep this module **free of any
non-stdlib imports** so the layering DAG stays clean.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

# ── Type aliases for genuinely-opaque user payloads ────────────────────────
#
# These aliases mark positions where the type system *cannot* help by design
# — user-supplied task return values, inputs, deps, message-bus payloads,
# and asset-store values are open-world Python objects. The aliases make
# the boundary explicit:
#
#   ✓ ``output: TaskOutput``  → "intentionally any user value at this point"
#   ✗ ``output: Any``         → flagged by ANN401 as a missing annotation
#
# Ruff's ANN401 sees the alias name, not the resolved type, so the alias
# passes the lint while keeping ``Any``'s runtime semantics. Use sparingly
# — only at positions that genuinely accept user-defined open-world data.

#: Return value of a user task body — open-world Python object.
TaskOutput: TypeAlias = Any
#: Input value flowing into a user task body — open-world Python object.
TaskInput: TypeAlias = Any
#: Application-level deps the caller forwards through ``Workflow.execute(deps=…)``.
UserDeps: TypeAlias = Any
#: Message payload on the actor channel bus — opaque to the workspace.
ChannelMessage: TypeAlias = Any
#: Hashable, JSON-shaped payload used for fingerprints / canonical hashing.
HashablePayload: TypeAlias = Any

# ── JSON-shaped recursive type ─────────────────────────────────────────────
#
# Replaces ``Mapping[str, Any]`` at config / persistence / API boundaries.
# Keeps the type-checker honest while still letting user configuration nest
# freely. Uses PEP 695 ``type`` syntax — required for recursion without
# forward strings.

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONMapping = Mapping[str, JSONValue]
