"""``ResolveCapabilities`` — plan step 3: pick the capabilities this needs.

molmcp grounds the FULL molcrafts toolchain onto ``ctx.capability_registry``;
this stage asks an LLM (the ``capability_selector`` agent) to choose the
*minimal* subset that realizes the concrete ``experiment_spec``, then renders
only those into the binder-facing ``capability_catalog`` artifact. The registry
stays full, so :class:`ValidateBoundWorkflow` can still validate any bound id;
the *catalog* the binder reads is narrowed to "what this experiment needs".

Three branches keep the pipeline shape (and audit trail) identical in every
configuration:

1. **No registry** — emit the explicit "no capabilities resolved" notice.
2. **Registry but no agent gateway** — render the FULL catalog under a loud note
   that the LLM selector was unavailable (transparent degradation, never
   silent).
3. **Registry + gateway** — the LLM selects; render the narrowed catalog. An
   empty / all-unrecognized pick falls back to the full catalog under a loud
   note so the downstream binder is never starved.
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.prompts.capability_catalog import (
    render_capability_catalog,
    render_selected_capability_catalog,
)
from molexp.harness.schemas import AgentCallSpec, ArtifactRef, CapabilitySelection
from molexp.harness.stages._resolve import require_latest

__all__ = ["ResolveCapabilities"]

_NO_REGISTRY_CATALOG = (
    "## Available molcrafts capabilities\n\n"
    "No capability registry was grounded for this run — the toolchain was "
    "not discovered. Binding will proceed unguided and validation may reject "
    "invented capabilities.\n"
)

_NO_GATEWAY_NOTE = "> LLM selector unavailable — showing the full grounded catalog.\n\n"
_EMPTY_SELECTION_NOTE = (
    "> Capability selector returned no usable capabilities — showing the full grounded catalog.\n\n"
)


class ResolveCapabilities(Stage):
    """LLM-select the capabilities this experiment needs from the grounded toolchain."""

    name: ClassVar[str] = "resolve_capabilities"

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        registry = ctx.capability_registry
        if registry is None:
            return ctx.artifact_store.put_text(
                kind="capability_catalog",
                text=_NO_REGISTRY_CATALOG,
                created_by=f"stage:{self.name}",
                parent_ids=[],
            )

        full_catalog = render_capability_catalog(registry.list_capabilities())
        if ctx.agent_gateway is None:
            return ctx.artifact_store.put_text(
                kind="capability_catalog",
                text=_NO_GATEWAY_NOTE + full_catalog,
                created_by=f"stage:{self.name}",
                parent_ids=[],
            )

        # LLM selection: feed the concrete spec + the full catalog, get back the
        # minimal subset this experiment needs.
        spec = require_latest(ctx, "experiment_spec", stage=self.name)
        catalog_ref = ctx.artifact_store.put_text(
            kind="prompt",
            text=full_catalog,
            created_by=f"stage:{self.name}",
            parent_ids=[spec.id],
        )
        result = await ctx.agent_gateway.call(
            AgentCallSpec(
                agent_name="capability_selector",
                input_artifact_ids=[spec.id],
                prompt_artifact_id=catalog_ref.id,
                output_schema=CapabilitySelection.model_json_schema(),
            )
        )
        selection = CapabilitySelection.model_validate_json(
            ctx.artifact_store.get(result.output_artifact.id)
        )
        if any(registry.has(sel.id) for sel in selection.selected):
            catalog = render_selected_capability_catalog(registry, selection)
        else:
            catalog = _EMPTY_SELECTION_NOTE + full_catalog
        return ctx.artifact_store.put_text(
            kind="capability_catalog",
            text=catalog,
            created_by=f"stage:{self.name}",
            parent_ids=[result.output_artifact.id],
        )
