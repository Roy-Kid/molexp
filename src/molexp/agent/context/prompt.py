"""PromptComposer: deterministic layered system prompt assembly.

Per spec §6.1 the layers are:

1. Base prompt (built into the harness or supplied by config).
2. Workspace addendum (workspace-level instructions).
3. Skill addendum (re-resolved every turn from ``SkillStore``).
4. Per-session ``instructions_override`` (replaces 1-3 entirely when
   set).

Each non-empty section is wrapped in a deterministic header so
prompts are stable across runs and trivially diffable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLayer:
    """One labeled section of the assembled system prompt."""

    title: str
    body: str


class PromptComposer:
    """Compose a system prompt from layered sections."""

    SECTION_HEADER = "## {title}"

    def compose(
        self,
        *,
        base: str,
        workspace: str = "",
        skill: str = "",
        override: str | None = None,
    ) -> str:
        """Return the rendered system prompt.

        ``override`` short-circuits all layering when set; the harness
        stores it as a flat string (spec §5.1) and the composer never
        merges it with other sources.
        """

        if override is not None:
            return override.strip()

        layers = [
            PromptLayer("Base", base),
            PromptLayer("Workspace", workspace),
            PromptLayer("Skill", skill),
        ]
        sections = [self._render(layer) for layer in layers if layer.body.strip()]
        return "\n\n".join(sections).strip()

    def _render(self, layer: PromptLayer) -> str:
        header = self.SECTION_HEADER.format(title=layer.title)
        return f"{header}\n{layer.body.strip()}"
