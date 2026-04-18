---
name: molexp-design
description: Audit or polish molexp frontend for visual quality, information density, and design system consistency. Use for UI polish passes, dense-layout redesigns, or before shipping a UI feature. Delegates to the molexp-designer agent for review and applies fixes.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob, Agent
argument-hint: <renderer, panel, or "diff" to audit current changes>
---

Audit or polish UI for: $ARGUMENTS

If $ARGUMENTS is empty or `diff`, audit files from `git diff --name-only HEAD -- 'ui/**'`.

## Phase 1: Ground Truth

Read the design stack to anchor the review:

1. `ui/src/styles/tailwind.css` — current theme tokens (zinc base, semantic `--color-*`)
2. `ui/src/components/ui/` — available primitives; never re-roll these
3. Two or three existing renderers in `ui/src/app/renderers/` — for consistency patterns
4. `ui/src/app/layout/` and `ui/src/app/panels/` — three-panel conventions

## Phase 2: Review

Invoke the `molexp-designer` agent on the target(s). Its output is the authoritative review.

- If $ARGUMENTS names a renderer/file, pass that path.
- If reviewing a diff, pass the file list from `git diff`.

Wait for its ✅/⚠️/❌ report across the 10 dimensions (density, hierarchy, tokens, spacing, interaction, states, a11y, icons, responsive, consistency).

## Phase 3: Apply Fixes

Work through the report by severity:

1. CRITICAL first: a11y blockers, token violations, contrast failures.
2. HIGH next: density regressions, missing empty/error states, inconsistency with neighbors.
3. MEDIUM when cheap: spacing drift, icon size fixes, missing hover.
4. LOW only if the user requests polish.

For each fix:
- Prefer a minimal diff using existing primitives (`Card`, `Badge`, `Tooltip`, `ScrollArea`, `Button`, …)
- Replace raw colors with semantic tokens (`bg-background`, `text-muted-foreground`, `border-border`)
- Collapse wasteful spacing (`py-8` → `py-2`, `text-2xl` → `text-sm font-semibold`)
- Ensure hover/selected/focus states on every interactive row
- Ensure loading (`<Skeleton>`), empty, and error branches exist

## Phase 4: Verify

```bash
cd ui && npx tsc --noEmit
cd ui && npm test
cd ui && npm run dev:mock   # visually confirm when a dev server is expected
```

If visual verification is not possible in this environment, say so explicitly — do not claim the change "looks good" without seeing it.

## Rules

- Do not edit `ui/src/api/generated/`.
- Do not invent new design tokens — add to `tailwind.css` only if the user confirms a theme change.
- Do not introduce a new UI primitive if a suitable one exists in `components/ui/`.
- Density first: if a change makes the layout sparser without a user-visible reason, reject it.
- This skill is for **visual quality**. For behavior changes, state reshaping, or new features, use `/molexp-ui` instead.
