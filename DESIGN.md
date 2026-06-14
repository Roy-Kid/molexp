# Design

Captured from the live UI in `ui/src/` (shadcn "new-york" + Tailwind 4). Source of truth for tokens: `ui/src/styles/tailwind.css`; component conventions from `ui/src/components/ui/`, `ui/src/app/`, and `ui/src/app/components/entity/`. This file documents what exists so variants stay on-brand — edit it when the real tokens change, and re-run `/impeccable document` after major UI shifts. (Note: the workflow canvas now lives in `ui/src/components/workflow/`; refresh this file if the token set has drifted on `dev`.)

## Theme

Precise, instrument-like research IDE. A three-panel layout (left nav tree · center content · right inspector) over a `Workspace → Project → Experiment → Run → Execution` hierarchy. Visual language is the shadcn **new-york** style on a **neutral / zinc** base: near-black-on-white in light, near-white-on-near-black in dark, with a small set of semantic status colors (success / warning / info / destructive) carrying the only saturated hues. Full light + dark parity, kept in sync with molvis. Restraint is the brand: density and legibility over decoration.

## Color

HSL channel triplets stored in CSS custom properties, consumed via `hsl(var(--token))` mapped into Tailwind `@theme` color names. Base color: neutral (zinc). Verify exact values against `ui/src/styles/tailwind.css` on `dev`.

| Role | Token | Notes |
|---|---|---|
| Background / Foreground | `--background` / `--foreground` | app surface + ink |
| Canvas | `--canvas` | editor / graph surface |
| Card / Popover | `--card` / `--popover` | raised surfaces |
| Primary | `--primary` | near-black (neutral, **not** a brand hue) |
| Secondary / Muted / Accent | `--secondary` / `--muted` / `--accent` | fills; `--muted-foreground` for meta text |
| Border / Input / Ring | `--border` / `--input` / `--ring` | hairlines + focus ring |
| Success / Warning / Info / Destructive | `--success` / `--warning` / `--info` / `--destructive` | each with `-foreground` and (success/warning/info) `-soft` variants |

**Strategy:** Restrained. Tinted neutrals + status colors only; primary is achromatic near-black/near-white. Saturated color is reserved for *meaning* (run state, info), never decoration. **Status must pair color with an icon/shape/label** — never color alone (WCAG AA, color-blind safety).

## Typography

System font stack (no custom webfonts) — fast, native, instrument-appropriate. Tailwind type scale: `text-xs` (labels/meta), `text-sm` (body/UI default), `text-base`, `text-2xl` (stat values / page titles). Weights: 400 / 500 (labels, buttons) / 600 (headings, values). Section/card headers ~11px `tracking-wide`, uppercase only as a deliberate component label — not a per-section eyebrow. Prose ≤75ch; tables may run denser. Mono (editor) via Monaco.

## Spacing & Layout

- **Shell:** `react-resizable-panels` — Left `{default 22%, min 16%, max 30%}` · Right inspector `{default 30%, min 20%, max 45%}`, toggleable · Center flex-1. Single-row `ContextBar` (breadcrumb · search · refresh · inspector toggle) on top.
- **Density tokens** (`density.ts`): compact rows `py-1.5 px-2`; default rows `py-3 px-6`. Sticky table headers, `divide-y divide-border/50`, hover `hover:bg-muted/40`.
- **Grids:** stat grids step 2→3→4→5 cols (xs→sm→lg→xl). Prefer `repeat(auto-fit, minmax(…, 1fr))`; flex for 1D.
- **Radius:** `--radius` base → `-lg` / `-md` (−2px) / `-sm` (−4px). Restrained, not pill-shaped.

## Components

Radix UI primitives wrapped shadcn-style; `class-variance-authority` for variants; `cn()` (clsx + tailwind-merge) in `@/lib/utils`; icons from **lucide-react**. Prefer existing `@/components/ui/*` before adding primitives.

Domain components to reuse: **StatCard**, **StatusDonut**, **MiniBars**, **DashboardCard** (entity dashboards); **FlowgramCanvas** + **WorkflowGraphViewer** (workflow graph, UML activity-diagram node styling under `ui/src/components/workflow/`); **AgentViewer**, **AssetViewer**, **RunMetricsTab** (live MolvisLineChart). **StatusBadge** is the canonical color+dot+label status primitive — never color alone.

## Motion

Functional only. Existing keyframes: `accordion-down/up` at `0.2s ease-out`. Use ease-out (no bounce/elastic). Animate transform/opacity, not layout. **Every animation needs a `prefers-reduced-motion: reduce` fallback** (crossfade/instant). No decorative glass/parallax.

## Iconography

lucide-react throughout. Pair status icons/dots with status color so state never relies on hue alone.
