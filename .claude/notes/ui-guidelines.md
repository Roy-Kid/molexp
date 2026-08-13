# UI Guidelines — MolExp

Product-local UI record, maintained by `/mol:ui`. The shared MolCrafts
constitution is **not** restated here; it lives in the `mol` plugin at
`skills/ui/references/visual-language.md`. This file records only what
*this product* decided.

Human prose may be added anywhere outside the managed markers below and
will never be rewritten.

<!-- mol:ui:begin -->

## Surface

| | |
|---|---|
| Frontend root | `apps/web/` |
| Archetype | `workbench` |
| Default theme | light, with a real dark theme |
| Token layer | `apps/web/src/styles/tailwind.css` |
| Last ladder stage applied | `motion` on `2026-07-27` (ladder complete); token/density hygiene on `2026-07-31`; **`info` experiment** on `2026-08-09` (entity overviews) |

## Accent

```css
:root { --molexp-accent: oklch(0.55 0.19 295); }
.dark { --molexp-accent: oklch(0.72 0.16 295); }
```

Hue 295 (violet) stays ≥40° from `--status-running` (250). Product accent
never signals run/plan state — running uses the status ramp only.

## Token decisions

MolExp owns an independent OKLCH token layer (not shared with MolVis).
Workbench surfaces: `background`, `surface`, `surface-subtle`, `canvas`.
`interactive` is the neutral hover surface so accent stays for emphasis and
selection. Fixed MolCrafts status vocabulary has separate accessible
foreground and soft-surface tokens. UI text uses the 11–24px semantic ramp
(`text-micro` … `text-display`); controls use named 28/32/36px heights;
coarse-pointer controls expand to a 44px minimum target.

Legacy shadcn aliases (`primary`, `success`, `destructive`, `info`,
`warning`) map onto accent / status tokens for existing feature code.
`info` aliases **running status**, not the product accent.

Scientific chart series palettes (including ΔF groups) use oklch
categorical colors — not brand/status tokens.

Stage 2 conformance is enforced across `apps/web/src/`: feature code contains no
raw Tailwind palette utilities, arbitrary font sizes, off-scale
margin/padding/gap values, multi-pixel borders, or unnamed shadows. Overlay
scrims and the single overlay elevation are local semantic tokens. Version and
configuration diffs own an independent `diff-added|removed|modified|unchanged`
ramp so they never borrow run-status meaning.

## Layout shell

Declared frame in `AppShell` (every route renders into it):

```text
┌───────────────────────────────────────────────────────────────┐
│ ContextBar — MolExp · filter · approvals · refresh      44px  │
├────────────┬─────────────────────────────────┬────────────────┤
│ Navigator  │ Breadcrumb + inspector toggle   │ Node Inspector │
│ tree 28px  │ Work surface (canvas)           │ (toggleable)   │
│ rows       │                                 │                │
│ 16–30%     │                                 │ 20–45%         │
├────────────┴─────────────────────────────────┴────────────────┤
│ Status bar (28px): ♡ heartbeat · Syncing / operation message  │
└───────────────────────────────────────────────────────────────┘
```

- Global toolbar (`ContextBar`) is **44px** (`h-11`).
- Work-surface header (breadcrumb) is **40px** (`h-10`).
- The status bar is always **28px** and never expands into a panel.
- Navigator/work-surface and work-surface/inspector layouts persist independently
  through stable panel ids. Conditional inspector layouts do not overwrite the
  single-panel work-surface layout.
- Runs reuses the declared `AppShell` inspector; it does not create a second
  fixed-width inspector inside the work surface.
- Heartbeat sits at the left of the status line; **click opens a connection-
  status popover** (active workspace label/path, link state, remote index). It
  does **not** trigger refresh — ContextBar and the left-panel list headers own
  refresh. Idle is neutral, an active refresh shows `Syncing…` in running blue,
  and completion gets one neutral 180ms acknowledgement.
- Active served workspace identity is always visible: ContextBar chip next to
  MolExp, and a mono subtitle under the Projects list header (even for a single
  remote mount such as `Arrhenius:/home/…`).
- Mobile: nav + the same stateful inspector become edge drawers; the status bar
  stays full-width.

## Product components

| Component | Wraps | Owns |
|---|---|---|
| `WorkbenchStatusStrip` | heartbeat + activity live region | Sync and operation status |
| `WorkflowNode` | domain node chrome | DAG node (lists + flowgram) |
| `NodeInspector` / `Section` / `Row` | semantic aside + section layout | Right-rail inspector frame |
| `ParameterField` / `ParameterGroup` | Input/Select | Schema-driven params |
| `RunStatusBadge` | `StatusBadge` + canonical status boundary | Run/execution status chips |
| `WorkbenchTag` | `Badge` | Category, metadata, selection, and outcome tags |
| `WorkbenchAction` / `Icon` / `Toggle` | `Button` | Product action vocabulary |
| `WorkbenchOperationState` | Skeleton / live region | loading·empty·error·disabled·running·success |
| Plan agent set | rail / deliverables / review | PlanOrchestrator UI |

Live under `apps/web/src/components/workbench/`. Feature chrome uses
`WorkbenchAction*` and `WorkbenchTag`; base `Button variant=` and `Badge`
remain implementation details of base or product/entity wrappers.

Stage 3 component conformance:

- Run and entity status presentation shares the nine-value canonical vocabulary;
  wire aliases normalize once at the presentation boundary.
- Primary data-table and runs-table rows activate with pointer, Enter, or Space
  and expose stable accessible names and focus treatment.
- Toolbars, headers, row actions, and inline feedback prefer borderless
  `WorkbenchIconAction` / `WorkbenchToggleAction` controls whenever a stable
  Lucide icon exists. Every icon action has a precise `label` (accessible name
  plus native tooltip). Text buttons are reserved for form confirmation,
  consequential review decisions, named navigation/modes, and actions whose
  icon would be ambiguous. The global command palette exposes
  combobox/listbox semantics.
- The plan progress rail becomes a horizontally scrollable stage selector on
  narrow screens and a vertical rail at the large breakpoint.
- Product component files keep one primary responsibility: status-line activity;
  inspector rows/sections; and parameter groups are split into focused modules.

## Surface hierarchy

Stage 4 de-card conformance:

- Page regions use a section label, whitespace, and one-dimensional separators;
  they do not get a rounded bordered `bg-card` wrapper.
- KPI/stat groups, timelines, tables, settings field groups, empty regions, and
  ordinary list rows are flat. Selected rows may use a quiet semantic fill.
- A full object boundary is reserved for a one-to-one domain object or primary
  control/artifact that can reasonably be opened, edited, copied, enlarged,
  dragged, removed, or acted on independently.
- Preserved object surfaces include workflow nodes, draggable dashboard panels,
  provider credentials, skills, MCP servers, approval requests, entity
  references, conversation turns, and reviewable code/chart artifacts.
- Status notices use their semantic soft token with a linear border rather than
  generic card chrome.

## Operation states

Stage 5 state conformance:

- **Transient app status (sync, mutation tips, errors) lives only in the
  bottom status bar** — the same channel and presentation as MolVis.
  No floating center/corner toast cards. Heartbeat shows live sync; bus
  messages (`toast` / `reportStatus`) share the mono activity line;
  warning/error click-to-dismiss. `WorkbenchOperationState` remains the
  in-surface loading/empty/error pattern for panels that fetch content.
- `WorkbenchOperationState` is the single accessible surface for loading,
  final empty, error, disabled, running, and success feedback. It owns
  `status` / `alert` live-region semantics and default, compact, and toolbar
  densities.
- Every audited fetching/computing surface uses it: ProjectViewer, AssetViewer,
  both workflow file viewers, DocTree, KnowledgeDocPanel,
  KnowledgeBacklinksCard, ApprovalsBell / Inbox, GlobalCommandPalette, and
  ModelPicker, in addition to the existing image, editor,
  activity, and log surfaces.
- Initial loading never renders as final empty, request failures never collapse
  to `[]`, `null`, “not found,” or zero metrics, and zero-length payloads get a
  deliberate final-empty state.
- Read failures stay visible with an explicit Retry. Background refresh keeps
  valid prior content visible under a running state.
- Mutations disable their initiating controls, expose `aria-busy`, announce
  running/error/success transitions, and retain a retry path where replay is
  safe.
- Presentation boundaries normalize wire aliases to the fixed Queued /
  Running / Completed / Failed / Cancelled vocabulary and status ramp.

## Motion

Stage 6 motion conformance:

- `--motion-fast/base/slow` = 120/150/180ms with
  `cubic-bezier(0.2, 0, 0, 1)`. Tailwind's default transition duration and
  easing resolve through those product tokens.
- `mol-motion-overlay`, `mol-motion-dialog`, `mol-motion-popup`, and
  `mol-motion-sheet` own Radix open/close motion across dialog, alert, sheet,
  select, popover, tooltip, dropdown, and context-menu primitives.
- Spatial motion is reserved for edge-owned panels: the mobile sheet enters
  from its declared edge and the desktop inspector enters from the right.
- `mol-motion-progress-spin` and `mol-motion-progress-pulse` are the only
  continuous chrome animations. They use linear cadence and appear only for
  running, loading, skeleton, or streaming feedback; idle/decorative loops are
  forbidden.
- Workflow port hover/link transitions override Flowgram's injected 200ms
  `transition: all` with explicit 150ms product transitions. Programmatic
  zoom/fit easing reacts to the same motion preference.
- Global `prefers-reduced-motion: reduce` removes chrome animations and
  transitions entirely. Workflow running edges and injected port transitions
  have explicit fallbacks in addition to the global rule.

## OpenAPI client

Regenerate with:

```bash
.venv/bin/python scripts/dump_openapi.py
npm run generate:api   # root; includes patch-generated-api.mjs for JSONValue
```

`PlanDetailResponse` tracks PlanOrchestrator fields from the live FastAPI
schema. Do not hand-edit `apps/web/src/api/generated/`.

## Base primitives installed

`button`, `input`, `textarea`, `select`, `badge`, `tabs`, `dialog`,
`alert-dialog`, `sheet`, `popover`, `dropdown-menu`, `context-menu`,
`tooltip`, `separator`, `scroll-area`, `resizable`, `command`, `skeleton`,
`accordion`, `collapsible`, `checkbox`, `slider`, `table`, `code`, `label`,
plus workbench needs `tree`, `markdown`, `toast`, `thinking-block`,
`tool-call-row`. **`card` removed** (zero feature callers).

## Permitted variance claimed

| Axis | This product | Rationale |
|---|---|---|
| Default theme | Light, with dark available | Parameters, tables, DAGs, logs are read for hours |
| Accent hue | 295 violet | Outside status-running band; distinct from MolVis teal |
| Layout topology | Nav / work surface / inspector (+ plan rail) | IDE-shaped scientific workbench |
| Information density | High, persistently visible chrome | Edit · configure · run is the dominant task |
| Panel behavior | Fixed resizable side panels; mobile edge drawers | State survives layout changes |
| Product component set | Local to `apps/web/` | Never import MolVis page components |

## Known debt

2026-07-31 hygiene pass fixed: breadcrumb wrap inside the 40px header,
dense-table row heights (DataTable now ~30px via `ROW_PADDING_DENSE`),
dead/non-conforming `density.ts` constants (now the live geometry source),
headline numbers on `text-display` / `text-title` tokens, and redundant
`md:px-4` no-ops. Gate pass (same run): molvis preview consumers moved from
`@molcrafts/molvis-core` to `@molcrafts/molvis-stage` (the package that owns
`mountMolvis` / `./io`), duplicate `inlineStructure` casing copy removed,
`chat-answer.test` moved to `@rstest/core`, motion-contract violations
(`animate-spin` → `mol-motion-progress-spin`) fixed, conditional hook in
AgentViewer removed, and `biome.json` migrated to CLI 2.5.0. Sticky table
headers are now opaque (`bg-background`).

| Item | Stage | Severity |
|---|---|---|
| Two-line Jobs-table rows (~52px) sit above the 28–32px single-line target by design | density | 🟢 |
| Dashboard-panel drag/remove chrome is hover-only (HTML5 DnD, no touch path) | states | 🟡 |
| `apps/web/src/app/renderers/agent/inlineStructure.tsx` tail was reconstructed after the casing cleanup deleted the only copy (macOS case-insensitive FS); review the render block | — | 🟡 |

New feature chrome should use the workbench product components rather than
importing base action or tag primitives directly.

<!-- mol:ui:end -->

## Hierarchy nav & information design (2026-08-09)

Canonical content rules for entity surfaces live in the mol plugin:

`skills/ui/references/information-design.md`

(loaded by `/mol:ui` stage `info` and by the `web-design` agent). This
section records **MolExp-only** field homes and nav labels; do not fork
the shared constitution here.

### Navigator label

- Left-rail view `projects` is labeled **Projects** (not Experiments). The tree
  top level is Project → Experiment → Run; calling the rail "Experiments"
  misnamed the middle tier as the section root.

### Overview = dashboard, not inventory

Overview tabs are **posture dashboards**, never a second inventory of the
children that live on Experiments / Runs / Assets tabs.

- Shell: `OverviewSurface` → `DashboardCanvas` (padded, max-width) —
  air, not edge-to-edge lists.
- Project: status donut + aggregate metrics.
- Experiment: richer posture (status, duration, latest run, tasks) +
  parameter **shadcn Table** + embedded **`WorkflowGraphViewer`** (same
  component as the Workflow tab — not a text stub).
- No “Needs attention” / action queues / mini entity lists on Overview.
- Inventories: Experiments, Runs, Assets tabs own full-height `DataTable`.
- Run overview: padded canvas + shadcn tables for params / results.

### Entity tabs (MolVis-aligned)

`EntityTabBar` matches MolVis `PanelTabStrip` **topology**: full-width band,
equal flex columns, line underline + accent text for active. Labels are
**text only** (no glyph strip) — MolExp is a workbench of named surfaces.

Inventory tabs (Experiments / Runs / Assets / Executions / …) use
`InventoryCanvas` (same air as Overview) + shadcn / `DataTable` — not card
lists. Backend-specific run tabs (e.g. **Molq**) register via
`registerEntityTabContribution` with optional `matches`; never hard-code
“Scheduler” in core viewers.

### Fact ownership (MolExp chrome)

| Fact | Home | Not on center overview |
|---|---|---|
| Tree position / name | Left nav + breadcrumb | Identity card |
| Lineage (project / experiment / workflow / plan) | Right inspector `RelatedPanel` → **Lineage** (`LINEAGE_RELATIONS`) | Parent crumbs, Related cards |
| Scalar ids, config_hash, paths | Inspector **Details** (config may appear once truncated+copyable on Run MetaStrip) | Full identity wall |
| Child status rollup | One `StatusInline` | Parallel StatCard grid of the same counts |
| Live sync / toasts | Bottom status bar | Center toast stack |

### Hierarchy priority (primary inventory)

| Level | Primary fold content | Pattern notes |
|---|---|---|
| **Project** | Experiments table: name, run count, status distribution, workflow task count, updated | Portfolio `StatusInline`; counts in MetaStrip only if decision-bearing |
| **Experiment** | Runs list/table: status, **varying** param preview, result preview, duration | Fixed params once (strip / constants), never repeated every row; full Runs tab for complete inventory |
| **Run** | Error banner (if any) + Parameters \| Results property grids | MetaStrip for started/finished/duration/backend/attempts/assets; lineage only in inspector |

### Scientific layers

- **Varying parameters** → Experiment run columns (keys that differ across the set).
- **Fixed parameters** → once above the table or inspector.
- **Status** → row glyph + StatusInline; never brand accent as status.
- **Results** → short mono preview on experiment list; full grid on Run.
- **Artifacts** → count in MetaStrip; list on Assets / bottom Artifacts.

### Agent procedure (before new overview UI)

1. Name level (Project / Experiment / Run).
2. Name page job (overview | detail tab | inspector | bottom).
3. List ≤3 user questions.
4. Assign every field to a chrome home (table above); demote the rest.
5. Pick one primary structure (table / MetaStrip / property list) — never default to Card.
6. Compose with the overview skeleton; audit for fact echo and vanity KPIs.

`/mol:ui info` is the stage that restructures content against this contract
without inventing a new visual language. Visual stages remain tokens →
components → de-card → states → motion.

### Info experiment applied (2026-08-09)

| Surface | Change |
|---|---|
| Project overview | Dropped MetaStrip `id` + experiments count (table is inventory); kept state/updated/runs/success/assets |
| Experiment overview | Dropped identity id, KPI `runs` count, Parameters wall, mini Workflow graph; fixed params as one constants strip; run rows prefer **varying** keys; workflow via strip trailing + Workflow tab |
| Run overview | Hide zero-noise MetaStrip fields (default profile, single attempt, zero assets) |
| Data | `varyingAxes` / `fixedAxes` on `ExperimentWorkbenchData` |
