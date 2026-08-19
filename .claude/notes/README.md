# `.claude/notes/` — passive internal context

This directory holds **passive** internal-agent context for the molexp
repository. Its contents outlive any single feature or spec.

| File | Purpose |
|---|---|
| `notes.md` | Evolving architectural decisions, captured by `/mol:note`. Stable entries are promoted to `CLAUDE.md` and pruned here. |
| `architecture.md` | Project blueprint — modules, public surfaces, layer roles. Built/refreshed by `/mol:map`; consumed by the `librarian` agent at spec time. |
| `harness-plugins.md` | Live harness discipline: plugin host, **AgentCall** atom, lifecycle, seams. Workspace / workflow / agent / science adapters are plugins. |
| `harness-goal.md` | What those plugins persist and validate (artifacts, IR, tests, audit). Plan/run are bundles on the host, not the kernel. |
| `integration.md` | How plugins cooperate — the WorkspaceContext → KnowledgeDelta loop as a bundle composition. Companion to `harness-plugins.md` + `harness-goal.md`. |
| `open-questions.md` | Uncertainties recorded over time; resolved as answers become clear. |
| `vision-gap-2026-07.md` | Snapshot of vision ↔ code gaps (July 2026); passive reference, not a live spec. |

## What does *not* belong here

- Active in-flight specs → `.claude/specs/` (alive, ticked off as `/mol:impl` works, deleted on completion)
- Public-user prose → `docs/`
- Skill / agent definitions, hooks, settings → `.claude/` (runtime config)

The split between `.claude/notes/` and `.claude/specs/` is **passive vs
active**: notes are kept; specs are intentionally ephemeral.

## Adding new content

Add subdirectories only when there is real content to seed them:

- `decisions/` — substantial architectural history beyond `notes.md`
- `contracts/` — agent-handoff contracts
- `rubrics/` — review checklists worth encoding
- `debt/` — tracked technical debt
- `handoffs/` — work regularly paused mid-flight

Empty directories are not value.
