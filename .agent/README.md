# `.agent/` — passive internal context

This directory holds **passive** internal-agent context for the
molexp repository. Its contents outlive any single feature or spec.

| File | Purpose |
|---|---|
| `notes.md` | Evolving decisions, captured by `/mol:note`. Stable entries are promoted to `CLAUDE.md` and pruned here. |
| `project-map.md` | Observed top-level layout and module boundaries — a quick orientation aid. |
| `open-questions.md` | Bootstrap-time uncertainties the user fills in over time. |

## What does *not* belong here

- Active in-flight specs → `.claude/specs/` (alive, ticked off as
  `/mol:impl` works, deleted on completion)
- Public-user prose → `docs/`
- Skill / agent definitions, hooks, settings → `.claude/`

The split between `.agent/` and `.claude/` is **passive vs active**.
Notes are kept; specs are intentionally ephemeral.

## Adding new content

Add subdirectories only when there is real content to seed them:

- `.agent/decisions/` if substantial architectural history needs
  preserving beyond `notes.md`
- `.agent/contracts/` if there are agent-handoff contracts to record
- `.agent/rubrics/` if there are review checklists worth encoding
- `.agent/debt/` if there is technical debt the user wants tracked
- `.agent/handoffs/` if work is regularly paused mid-flight

Empty directories are not value.
