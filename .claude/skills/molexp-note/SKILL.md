---
name: molexp-note
description: Capture an evolving decision into NOTES.md, detect conflicts with existing notes or CLAUDE.md, and promote stable entries to CLAUDE.md. Writes NOTES.md and CLAUDE.md.
argument-hint: "<decision to capture, or 'promote' to sweep for stable entries>"
user-invocable: true
---

# molexp-note

Read CLAUDE.md and .claude/NOTES.md before doing anything.

## Procedure — Capture mode (default)

1. **Read context** — read `.claude/NOTES.md` and `CLAUDE.md` fully.
2. **Conflict check** — does the new decision contradict any existing note or any rule in CLAUDE.md? If yes, surface the conflict and ask the user to resolve before writing.
3. **Duplicate check** — is the decision already captured (same or equivalent)? If yes, report and stop.
4. **Write** — append to `.claude/NOTES.md` under the format:
   ```
   ## <slug> | <date> | <author>
   <decision body>
   **Status:** evolving
   ```
5. **Report** — confirm slug written.

## Procedure — Promote mode (`/molexp-note promote`)

1. Read all entries in `.claude/NOTES.md`.
2. For each entry marked `**Status:** stable`:
   - Determine the correct section in `CLAUDE.md`.
   - Append the rule to `CLAUDE.md` under that section.
   - Delete the entry from `NOTES.md`.
   - Report: `Promoted <slug> → CLAUDE.md §<section>`.
3. Report count of promoted entries and any skipped (still evolving).

## Output

Capture: `Noted <slug> in NOTES.md.`
Promote: `Promoted <N> entries. Skipped <M> (evolving).`
