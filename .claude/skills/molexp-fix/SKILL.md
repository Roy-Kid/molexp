---
name: molexp-fix
description: Minimal-diff bug fix — reads, diagnoses, patches the smallest surface that eliminates the defect. Writes code only.
argument-hint: "<symptom or failing test>"
user-invocable: true
---

# molexp-fix

Read CLAUDE.md for molexp conventions before touching any file.

## Procedure

1. **Reproduce** — locate the failing test or repro path from the argument. Run it.
2. **Isolate** — read the stack trace or error; identify the single smallest change that fixes it. Do not refactor surrounding code.
3. **Patch** — apply the minimal edit. One concern per edit.
4. **Verify** — re-run the failing test and the full test suite for the affected module (`pytest tests/<module>/`).
5. **Done** — report: file patched, lines changed, test result.

## Constraints

- Do NOT refactor, rename, or restructure beyond the fix.
- Do NOT add new abstractions or helpers unless they are the fix itself.
- If the root cause requires a larger change, stop and tell the user — do not silently expand scope.

## Output

`Fixed <file>:<lines> — <one-line description of the defect and fix>. Tests: <pass/fail>.`
