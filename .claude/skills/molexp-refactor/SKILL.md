---
name: molexp-refactor
description: Restructure code preserving all invariants — delegates architecture compliance to molexp-architect. Writes code.
argument-hint: "<what to restructure and why>"
user-invocable: true
---

# molexp-refactor

Read CLAUDE.md for molexp layer conventions before touching any file.

## Procedure

1. **Scope** — identify all files in the refactor surface. State what changes and what is preserved.
2. **Architecture pre-check** — delegate to `molexp-architect` agent: confirm the proposed structure is layer-compliant before writing.
3. **Test baseline** — run `pytest tests/<module>/` and record the pass count. This is the invariant floor.
4. **Refactor** — apply the structural change. No behavior changes, no new features.
5. **Architecture post-check** — delegate to `molexp-architect` agent again on the result.
6. **Verify** — re-run tests. Pass count must be ≥ baseline. If it drops, revert and report.
7. **Done** — report files changed, lines delta, test result.

## Constraints

- Preserve all public interfaces unless the refactor explicitly changes them (state this upfront).
- Do not fix bugs or add features during a refactor — separate concerns.
- Architecture checks run at both Step 2 (before) and Step 5 (after); do not skip either.

## Output

`Refactored <N> files (+X/-Y lines). Architect: compliant. Tests: <pass_count> passed.`
