---
name: molexp-debug
description: Diagnose-only investigation — reads code and logs, produces a root-cause report. Never edits files.
argument-hint: "<symptom, error message, or failing test>"
user-invocable: true
---

# molexp-debug

Read CLAUDE.md for molexp conventions. **This skill never edits files.**

## Procedure

1. **Gather signals** — read error message, stack trace, or test output from the argument.
2. **Trace callpath** — grep and read the relevant source; follow the call chain from entry point to failure.
3. **Identify invariant** — state which invariant, contract, or assumption is violated and where.
4. **Hypothesize** — propose the most likely root cause with evidence (file:line references).
5. **Distinguish** — separate root cause from symptoms; list any secondary effects.
6. **Recommend** — suggest the fix strategy without implementing it.

## Hard Rules

- **Never** call Edit, Write, or Bash commands that mutate files.
- If fixing requires understanding is wrong, say so — do not guess.
- Read-only tools only: Read, Grep, Glob, Bash (read-only commands).

## Output

```
Root cause: <one sentence>
Evidence:   <file>:<line> — <what it shows>
            <file>:<line> — <what it shows>
Fix strategy: <one paragraph — no code>
```
