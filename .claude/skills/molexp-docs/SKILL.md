---
name: molexp-docs
description: Write or update documentation — tutorials, API docs, docstrings — by delegating to molexp-documenter. Writes docs.
argument-hint: "<module, feature, or doc type to document>"
user-invocable: true
---

# molexp-docs

Read CLAUDE.md for molexp conventions and the existing docs layout under `docs/`.

## Procedure

1. **Identify scope** — from the argument, determine: which layer(s), which public API, and which doc type (docstring / API reference / tutorial / changelog).
2. **Audit existing docs** — glob `docs/**/*.md` and the relevant source for existing coverage; identify gaps.
3. **Delegate** — hand off to `molexp-documenter` agent with: scope, gap list, doc type. For any tutorial or conceptual doc, explicitly require textbook prose style: concept → motivation → mechanics structure, headings that summarize content (not "What/Why/How" labels), prose paragraphs over bullet lists.
4. **Review output** — confirm the agent's additions are accurate against the source code (grep for public symbols cited).
5. **Write** — apply the documented output to the target files.
6. **Done** — report: files written, symbols documented.

## Output

`Documented <module>: <N> symbols / <M> pages. Files: <list>.`
