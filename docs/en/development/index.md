# Development

Contributor-facing docs for working on `molexp` internals.

## Internals

- [Compiler](compiler.md) — DSL → `CompiledWorkflow` → `ExecutionPlan` lowering + structural engine, identity, caching
- [Task Protocols](task-protocols.md) — `Runnable` / `Streamable` structural contracts

## Documentation code blocks run in CI

Every ```` ```python ```` block in `docs/**/*.md` is executed as a pytest case
(`tests/test_docs/test_doc_code_blocks.py`), and every example script under
`examples/` is subprocess-gated (`tests/test_examples_smoke.py`). When writing
docs:

- Blocks on one page share a namespace and a temp working directory and run in
  page order — tutorial pages may build state progressively.
- Mark non-runnable blocks with a first-line `# docs: skip — <reason>` comment
  (pseudo-code, LLM-key/network-dependent, or server/daemon-starting snippets).
- Mark blocks waiting on an in-flight feature with `# docs: xfail — <reason>`.
- Top-level `await` is allowed — blocks compile with
  `PyCF_ALLOW_TOP_LEVEL_AWAIT`.

The full convention lives in `tests/test_docs/README.md`.
