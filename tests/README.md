# Test suite contract — orthogonal by construction

Pruned to a critical core on 2026-07-04 (owner directive). Every new test must
obey this contract; a PR that violates it gets the test deleted, not merged.

## The one rule

**A behavior is tested exactly once, in the layer that owns it.**

The layer DAG (`CLAUDE.md`) decides ownership: run semantics → `test_workspace`,
graph execution → `test_workflow`, LLM loops/sessions → `test_agent`, pipeline
orchestration → `test_harness`, shared application verbs → `test_services`.
`test_server` and `test_cli` test **shells only**: route registration, status-code
domains, request validation/aliases, wire shapes, flag parsing, exit codes —
never domain outcomes (those are already owned below; CLI and server share one
services code path by law, so re-asserting the outcome twice tests nothing).

## Always keep

- Architectural invariant locks: import-guard AST scans, public-surface locks,
  engine-boundary scan, on-disk layout naming law.
- One test per verb/state transition of a core contract (run/resume/rerun
  domains, gate grant/reject/suspend, cache identity, seed validation).
- One regression test per real production bug, docstring citing the bug.

## Never add

- Trivial assertions: field defaults, `__repr__`, `X in __all__` (the surface
  lock owns that), "constructor stores its args", copy/docstring text.
- Cosmetic input permutations of one code path — keep the strongest case plus
  the boundary; parametrize only when the table is the point.
- Re-tests of dependencies (pydantic validation mechanics, pydantic-ai retries,
  stdlib behavior) or of lower layers from upper suites.
- Timing/threshold tests (flaky by construction; performance belongs in a
  dedicated bench repo, not this suite).
- More than one test per deprecation alias.
- UI source-contract tests that assert component source as text — test pure
  logic modules instead; extract logic out of components to make it testable.

## Environment rules

- CLI output assertions go through `click.utils.strip_ansi` — assert the text,
  never the styling (rich styles help output on CI terminals).
- No test may depend on ambient env (color vars, terminal width, wall clock
  thresholds) or on a running server.
