# tests/test_docs — the documentation drift gate

Every fenced ```` ```python ```` block in `docs/**/*.md` is executed as a
pytest case (`test_doc_code_blocks.py`), and every example script is
subprocess-gated (`tests/test_examples_smoke.py`, completeness enforced by
`test_examples_coverage.py`). If a documented snippet stops running against
the shipped API, CI fails — the docs cannot silently rot.

## Conventions for doc authors

1. **Blocks on one page share a namespace and a working directory** and run
   in page order. Tutorial pages may build state progressively: a class
   defined in an earlier block is visible to later blocks on the same page.
   Files a block writes with relative paths (e.g. `./lab`) land in a
   page-private temp dir, never in the repo.

2. **`# docs: skip — <reason>`** as the block's *first line* excludes it from
   execution. Use it for:
   - pseudo-code / signature overviews (`...` placeholders, undefined names),
   - blocks that need an LLM API key or the network,
   - blocks that start servers, daemons, or scheduler jobs.
   The reason is mandatory and shows up in the pytest skip report.

3. **`# docs: xfail — <reason>`** runs the block but expects failure until
   the named in-flight workflow lands (non-strict: it flips green silently).
   The reason must say *what* it is waiting for.

4. **Top-level `await` is fine.** Blocks compile with
   `PyCF_ALLOW_TOP_LEVEL_AWAIT` and run under `asyncio.run`, so docs can show
   idiomatic async driver code without `asyncio.run(...)` boilerplate.

5. **Only `python` blocks are collected** (` ```python `, ` ```py `,
   ` ```python3 `). `bash` / `json` / `yaml` / `pycon` blocks are prose.

6. **Pages being rewritten by another workflow** can be listed in
   `PENDING_PAGES` in `test_doc_code_blocks.py` (page → reason); their blocks
   run as non-strict xfail until the rewrite lands.

## Ordering caveat

Because a page's blocks share state, this suite relies on pytest's default
collection order — do not shard `test_doc_code_blocks.py` with `pytest-xdist`
or randomize its test order.

## Adding an example script

New `examples/**/*.py` files must be wired into `tests/test_examples_smoke.py`
(standalone or CLI-profile) or excused with a reason in
`tests/test_docs/test_examples_coverage.py::EXCUSED` — the coverage test fails
otherwise.
