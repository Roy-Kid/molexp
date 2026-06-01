# benches

Dependency-light (stdlib-only, no `pytest-benchmark`) measurement harnesses.
Dev-only — the top-level `benches/` directory is excluded from the wheel and
sdist (src-layout ships only `src/molexp`).

## bench_provenance_lineage

```bash
python -m benches.bench_provenance_lineage
```

Builds a synthetic deep lineage of N=500 `artifact_edges` (a chain) in a
temporary directory, then measures one `SQLiteProvenanceStore.trace_backward`
from the leaf: the **edge-walk SQL statement count** (counted via
`sqlite3.Connection.set_trace_callback`) and the **wall-clock** (via
`time.perf_counter`). `get_ref` hydration hits the filesystem store, not the
SQLite connection, so it does not inflate the edge-walk count.

This is the headline evidence for `perf-hardening-02` **ac-008**: the per-node
BFS emits ~N edge-walk statements, while the recursive-CTE rewrite collapses
the edge walk to a single statement (or O(depth)). The script only prints
before/after-comparable numbers; it never asserts.
