"""Dependency-light benchmark harnesses for molexp (dev-only, not shipped).

Run a bench as a module, e.g.::

    python -m benches.bench_provenance_lineage

Benches are stdlib-only measurement harnesses (no ``pytest-benchmark``);
they print comparable before/after numbers and never assert. This package
lives at the repo top level under src-layout, so it is excluded from the
wheel and sdist (which ship only ``src/molexp``).
"""
