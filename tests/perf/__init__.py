"""Dependency-light performance probes for the molexp harness.

These tests measure event-loop responsiveness with ``time.perf_counter`` +
``asyncio`` stdlib only (no pytest-benchmark). They are regression guards
for the perf-hardening chain — e.g. perf-hardening-01 offloads blocking
SQLite + file I/O off the harness event loop, and the probe here asserts a
co-running heartbeat coroutine is not starved during a multi-stage run.

Tests are marked ``@pytest.mark.perf``.
"""
