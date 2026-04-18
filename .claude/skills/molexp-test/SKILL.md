---
name: molexp-test
description: User-invoked TDD workflow to write or improve tests for molexp modules with coverage analysis. Orchestrates the process; delegates test authoring to the molexp-tester agent when useful.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
argument-hint: <module or feature to test>
---

Write tests for: $ARGUMENTS

## Steps

1. **Identify scope**: Read source code and existing tests. Check `conftest.py` for fixtures.

2. **Organize**: Tests mirror source:
   ```
   tests/agent/     → src/molexp/agent/
   tests/server/    → src/molexp/server/
   tests/workflow/  → src/molexp/workflow/
   tests/workspace/ → src/molexp/workspace/
   ```
   Shared fixtures in `tests/<module>/conftest.py`.

3. **TDD cycle** for each function/class:
   - Write test → FAIL (RED)
   - Implement → PASS (GREEN)
   - Refactor (IMPROVE)

4. **Backend patterns**:
   ```python
   # Workspace: use tmp_path, not mocks
   def test_run_persists(tmp_path):
       ws = Workspace(root=tmp_path)
       ws.materialize()

   # Server: use TestClient
   def test_endpoint(client: TestClient):
       resp = client.post("/api/projects", json={"name": "test"})
       assert resp.status_code == 201

   # Workflow: test step in isolation, then in spec
   async def test_step():
       result = await step.execute(ctx)
   ```

5. **Frontend**: Vitest, fixtures in `ui/src/__fixtures__/`.

6. **Coverage**: `pytest tests/<module>/ --cov=src/molexp/<module> --cov-report=term-missing` — target 80%+.

## Rules

- Don't mock filesystem in integration tests — use `tmp_path`
- Don't test `_pydantic_graph/` internals — test through public API
- Don't duplicate fixtures — use `conftest.py`
- Tests must be order-independent
