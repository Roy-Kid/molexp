---
name: molexp-api
description: Develop a new API endpoint or modify an existing one — route, schema, client regen, mock handler.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
argument-hint: <endpoint description>
---

Develop API endpoint: $ARGUMENTS

## Steps

1. **Read conventions**: Check existing routes in `src/molexp/server/routes/` and schemas in `src/molexp/server/schemas/`.

2. **Define schemas** in `src/molexp/server/schemas/requests.py` and `responses.py`:
   - Request: `BaseModel` with strict types
   - Response: Use `EntityMixin` (id) and `TimestampMixin` (created_at) where appropriate

3. **Implement route** in `src/molexp/server/routes/<module>.py`:
   - Use `Depends(get_workspace)` for workspace access
   - Keep handlers thin — delegate to workspace/workflow/agent layer
   - Register router in `routes/__init__.py`

4. **Write tests** in `tests/server/`:
   - Route with `TestClient`
   - Request validation (invalid inputs → 400)
   - Response schema compliance
   - Error cases (404, 500)

5. **Regenerate frontend client**:
   ```bash
   cd ui && npm run generate:api
   ```

6. **Add MSW mock** in `ui/mocks/handlers/`:
   - Handler matching the endpoint
   - Register in `ui/mocks/handlers/index.ts`

7. **Verify**: `pytest tests/server/` and `cd ui && npx tsc --noEmit`

## Conventions

- All routes under `/api` prefix
- Plural nouns for collections (`/api/projects`)
- Pydantic schemas for all request/response bodies
- SSE for streaming (see `routes/agent.py` pattern)
