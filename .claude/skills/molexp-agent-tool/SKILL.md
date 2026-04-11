---
name: molexp-agent-tool
description: Develop a new agent tool for molexp's PydanticAI agent system with proper approval levels.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
argument-hint: <tool description>
---

Develop agent tool: $ARGUMENTS

## Steps

1. **Read patterns**: `src/molexp/agent/tools.py` (base class + decorator), `src/molexp/agent/_pydantic_ai/workspace_tools.py` (examples), `src/molexp/agent/policy.py` (approval).

2. **Choose style**:
   ```python
   # Decorator (simple)
   @agent_tool(level="workspace", requires_approval=False)
   async def my_tool(ctx: ToolContext, path: str) -> str: ...

   # Class (complex, stateful)
   class MyTool(Tool):
       name = "my_tool"
       level = "product"
       requires_approval = True
       async def call(self, ctx: ToolContext, **kwargs) -> dict: ...
   ```

3. **Set approval level**:
   - `workspace` — read-only / low-risk, auto-approved
   - `product` — creates/modifies data, may need approval
   - `system` — affects system, always needs approval

4. **Implement**: Tools get `ToolContext` (workspace, run, services). Return Pydantic models. Handle errors gracefully.

5. **Register**: Add to `src/molexp/agent/_pydantic_ai/workspace_tools.py` or new module in `_pydantic_ai/`. Export from `src/molexp/agent/__init__.py` if public.

6. **Test** in `tests/agent/`: tool execution, approval requirements, error handling.

7. **Verify**: `pytest tests/agent/`
