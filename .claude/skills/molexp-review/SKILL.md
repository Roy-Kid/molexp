---
name: molexp-review
description: Comprehensive code review aggregating architecture, performance, documentation, and async safety checks. Use after writing code or during PR review.
argument-hint: "[path or module]"
user-invocable: true
---

Review code for: $ARGUMENTS

If no path given, review all files modified in `git diff --name-only HEAD`.

**Invoke all dimensions in parallel:**

1. **Architecture** → invoke `/molexp-spec` validation checks (layer compliance L1-L5)
2. **Performance** → check async/I/O anti-patterns
3. **Documentation** → check docstring completeness
4. **Async & System Safety**:
   - No blocking calls in async context (sync I/O, time.sleep)
   - Concurrency: proper locking, no race conditions on shared state
   - Caching correctness: cache invalidation on mutation
   - Atomic persistence: temp-file + os.rename pattern
   - N+1 patterns: batch queries instead of per-item
   - WebSocket: proper cleanup on disconnect
   - Private modules: no imports from `_pydantic_graph/` or `_pydantic_ai/`
   - Generated code: `ui/src/api/generated/` never manually edited
5. **Code Quality** (inline):
   - Functions < 50 lines, files < 800 lines
   - No deep nesting (> 4 levels)
   - No hardcoded magic numbers
   - Type annotations on all public APIs
   - Google-style docstrings (Python), JSDoc (TypeScript)
   - Module = Feature organization
6. **Immutability** (inline):
   - No mutation of input objects
   - Pydantic models use model_copy() not direct assignment
   - New dicts/lists for transformed data

**Severity levels**:
- CRITICAL — must fix (architecture violations, async safety)
- HIGH — should fix (missing tests, performance issues)
- MEDIUM — fix when possible (style, documentation gaps)
- LOW — nice to have

**Output**: Merged report:
```
CODE REVIEW: <path>
ARCHITECTURE: ✅/❌ per check
PERFORMANCE: ✅/⚠️ per check
DOCUMENTATION: ✅/⚠️ per check
ASYNC & SYSTEM SAFETY: ✅/❌ per check
CODE QUALITY: ✅/⚠️ per check
IMMUTABILITY: ✅/❌ per check
SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
```
