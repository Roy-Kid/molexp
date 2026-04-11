---
name: molexp-spec
description: Convert a natural language requirement into a structured technical spec for molexp.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Agent
argument-hint: <natural language requirement>
---

Convert this requirement into a technical spec: $ARGUMENTS

## Process

1. **Analyze**: Read relevant source files to understand current state. Identify which layers are involved (workspace, workflow, agent, server, UI).

2. **Generate spec** in this format:

```markdown
# Spec: <Feature Name>

## Summary
One paragraph: what and why.

## Affected Layers
- [ ] workspace — data models, persistence
- [ ] workflow — steps, runtime, caching
- [ ] agent — tools, service, policy
- [ ] server — routes, schemas
- [ ] UI — renderers, state, components

## Data Models
New/modified Pydantic models with fields and types.

## API Changes
| Method | Path | Request | Response | Breaking? |

## Workflow Changes
New steps/actors, modified specs, dependency changes.

## UI Changes
New renderers, state changes, mock handlers.

## Test Plan
Per-module unit tests + cross-layer integration tests.

## Open Questions
Decisions that need user input.
```

3. **Validate**: Check for layer violations, Module=Feature compliance, backwards compatibility.

4. **Save**: On approval, write to `docs/developer/<feature-name>-spec.md`.
