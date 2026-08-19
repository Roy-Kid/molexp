# Concepts

MolExp is easier to learn when you see it as a few connected models, not one large framework.

## The four layers

| Layer | Question it answers | Key idea |
|---|---|---|
| **Workflow** | What should happen? | A graph of tasks with named data flow |
| **Workspace** | What survives after execution? | Persistent hierarchy: Workspace → Project → Experiment → Run |
| **Agent** | How does the LLM drive this? | Natural-language intent → tool calls → real workspace mutations |
| **Assets** | Where do the data and outputs live? | Named resources at defined scopes, not undocumented paths |

These boundaries are not academic. They let the same workflow run locally during development, from the CLI, or on a cluster — without changing what the workflow means.

## The split that matters

Research code becomes untrustworthy for a boring reason: the script, the parameters, the logs, the outputs, and the "real" dataset path all drift apart. MolExp prevents that drift by making each kind of state live in an explicit place. A workflow stays a reusable graph. An experiment stays a repeatable definition. A run stays one concrete attempt. An asset stays a named resource at a defined scope.

## Reading this section

- **[Workflow](workflow.md)** — The computation model: tasks, dependencies, compilation, execution. What stays *outside* the workflow layer.
- **[Workspace](workspace.md)** — The on-disk hierarchy and the distinction between experiment (definition) and run (outcome).
- **[Agent](agent.md)** — LLM conversation layer: the pydantic-ai facade, one-shot chat or one ReAct per turn, sessions, and events.
- **[Assets and Reproducibility](assets-and-reproducibility.md)** — Why reusable data needs first-class names and scopes. What FAIR means inside a workspace.
- **[Plugins](plugins.md)** — The optional transport layer that reaches beyond local execution.
