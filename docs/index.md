# MolExp Documentation

Welcome to MolExp! A tiny yet fully-typed task-graph framework built on top of Pydantic, designed for scientific computing workflows. MolExp provides a pure functional task abstraction, a static compiler that produces deterministic graph orders, a runtime engine, and a tiny DSL for common data-flow patterns.

## Why MolExp?

MolExp's design philosophy is "minimal yet complete". We believe a good framework should make every component's role clear and visible, without hidden magic. Therefore, MolExp's codebase stays lean, with each layer having explicit responsibilities, allowing you to fully understand how workflows are compiled and executed.

MolExp is particularly suited for scientific computing scenarios that require reproducibility, traceability, and extensibility. Whether it's molecular dynamics simulations, machine learning training, or data analysis pipelines, MolExp helps you organize code, manage dependencies, and track results.

MolExp exposes a protocol interface for in-process compatibility, but persisted workflows require `Task` classes with Pydantic configurations and explicit registration. This keeps workflows deterministic and replayable.

## Core Architecture

MolExp's architecture consists of three core layers:

<div class="grid cards" markdown>

-   :material-code-tags: **Task Abstraction**

    ---

    **Pure Functional Task Definition**
    
    Each task is an independent computation unit with type-safe configuration via Pydantic models. Persisted workflows use explicit registration and deterministic task IDs.
    
    [:octicons-arrow-right-24: Learn about Tasks](core/task.md)

-   :material-compile: **Compiler Layer**

    ---

    **Static Graph Compilation**
    
    The compiler statically analyzes task graphs, generates deterministic execution orders, and detects circular dependencies.
    
    [:octicons-arrow-right-24: Developer Docs](developer/ir-and-compiler.md)

-   :material-play: **Execution Engine**

    ---

    **Parallel Execution & Failure Propagation**
    
    The engine supports parallel execution of independent tasks, automatic failure propagation, and execution hooks for monitoring.
    
    [:octicons-arrow-right-24: Learn about Engine](core/engine.md)

</div>

## Workspace Architecture

MolExp provides a complete Project-Experiment-Run three-tier architecture to help you organize and manage scientific computing workflows:

<div class="grid cards" markdown>

-   :material-folder: **Project**

    ---

    **Top-level container for research areas**
    
    A project represents a research domain or topic, containing multiple experiments.
    
    [:octicons-arrow-right-24: Learn about Workspace](workspace/architecture.md)

-   :material-flask: **Experiment**

    ---

    **Repeatable workflow definitions**
    
    Experiments define repeatable workflow templates with parameter space definitions.
    
    [:octicons-arrow-right-24: Learn about Workspace](workspace/architecture.md)

-   :material-play-circle: **Run**

    ---

    **Single execution instance**
    
    A run is a concrete execution of an experiment, containing complete reproducibility information.
    
    [:octicons-arrow-right-24: Learn about Workspace](workspace/architecture.md)

</div>

## Quick Start

If you're new to MolExp, we recommend starting with the quick start guide, which will walk you through creating and executing your first workflow.

[:octicons-arrow-right-24: Get Started](get-started/quick-start.md)

## License

MIT License - see [LICENSE](https://github.com/molcrafts/molexp/blob/main/LICENSE) for details.
