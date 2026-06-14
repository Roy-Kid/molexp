# Product

## Register

product

## Users

Computational researchers and research teams — molecular dynamics, machine learning, computational chemistry — who run the same idea many times with different parameters, on different machines, until something works. They are technically fluent but not necessarily UI-natives. They live in this tool during long, multi-stage, multi-parameter campaigns and come back to it across days or weeks to compare runs, diagnose failures, and recover context.

The job to be done: author a workflow once, launch it many ways (locally or on a cluster via molq/SLURM/PBS), and keep the *definition* of an experiment and the *record* of every execution as one connected, auditable object — instead of folklore scattered across scripts, notebooks, and directory names.

## Product Purpose

molexp is an agent-assisted scientific-workflow platform for FAIR research, shipped as a single Python wheel (FastAPI + pydantic-graph + PydanticAI backend, React 19 SPA frontend). The UI is a three-panel, IDE-like surface over a `Workspace → Project → Experiment → Run → Execution` hierarchy:

- **Author** task-graph workflows (Flowgram canvas + Monaco Python source).
- **Launch & monitor** runs — KPI strip, status donut, activity timeline, Gantt, live-polling metric charts (tensorboard scalars).
- **Track** experiments with parameter sweeps, config hashing, and content-addressed assets (logs, checkpoints, artifacts, error traces).
- **Inspect** any run, task, or asset through a pinnable right-hand inspector without losing place.
- **Converse** with LLM agent sessions (thinking blocks, tool calls) tied to the same workspace.

Success looks like: a researcher can answer "what did I run, with what parameters, on what machine, and what came out" for any past execution — and reproduce it — without leaving the tool.

## Brand Personality

Precise and instrument-like. molexp should read like a scientific instrument or a professional IDE: calm, exact, and trustworthy. Restraint over flourish. The interface earns confidence by being legible and predictable under load, not by being decorative. Three words: **precise, trustworthy, unobtrusive.** Voice in copy is plain and specific — name the noun, describe what the thing literally does; no marketing cadence.

## Anti-references

- **Consumer/playful app.** No bubbly rounded shapes, big emoji, pastel gradients, or bouncy/elastic animation. This is work, not a toy. (Radius stays restrained — `--radius: 0.5rem` base.)
- **Cluttered enterprise tool.** No toolbar soup, no endless nested panels, no low-contrast gray-on-gray. Density is earned through hierarchy and breathing room, not by cramming.
- **Flashy / over-animated.** No decorative glassmorphism, parallax, or motion that slows real work. Motion is functional (state transitions, reveals that aid orientation) and respects reduced-motion.
- Also off-limits (cross-register): gradient text, side-stripe accent borders, hero-metric template cards, identical icon+heading+text card grids, tracked-uppercase eyebrows on every section.

## Design Principles

1. **The record is the product.** Provenance and reproducibility come first. The UI's job is to make the connection between an experiment's definition and every execution's record visible and trustworthy. (See the project's provenance-first harness goal.)
2. **Instrument, not dashboard.** Calm precision. Every pixel earns its cost; decoration that doesn't carry information is removed. The tool inspires confidence by being exact, not by being loud.
3. **Density without clutter.** Serve power users who manage 100+ runs: high information density, but always with clear hierarchy, deliberate spacing rhythm, and room to breathe. Compact ≠ cramped.
4. **Status is never ambiguous.** Run/task/execution state (running, succeeded, failed, queued) is communicated by color *and* icon/shape *and* label — never color alone. A researcher should never guess what state something is in.
5. **Get out of the way.** The UI serves the workflow. Fast feedback, pinnable context, predictable refresh and navigation. Nothing decorative should ever compete with the work in front of the researcher.

## Accessibility & Inclusion

Target **WCAG 2.1 AA.** Body text ≥4.5:1 contrast against its background (watch muted-foreground gray on tinted near-white — bump toward ink if close); large text ≥3:1. Full keyboard navigation across panels, tables, and the workflow canvas; visible focus rings (the existing `--ring` VSCode blue). Status colors must pair with icon/shape/label so the success/warning/destructive trio remains distinguishable for color-blind users. Every animation needs a `prefers-reduced-motion: reduce` alternative (crossfade or instant). Full light and dark theme parity (theme kept in sync with molvis).
