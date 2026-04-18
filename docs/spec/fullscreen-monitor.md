# Spec: Full-screen Run Monitor for Molexp, Powered by Molq Panels

## Goal

Design a full-screen terminal monitor for `molexp` that observes the status of local and remote runs.

The UX target is `htop` / `btop`-like: not a plain printed table, but a **full-terminal dashboard** with a top-level overview, progress visualization, a bottom-section job list, and basic open / close interactivity.

This design explicitly scopes:

- How the interface is organized
- What the `molq` plugin is responsible for
- What `molexp` is responsible for
- How the two cooperate
- Which interaction requirements are mandatory

This spec defines **product and architectural requirements only**. It does not prescribe specific code structures, class names, or implementation details.

---

## Design Principles

### 1. Full-screen monitor, not inline table

The current small-panel + single-table form is not acceptable. The monitor must be a complete terminal view; its visuals and interactions should feel like a system monitor, not a one-shot CLI dump.

### 2. Overview-first, list-second

The most important information is not a single-line job table but the overall state. The interface must prioritize:

- Is something running right now?
- Overall progress
- Running / pending / done / failed totals
- Current workflow / run overall status

The job list is a bottom-half element, not the focal point of the interface.

### 3. Single-job and multi-job both look natural

Even with just one job, the layout must not degrade into an awkward one-row table. The same full-screen monitor must gracefully handle single jobs, multi-job sweeps, local runs, and remote scheduler-backed runs.

### 4. Status-first, table-second

Status must be the primary visual anchor. Users should see running / pending / done / failed at a glance before they notice run ID or scheduler ID.

### 5. Monitor is a viewer, not the run itself

Closing the monitor must not terminate the run. The monitor is an observation surface, not the job's lifeline.

---

## UI Requirements

### Overall Layout

The monitor occupies the whole terminal and contains the following regions:

```
┌─────────────────────────────────────────────────────────┐
│  A. Header / Title Area                                 │
│  workflow name · overall status · last updated          │
├───────────────────────────────┬─────────────────────────┤
│  B. Overview Area             │                         │
│  total · running · pending    │   progress bar / vis    │
│  done · failed                │                         │
├───────────────────────────────┴─────────────────────────┤
│  C. Job List Area                                       │
│  state · run identity · scheduler id · elapsed ...      │
│  ...                                                    │
├─────────────────────────────────────────────────────────┤
│  D. Footer / Hint Area        q: quit  r: refresh  ...  │
└─────────────────────────────────────────────────────────┘
```

### A. Header / Title Area

Identity and overall state of the monitor. Minimum contents:

- Identifier of the current workflow / run / experiment
- Current overall status (running / finished / failed / mixed)
- Last update / refresh time

### B. Overview Area

The interface's core, top half. Minimum contents:

- Total job count
- Running / pending / done / failed counts
- Overall progress (must include an explicit progress visualization, e.g. a progress bar)

Progress cannot be weakly expressed as "0/1 done"; an overview-level visual is required.

### C. Job List Area

Bottom half, showing each job / run's status. Each entry must include at least:

- state
- run identity
- scheduler id (if applicable)
- additional secondary metadata (elapsed, node, message, …) at the agent's discretion

The list is for rapid status scanning, not a database-style wide table.

### D. Footer / Hint Area

A concise interaction-hint region at the bottom. Minimum hints:

- `q` to close the monitor
- Additional interaction hints at the agent's discretion

---

## Visual Requirements

### 1. Avoid heavy boxed table aesthetics

"A panel wrapping a box-heavy table" is not acceptable. Minimize decorative borders; emphasize regions, status, and progress.

### 2. Strong visual status hierarchy

Status must be visually distinctive. Color, icons, position, and text must all support rapid recognition.

### 3. Scheduler ID is secondary

`scheduler id` is not top-priority information; it must not outweigh status or run identity.

### 4. Progress must be visually central

The overview must contain an explicit progress visualization. This is not an optional enhancement — it is part of the monitor's core.

---

## Interaction Requirements

### 1. `q` closes the monitor

Pressing `q` closes the full-screen monitor and returns the user to the normal CLI.

### 2. Closing the monitor must not cancel jobs

`q` only closes the viewer / quits the monitor / returns to the normal CLI. It must never terminate remote jobs, local workflows, or scheduler jobs.

### 3. Monitor must be reopenable

After closing, the user must be able to reopen the monitor via an explicit command. The command name is not mandated, but the monitor is not single-use; re-entry must be supported.

### 4. Run and watch commands should cooperate naturally

If `run` enters the monitor automatically after submission, exiting the monitor should return to the normal CLI and clearly note:

- Jobs are still running
- The user can reopen the monitor

---

## Molexp vs Molq Responsibilities

### Molexp responsibilities

`molexp` owns workflow / run orchestration and is responsible for:

- Deciding when to enter and leave the monitor
- Opening / closing / reopening the monitor
- Managing run / workflow / experiment lifecycle
- Collecting and normalizing status data
- Translating run state into the generic status input the monitor consumes
- Deciding how the CLI flow switches between monitor and normal output

**`molexp` owns the monitor's lifecycle.**

### Molq responsibilities

The `molq` plugin is responsible for the monitor's **UI capability**, not the workflow lifecycle:

- Providing full-screen panel / dashboard capability
- Providing the overview and job-list regions
- Providing interaction semantics of the monitor
- Offering a reusable terminal-viewer experience

`molq` does not drive the job lifecycle, does not decide whether a run finishes or continues submitting, and does not own workflow semantics.

### Interaction contract between Molexp and Molq

| Responsibility | Owner |
|----------------|-------|
| When the monitor is created, closed, re-entered | molexp |
| How the monitor is rendered; basic interaction UI semantics | molq |
| Whether a workflow finishes / keeps submitting | molexp |
| Rich layout details, panel components | molq |
| Translating UI actions into CLI / lifecycle operations | molexp |

Rules:

1. **Molexp owns control flow.** `molexp` drives the monitor's lifecycle.
2. **Molq owns presentation semantics.** `molq` dictates how the monitor is presented.
3. **Molq must not own workflow lifecycle.** `molq` does not decide "stop the job" or "terminate the workflow" on its own.
4. **Molexp must not reimplement panel logic.** `molexp` must not hand-roll every Rich layout, or `molq` loses its point as a panel/plugin layer.
5. **UI actions must be interpretable by Molexp.** Interaction results inside the monitor must bubble back to `molexp` for CLI / lifecycle action.

---

## Architectural Constraints

1. **No giant monolithic CLI table renderer** — do not fold every concern into a single CLI-output function.
2. **No inversion of control from Molq into Molexp lifecycle** — `molq` must not drive `molexp`'s main loop.
3. **No scheduler-specific UI hardcoding** — the UI must not be locked to slurm/pbs/lsf specifics; design around run/job state.
4. **UI design should scale from local to remote** — one monitor must serve local, remote, and scheduler-backed runs alike.

---

## Agent Tasks

The agent must, based on the current codebase, design and implement:

1. A full-screen terminal monitor's information architecture
2. A layout that works naturally for both single-job and multi-job cases
3. The responsibility boundary and interface contract between `molexp` and `molq`
4. The open / close / reopen interaction flow
5. A state model that can support future remote backends

At the agent's discretion:

- Specific layout components
- Specific Rich usage (layout / panel / live / screen, …)
- Specific keybindings
- Specific state-object / panel-object / plugin-object designs

---

## Acceptance Criteria

| Requirement | Category |
|-------------|----------|
| Monitor is a full-screen dashboard, not a small panel | UI |
| Top half has a clear overview and progress visualization | UI |
| Bottom half has a job list | UI |
| Status outweighs table columns; visually prioritized | Visual |
| Single-job case still looks natural; no degradation | UI |
| `q` only closes the monitor; does not cancel runs | Interaction |
| Monitor is reopenable | Interaction |
| `molq` owns panel/view capability | Architecture |
| `molexp` owns monitor lifecycle and CLI switching | Architecture |
| Responsibilities are clear; neither side swallows the other | Architecture |

---

## Non-goals

This spec does **not** require:

- Specific class designs or function signatures
- Specific keyboard-event implementation
- Specific scheduler adapter code
- Specific Rich API usage
- A full TUI framework abstraction

Those are for the agent to decide based on the codebase.
