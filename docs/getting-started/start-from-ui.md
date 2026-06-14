# Start from the UI

This guide is for the user who wants to create and manage experiments
from the browser, without writing Python first. You will bring up the
MolExp server, open the bundled web UI, create a project and an
experiment through dialogs, launch a run, and then monitor and manage
that run from the same screens.

Everything described here was verified against the current UI. Where a
capability is not implemented yet, it is marked **TODO** inline, and the
[feature status table](#feature-status) at the end collects all of them
in one place.

## The model in thirty seconds

The UI presents the same hierarchy the workspace stores on disk:

| Level | What it is |
|---|---|
| **Workspace** | A directory MolExp owns. One server can serve several. |
| **Project** | A container that groups related experiments. |
| **Experiment** | A workflow plus a parameter space plus an optional default compute target. |
| **Run** | One immutable execution request: concrete parameters, a status (`pending → running → succeeded / failed / cancelled`). |
| **Execution** | One physical attempt of a run (`exec-<run_id>[-N]`). A run accumulates attempts when it is resumed or rerun. |

## 1. Initialize a workspace and start the server

```bash
pip install molexp
molexp init ./lab
molexp serve --workspace ./lab --port 8000
```

Open <http://localhost:8000>. The wheel ships the production UI build,
so this one process serves both the REST API and the interface — no
Node.js required.

> **Editable installs:** if you installed from a source checkout with
> `pip install -e .`, the UI bundle may be absent and the server falls
> back to API-only. Build it once with `cd ui && npm install && npm run
> build`, or run the dev server (`npm run dev`, port 5173) alongside the
> backend. See [Server Lifecycle](../guide/server-lifecycle.md).

You can serve several workspaces at once by repeating `--workspace`;
the UI shows each one in the left panel and you switch by clicking into
its tree.

## 2. Find your way around

The activity bar on the left switches the panel between six views:

- **Experiments** — the Project → Experiment → Run tree. This is where
  you create and manage everything; most of this guide lives here.
- **Runs** — a workspace-wide dashboard: KPI strip, status mix,
  activity and Gantt charts, and a run table with facet filters
  (status, project, and friends) that persist in the URL.
- **Workflow** — all workflow definitions as cards; opening one shows
  its task graph on an editable canvas.
- **Workspace** — the raw file tree of the workspace directory, with a
  file viewer/editor and "New file / New folder" actions.
- **Asset** — the asset inventory (artifacts, logs, checkpoints) with
  lineage back to the run and task that produced each one.
- **Agent Tasks** — LLM-driven sessions; see the
  [Agent concept](../concept/agent.md) for the loop model behind them.

A command palette and a **Settings** page (remote workspaces, compute
targets) round out the shell.

## 3. Create a project

In the **Experiments** panel, click the **+** button in the header. The
*Create Project* dialog asks only for a name. The project appears in the
tree immediately, with an experiment counter on the right.

## 4. Create an experiment

Two equivalent entry points:

- right-click the project in the tree → **New experiment**, or
- open the project and click the **New Experiment** button.

The *Create Experiment* dialog takes:

| Field | Meaning |
|---|---|
| **Name** | Required. The experiment id is derived from it. |
| **Workflow** | Optional path to a workflow file. Leave blank to start with an empty canvas and draft the workflow in the UI (step 5). |
| **Parameters (JSON)** | The experiment's parameter space, e.g. `{"temperature": 300}`. Defaults inherited by every run. |
| **Default target** | Optional compute target preselected for every run. **+ Add new target…** registers one inline without leaving the dialog. |

## 5. Give the experiment a workflow

An experiment needs a workflow before a run can compute anything. Two
paths:

**Draft it on the canvas.** Open the **Workflow** panel and click **New
workflow** (pick the project, name it — this creates the backing
experiment and opens an empty canvas), or right-click an existing
experiment → **Open workflow**. The canvas is editable: add and connect
tasks, click a node to inspect it in the right panel, and save with
**⌘S / Ctrl+S**. Unsaved edits prompt before you navigate away, and
**Discard** reverts to the last saved graph.

**Author it in Python.** Define the workflow in code as shown in
[Your First Workflow](first-workflow.md) and declare it on the
experiment from a script. The UI then displays that graph read-along —
this is the usual path for non-trivial scientific workflows today.

## 6. Launch a run

From the experiment page click **Run** (or right-click the experiment →
**New run**). The *Launch Run* dialog shows the workflow (read-only),
takes the run's **Parameters (JSON)**, and lets you pick a **Target**:

- **With a compute target** — the run is created and immediately
  dispatched to the molq scheduler on that target. Status moves to
  `running` on its own.
- **No target (local)** — the run is created in `pending` state and is
  **not** executed by the server. Execute it from a terminal in the
  workspace:

  ```bash
  molexp run
  ```

  which picks up pending runs, executes them, and streams status back
  to the UI (the tree refreshes within a few seconds).

> **TODO** — *Execute locally from the UI*: there is no button that
> makes the server execute a target-less run in place. Today the UI
> creates the pending run and the CLI (`molexp run`) or a compute
> target does the executing.

## 7. Monitor a run

Click any run in the tree to open the run page:

- **Overview** — run id, project, experiment, parameters, results,
  duration, backend, and execution count.
- **Executions** — one row per attempt (`exec-<run_id>-N`) with its
  status and timing; selecting an attempt scopes the Logs tab to it.
- **Logs** — captured stdout/stderr per attempt, with a "view latest"
  shortcut.

Result-type plugins (e.g. LAMMPS logs, TensorBoard) contribute extra
tabs automatically when matching files exist in the run directory.

For a fleet-level view, switch to the **Runs** panel and filter the
dashboard by status or project; the Gantt chart shows attempts over
time.

## 8. Manage runs and clean up

What the tree's context menus offer today:

| Object | Actions |
|---|---|
| Project | Open project · New experiment · Refresh · **Delete project** |
| Experiment | Open experiment · New run · Open workflow · **Delete experiment** |
| Run | Open run · View logs · Copy run ID · **Mark cancelled** |

Deletes ask for confirmation and remove the object and its children
from the workspace. Runs themselves cannot be deleted — a run is the
immutable record of an execution request; you cancel it instead.

**Mark cancelled** flips the run's workspace status to `cancelled`. Be
aware of its scope: it updates the record only — it does **not** kill a
scheduler job that is already executing on a compute target (the
tooltip in the UI says the same).

A `failed` or `cancelled` run can be continued — two distinct verbs,
both keeping the same run id:

- **Resume** — reopen the existing attempt, keep the completed tasks'
  outputs, recompute only what didn't finish.
- **Rerun** — open a fresh attempt (`exec-<run_id>-N`) from the top of
  the graph.

> **TODO** — *Resume / Rerun buttons*: both operations exist as API
> endpoints (`POST …/runs/{id}/resume`, `POST …/runs/{id}/rerun`) and in
> the CLI (`molexp run --resume / --rerun`), but the run page does not
> surface buttons for them yet. Until then, use the CLI.

> **TODO** — *Export a run*: the server can stream a run directory as a
> ZIP (`GET …/runs/{id}/export`), but no download button is wired up in
> the run page yet.

> **TODO** — *Rename and archive*: projects, experiments, and runs can
> be created and deleted from the UI, but not renamed or archived.

> **TODO** — *Bulk operations*: no multi-select; every action is
> per-object.

> **TODO** — *Run comparison*: there is no side-by-side view of
> parameters/results across runs of an experiment yet; the Runs
> dashboard aggregates, but does not diff.

## 9. Compute targets and settings

**Settings** has two tabs:

- **Remote workspaces** — register another MolExp server's workspace so
  it appears in your tree (descriptors live in `~/.molexp/`).
- **Compute targets** — register molq execution targets (stored in the
  workspace's `workspace.json`). Targets registered here are what the
  Create Experiment and Launch Run dialogs offer, and the **+ Add new
  target…** link in those dialogs lands in the same registry.

## 10. Where the agent fits

Everything above is also reachable conversationally: the **Agent
Tasks** panel's **New goal** button starts an LLM session that can
create projects, experiments, and runs through tool calls while you
watch the tree update. That session is driven by the agent's
`InteractiveLoop` — see the [Agent concept](../concept/agent.md) for the
model, and set your provider key with `molexp config set agent.model …`.

## Feature status

| Capability | From the UI | Notes |
|---|---|---|
| Create project / experiment / run | ✅ | Dialogs in the Experiments panel |
| Draft a workflow on the canvas | ✅ | Editable graph, ⌘S saves |
| Monitor runs (overview, attempts, logs) | ✅ | Plus the Runs dashboard |
| Delete project / experiment | ✅ | With confirmation |
| Mark a run cancelled | ✅ | Workspace status only — does not kill a live scheduler job |
| Register compute targets / remote workspaces | ✅ | Settings, or inline from dialogs |
| Assets and lineage browsing | ✅ | Asset panel |
| Agent-driven creation | ✅ | Agent Tasks panel |
| Execute a target-less run server-side | **TODO** | Run stays `pending`; use `molexp run` |
| Resume / Rerun buttons | **TODO** | API + CLI exist; no UI buttons yet |
| Export run as ZIP from the run page | **TODO** | Endpoint exists; no download button yet |
| Rename / archive objects | **TODO** | Create + delete only |
| Bulk operations | **TODO** | No multi-select |
| Run comparison view | **TODO** | Dashboard aggregates, doesn't diff |

## Where to go next

Once the click-through model feels natural, [Your First
Workflow](first-workflow.md) shows how real workflows are authored in
Python, and [Track a Run](tracked-runs.md) explains what all of this
looks like on disk.
