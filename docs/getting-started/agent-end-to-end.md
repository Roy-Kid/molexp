# Agent end-to-end: 全程通过 UI 让 AI 完成任务

This guide walks you through a fully UI-driven workflow: you start the
backend and the React UI, paste your DeepSeek (or Anthropic / OpenAI /
Google) API key, then drive every subsequent step — creating a project,
creating an experiment with a built-in workflow, submitting and
executing runs, reading back the results — by chatting with the agent.

No CLI commands beyond bringing up the two processes. No Python files
to author. No chemistry.

## 0. Prerequisites

```bash
pip install -e '.[agent]'   # editable install with agent extras
node --version              # 18+
```

Get a DeepSeek API key at <https://platform.deepseek.com/api_keys>.
(Anthropic / OpenAI / Google work identically — the UI flow is the
same.)

## 1. Initialize a workspace

```bash
mkdir -p /tmp/molexp-demo
molexp init /tmp/molexp-demo
```

This is the only filesystem state you'll create by hand. Everything
inside it from this point forward is created by the agent.

## 2. Start the backend (terminal A)

```bash
molexp serve /tmp/molexp-demo --port 8000
```

Leave it running. You should see uvicorn report
`Uvicorn running on http://localhost:8000`.

## 3. Start the UI dev server (terminal B)

```bash
cd /path/to/molexp/ui
npm install                 # first run only
npm run dev                 # http://localhost:5173, proxies /api → :8000
```

> ⚠️ Use `npm run dev`, **not** `npm run dev:mock`. Mock mode never
> reaches the real backend, and `Test connection` will return a
> `[MOCK]` placeholder rather than actually validating your key.

## 4. Configure the LLM provider

In the browser:

1. Left panel → **Agent** view (robot icon).
2. Click the gear icon at the top → opens the Agent settings.
3. **Provider** tab → choose **DeepSeek** (or your provider of choice).
4. Model: leave default (`deepseek-chat`) or pick `deepseek-reasoner`.
5. Paste the API key into the *API key* field.
6. Click **Test connection**.

You should see the green confirmation: `✓ Connection OK
deepseek:deepseek-chat · ~XXX ms` with `pong` as the reply. If you see
a 401 instead, the key is wrong — the failure box shows the exact
error.

7. Click **Save**. The badge flips to `Key configured`.

## 5. Drive everything from the chat

Click **+** to start a new agent session. Paste this goal — copy
verbatim, the agent has tools matching every verb here:

> 在 workspace 里建一个 project 叫 `math`，在它里面建一个 experiment
> `square`，用内置的 `square` 模板（计算 y = x²）。
> 然后提交并执行三次 run，参数分别是 x=3、x=7、x=11。
> 最后把每一个 run 的 x、y、状态用表格列给我。

(English equivalent if you prefer:)

> Create a project called `math` in this workspace. Inside it, create
> an experiment `square` using the built-in `square` template
> (computes y = x²). Submit and execute three runs with x=3, x=7, and
> x=11. When they're done, show me x, y, and status as a table.

What you'll see in the timeline:

| Step | Tool the agent calls | Effect on disk |
|---|---|---|
| 1 | `list_workflow_templates` | none — discovery |
| 2 | `create_project(name="math")` | `projects/math/project.json` |
| 3 | `create_experiment(project_id="math", name="square", template="square")` | `projects/math/experiments/square/experiment.json` |
| 4 | `submit_run(parameters={"x": 3})` × 3 | `…/runs/run-…/run.json` (status pending) |
| 5 | `execute_run(run_id=…)` × 3 | runs the bound `square` callable; each run flips to `succeeded` |
| 6 | `get_run_results(run_id=…)` × 3 | reads `run.json["context"]["results"]` |
| 7 | Final reply | a markdown table with x, y, status |

The center panel shows the SSE timeline (Plan → Tool calls → Tool
results → SessionCompleted). The left panel auto-refreshes — you'll
see the new `math` project, the `square` experiment, and the three
runs appear under it in real time.

## 6. Verify in the workspace tree

Click `math` → `square` → any run in the left panel. The center panel
shows that run's parameters, status, results (x and y), and the
`compute.log` line written by the workflow.

## 7. Sanity check: keep talking to the agent

Start a fresh session and ask:

> What runs exist in project `math` / experiment `square`? List them
> with their parameters and results.

The agent will call `list_runs` and `get_run_results` for each one,
then format a table. No new disk state should be written — purely a
read query.

You can also ask things like:

> Now run x=42 too.

The agent will call `submit_run` then `execute_run` and tell you
`y = 1764`.

## What's covered by the built-in template registry

Three templates ship today (see
`src/molexp/plugins/agent_pydanticai/_pydantic_ai/workflow_templates.py`):

| Template | Parameters | Computes |
|---|---|---|
| `square` | `x` | `y = x²` |
| `cube` | `x` | `y = x³` |
| `add` | `a`, `b` | `z = a + b` |

To add another, append a `(callable, description, [params])` tuple to
`TEMPLATES`. The agent picks it up automatically via
`list_workflow_templates`.

## Server restart caveat

Workflow callables are bound to experiments **in the live server
process**, not persisted on disk. If you restart `molexp serve`, the
experiment row stays but its workflow attachment is gone. Re-call
`create_experiment` with the same name and template in the next chat
session — it's idempotent and re-attaches the workflow without
duplicating the experiment.

## Default approval policy

By default no tool requires explicit approval — your typed goal *is*
the consent. To switch to a confirm-before-act model (useful in
production deployments where the agent may run expensive workloads),
construct `AgentService` with an explicit policy:

```python
from molexp.plugins.agent_pydanticai import AgentService, ApprovalPolicy

service = AgentService.from_workspace(
    "/tmp/molexp-demo",
    approval_policy=ApprovalPolicy(require_approval_for=["execute_run", "retry_run"]),
)
```

The chat will then pause with an `ApprovalRequestEvent` before each
matching tool call; the user clicks Approve / Deny inline.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Test connection` returns `[MOCK] dev:mock cannot validate…` | UI is in mock mode | restart UI with `npm run dev` (drop `:mock`) |
| `Test connection` returns 401 | wrong / expired key | regenerate from the provider console |
| Goal submission returns 400 `agent_not_configured` | no key + no env var | save a key in Provider tab, or `export DEEPSEEK_API_KEY=…` and restart `molexp serve` |
| `execute_run` says "no workflow attached" | server was restarted since `create_experiment` ran | tell the agent "re-attach the `square` template to experiment `square` in project `math`" — it'll re-call `create_experiment` |
| New runs don't appear in the left panel | snapshot poll hasn't refreshed yet (≤5 s) | click any other tree node and back |

## What you've validated end-to-end

- HTTP: UI ↔ FastAPI proxy, OpenAPI client, SSE event stream.
- Provider config: workspace-scoped storage, masked-key API surface,
  live probe against the real LLM.
- Agent runtime: pre-flight credential check, session creation,
  tool dispatch, SSE timeline, persisted-session listing across
  restarts.
- Workspace: project / experiment / run materialization, atomic
  writes, on-disk catalog.
- Workflow: `RunContext`, results, logs, idempotent setup, live
  workflow execution from a binding the agent attached.

If any layer breaks, the chat timeline localizes it: the failed tool
call is right there with its arguments and the error string returned
by the workspace.
