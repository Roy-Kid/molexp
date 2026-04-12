# Spec: Full-screen Run Monitor for Molexp, Powered by Molq Panels

## Goal

为 `molexp` 设计一个全屏终端 monitor，用于观察本地或远程任务的运行状态。

该 monitor 的体验目标类似 `htop` / `btop`：不是简单打印一张表，而是一个**占满终端的 dashboard**，包含顶部 overview、进度展示、底部任务列表，以及基本的交互关闭/重开能力。

该设计明确以下边界：

- 界面如何组织
- `molq` 插件负责什么
- `molexp` 负责什么
- 两者如何协作
- 哪些是必须满足的交互需求

本 spec 只定义**产品与架构需求**，不规定具体代码结构、类名或实现方式。

---

## Design Principles

### 1. Full-screen monitor, not inline table

当前的小 panel + 单个表格形式不可接受。monitor 必须是一个完整终端视图，视觉和交互都应接近系统监控工具，而不是 CLI 的一次性文本输出。

### 2. Overview-first, list-second

最重要的信息不是单行任务表，而是整体状态。界面必须优先展示：

- 当前是否在运行
- 总体进度
- running / pending / done / failed 的总体数量
- 当前 workflow / run 的整体状态

任务列表作为下半部分内容，而不是整个界面的核心。

### 3. Single-job and multi-job must both look natural

即使只有 1 个 job，也不应退化成一张尴尬的小表。同一套 full-screen monitor 应同时适用于单任务、多任务 sweep、本地运行、远程调度运行。

### 4. Status-first, table-second

状态必须是界面中的第一视觉锚点。用户应一眼看到 running / pending / done / failed，而不是先看 run id 或 scheduler id。

### 5. Monitor is a viewer, not the run itself

关闭 monitor 不应终止运行。monitor 是观察界面，不是任务本体。

---

## UI Requirements

### Overall Layout

monitor 占用整个 terminal，包含以下三个区域：

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

展示 monitor 的身份信息和当前整体状态。至少包含：

- 当前 workflow / run / experiment 的标识
- 当前整体状态（running / finished / failed / mixed）
- 最近更新时间或刷新时间

### B. Overview Area

界面的核心，位于上半部分。至少展示：

- 总任务数
- running / pending / done / failed 数量
- 整体进度（必须有明确可读的 progress visualization，例如进度条）

进度不能只是 `0/1 done` 这种弱表达，必须有更强的 overview 感知。

### C. Job List Area

位于下半部分，展示每个 job / run 的状态。每条目至少包含：

- state
- run identity
- scheduler id（如有）
- 其他次要元信息（elapsed、node、message 等由 agent 自行判断）

列表服务于"快速扫状态"，而不是做数据库式大表格。

### D. Footer / Hint Area

界面底部保留简洁的交互提示区域。至少提示：

- `q` 关闭 monitor
- 其他交互提示由 agent 自行设计

---

## Visual Requirements

### 1. Avoid heavy boxed table aesthetics

不接受"一个 panel 里套一张 box-heavy 表格"的风格。减少无意义边框，强调分区、状态和进度。

### 2. Strong visual status hierarchy

状态必须有明显视觉区分。颜色、图标、位置和文本都应服务于快速识别。

### 3. Scheduler ID is secondary

`scheduler id` 不是第一重要信息，不应压过状态和 run identity。

### 4. Progress must be visually central

overview 中必须有明确的 progress visualization。这不是可选增强，而是 monitor 的核心组成。

---

## Interaction Requirements

### 1. `q` closes the monitor

用户按 `q` 时，关闭 full-screen monitor，返回普通 CLI。

### 2. Closing the monitor must not cancel jobs

`q` 的语义只能是 close viewer / quit monitor / return to normal CLI。绝不能终止远程任务、本地 workflow 或 scheduler jobs。

### 3. Monitor must be reopenable

关闭后，用户必须能通过显式命令再次打开 monitor。命名不强制，但 monitor 不是一次性的，必须可重复进入。

### 4. Run command and watch command should cooperate naturally

如果 `run` 命令在提交后自动进入 monitor，退出 monitor 后应能回到普通 CLI，并给出明确提示：

- 任务仍在继续
- 用户可以再次打开 monitor

---

## Molexp vs Molq Responsibilities

### Molexp responsibilities

`molexp` 是 workflow / run orchestration 层，负责：

- 决定何时进入 monitor，何时退出 monitor
- 管理 monitor 的打开 / 关闭 / 重新打开
- 管理 run / workflow / experiment 的生命周期
- 收集和整理运行状态数据
- 将运行状态转换成 monitor 所需的通用状态输入
- 在 CLI 流程中决定 monitor 与普通输出之间如何切换

**`molexp` 拥有 monitor 的生命周期控制权。**

### Molq responsibilities

`molq` 插件负责 monitor 的**界面能力**，而不是 workflow 生命周期。它负责：

- 提供 full-screen panel / dashboard 能力
- 提供 overview 区域和 job list 区域的界面表达
- 提供 monitor 的交互语义解释
- 提供可复用的 terminal viewer 体验

`molq` 不主导任务生命周期，不决定 run 是否结束、是否继续提交等 workflow 语义。

### Interaction contract between Molexp and Molq

| 职责 | 归属 |
|------|------|
| monitor 何时创建、关闭、重新进入 | molexp |
| monitor 如何呈现、基本交互 UI 语义 | molq |
| workflow 是否结束、是否继续提交 | molexp |
| Rich 布局细节、panel 组件 | molq |
| UI 交互结果转化为 CLI/生命周期动作 | molexp |

规则：

1. **Molexp owns control flow** — `molexp` 控制 monitor 生命周期
2. **Molq owns presentation semantics** — `molq` 提供 monitor 如何呈现
3. **Molq must not own workflow lifecycle** — `molq` 不自己决定"停止任务""终止 workflow"
4. **Molexp must not reimplement panel logic** — `molexp` 不应自己硬写所有 Rich 布局，否则 `molq` 作为 panel/plugin 层失去意义
5. **UI actions must be interpretable by Molexp** — monitor 中的交互结果应能被 `molexp` 接住并转化为 CLI/生命周期动作

---

## Architectural Constraints

1. **No giant monolithic CLI table renderer** — 不接受把所有逻辑直接写成单个 CLI 输出函数
2. **No inversion of control from Molq into Molexp lifecycle** — `molq` 不能反过来控制 `molexp` 的主流程
3. **No scheduler-specific UI hardcoding** — UI 不应被 slurm/pbs/lsf 某一种调度器结构绑死；monitor 应围绕 run/job 状态设计
4. **UI design should scale from local to remote** — 同一套 monitor 应同时服务于 local、remote、scheduler-backed runs

---

## Agent Tasks

agent 需要基于当前 codebase 设计并实现：

1. 一套 full-screen terminal monitor 的信息架构
2. 单任务和多任务都自然的界面组织方式
3. `molexp` 与 `molq` 之间的职责边界与接口契约
4. monitor 打开、关闭、重新打开的交互流程
5. 能够支撑 future remote backends 的状态模型

agent 自行决定：

- 具体布局组件
- 具体 Rich 组织方式（layout / panel / live / screen 等）
- 具体交互键位集合
- 具体 state object / panel object / plugin object 设计

---

## Acceptance Criteria

| 要求 | 类别 |
|------|------|
| monitor 是 full-screen dashboard，而不是小 panel | UI |
| 上半部分有清晰 overview 和进度展示 | UI |
| 下半部分有 job list | UI |
| 状态比表格列更重要，视觉上优先 | Visual |
| 单任务场景也自然，不退化 | UI |
| `q` 只关闭 monitor，不取消运行 | Interaction |
| monitor 可重新打开 | Interaction |
| `molq` 负责 panel/view 能力 | Architecture |
| `molexp` 负责 monitor 生命周期与 CLI 切换 | Architecture |
| 两者职责清晰，不互相吞没 | Architecture |

---

## Non-goals

本 spec 不要求：

- 具体类设计或函数签名
- 具体键盘事件实现方式
- 具体 scheduler adapter 代码
- 具体 Rich API 写法
- 完整 TUI 框架抽象

这些由 agent 根据 codebase 自行判断和设计。
