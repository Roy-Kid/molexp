---
title: Flowgram canvas edit + write-back loop
status: code-complete
created: 2026-06-03
---

# Flowgram canvas edit + write-back loop

## Summary
在 sub-spec 03 落地的只读 flowgram 画布之上，本子规约（链条 `flowgram-workflow-canvas` 的第 4 也是最后一环）补齐**编辑能力**与**写回闭环**：用户在画布上拖拽节点、增删连线 → 将编辑后的 flowgram document **反向序列化**回 molexp IR（`TaskGraphJson`，即 01 的 `task_configs` + 带 `kind` 的 `links` + `position`）→ 通过 sub-spec 02 重新生成的 TS client PUT 到写回端点 → 重载后图与编辑前语义一致。本子规约依赖两个上游：**03**（提供只读画布、前向构建器 `buildFlowgramDocument`、build 配置、`inspectedTask` 点击行为、唯一 IR 类型 `task_graph_ir.ts`）与 **02**（提供写回端点及 `ui/src/api/generated/services/` 中重新生成的 TS client 方法）。用户可见结果：workflow 画布从「只读查看」升级为「可编辑并保存」，保存动作走生成的 client、不手写 fetch，重载后图保持一致。

## Design
- **反向序列化器（前向构建的逆）。** 在 `ui/src/app/state/api.ts` 中、紧邻 03 落地的 `buildFlowgramDocument`，新增纯函数 `flowgramDocToTaskGraphJson(doc): TaskGraphJson`（命名与前向构建器对称）。它消费 flowgram document（节点/连线/位置），产出 03 重塑后的唯一 IR 类型 `ui/src/types/task_graph_ir.ts` 的 `TaskGraphJson`（`task_configs` + 带 `kind` 的 `links` + `position`），**不引入第二套 IR 类型**。该函数与 `ui/src/lib/workflow-utils.ts` 现有的 `toTaskGraphJson` 协调：两者均产出同一 `TaskGraphJson` 形状，共享字段约定（`config` 静态配置、节点 `position`、edge `kind`、`isOutput → targets`）。可复用的纯映射步骤（如 edge → `EdgeJson{from,to,kind}`、isOutput → targets 收集）抽成共享 helper 由两者调用，而非复制逻辑。
- **可编辑画布。** 在 03 的只读 flowgram 画布组件上开启编辑交互：节点拖拽（更新 position）、连线创建/删除（携带 `kind`）、节点增删。编辑状态由组件本地持有（flowgram document 为单一可信源）。03 的 `inspectedTask` 点击 → 右栏 `TaskViewer`（`ui/src/app/renderers/TaskViewer.tsx`，经 `ui/src/app/state/inspectedTask.tsx` 的 `inspectTask`）行为必须保留：点击节点仍 pin 到右栏，不被编辑交互吞掉。
- **写回 chrome 与动作。** 新增一个 toolbar（含「保存」按钮），优先用 shadcn/ui 组件（`Button` 等）。点击保存 → 调用 `flowgramDocToTaskGraphJson` → 通过 02 重新生成的 TS client 服务方法（位于 `ui/src/api/generated/services/`，模式参考 `ui/src/plugins/tensorboard/TensorBoardTab.tsx` 经 `workspaceApi` 调用生成服务）发起 PUT 写回。**禁止手写 fetch**；写回封装为 `api.ts` 中 `workspaceApi` 的一个新方法（如 `updateWorkflowGraph`），内部转调生成的服务方法。保存成功后触发重载，使画布从后端最新状态重新渲染。
- **生命周期/归属。** 反向序列化器与写回封装是无状态纯函数 / thin client wrapper，归属 UI 状态层（`api.ts`）；可编辑画布与 toolbar 归属 03 已建立的 flowgram 渲染器组件。生成的 client（`ui/src/api/generated/`）由 02 产出，本子规约只消费、绝不手改。

## Files to create or modify
- `ui/src/app/state/api.ts` — 新增 `flowgramDocToTaskGraphJson` 反向序列化器（紧邻 03 的 `buildFlowgramDocument`）；在 `workspaceApi` 上新增写回方法，转调 02 重新生成的 `ui/src/api/generated/services/` 服务方法。
- `ui/src/lib/workflow-utils.ts` — 抽出可复用的纯映射 helper（edge→`EdgeJson`、isOutput→targets），供 `toTaskGraphJson` 与新反向序列化器共享，避免重复 IR 构造逻辑。
- `ui/src/app/renderers/WorkflowGraphViewer.tsx`（或 03 实际落地的 flowgram 画布组件）— 开启编辑交互（拖拽/连线/增删），保留 `inspectedTask` 点击行为，挂接 toolbar。
- `ui/src/app/renderers/FlowgramCanvasToolbar.tsx` (new) — shadcn/ui toolbar，含保存按钮，触发写回。
- `ui/src/app/state/api.reverse-serialize.test.ts` (new) — forward(03)+reverse round-trip identity 单测。
- `ui/src/app/renderers/FlowgramWorkflowCanvas.writeback.test.tsx` (new) — 编辑→写回→重载的 ui_runtime 测试，断言走生成的 client、非 raw fetch。

## Tasks

- [ ] Write failing round-trip test for reverse serializer in `ui/src/app/state/api.reverse-serialize.test.ts` (forward∘reverse / reverse∘forward field-equal on representative IR)
- [ ] Extract shared edge->EdgeJson and isOutput->targets mapping helpers in `ui/src/lib/workflow-utils.ts`, refactoring `toTaskGraphJson` to call them
- [ ] Implement `flowgramDocToTaskGraphJson` reverse serializer in `ui/src/app/state/api.ts` beside the 03 `buildFlowgramDocument`, reusing the shared helpers and the sole `TaskGraphJson` type
- [ ] Enable editable interactions (node drag, connect, add/remove with edge kind) on the 03 flowgram canvas while preserving the `inspectedTask` click->TaskViewer path
- [ ] Implement `FlowgramCanvasToolbar` with a shadcn/ui save button in `ui/src/app/renderers/FlowgramCanvasToolbar.tsx` (new)
- [ ] Add `updateWorkflowGraph` write-back method on `workspaceApi` in `ui/src/app/state/api.ts` that calls the 02-regenerated service in `ui/src/api/generated/services/` (no hand-rolled fetch)
- [ ] Write failing ui_runtime test asserting edit -> save -> reload renders identically and save invokes the generated service, not `fetch`
- [ ] Wire the toolbar save action: reverse-serialize -> `updateWorkflowGraph` -> reload
- [ ] Run full check + test suite

## Testing strategy
- **Happy path（round-trip identity / type: code）：** 取一组代表性 IR（多节点、多连线、含 `targets`、含 `position`、含 `kind`），断言 `buildFlowgramDocument ∘ flowgramDocToTaskGraphJson` 与 `flowgramDocToTaskGraphJson ∘ buildFlowgramDocument` 在 `TaskGraphJson` 字段层 field-equal（节点 id/type/config、edge from/to/kind、targets、position 保持）。
- **写回闭环（ui_runtime）：** 渲染可编辑画布 → 模拟一次编辑（拖动节点 / 加一条连线）→ 点保存 → 断言保存动作调用 02 生成的服务方法（mock 该服务，断言被调用且入参为反向序列化后的 `TaskGraphJson`）、**未**调用全局 `fetch` → 模拟重载后画布与编辑后状态渲染一致。
- **inspectedTask 回归（ui_runtime）：** 编辑模式下点击节点仍调用 `inspectTask` 并 pin 到右栏 `TaskViewer`，编辑交互不吞掉点击。
- **IR 类型单一性（type: code）：** 静态/单测断言反向序列化器返回值类型为 `task_graph_ir.ts` 的 `TaskGraphJson`，且 `workflow-utils.ts` 的共享 helper 被两个调用方复用（无重复 IR 构造分支）。

## Out of scope
- 写回**端点**本身（服务端路由 + schema）与 TS client 的**重新生成** — 属 sub-spec 02（本子规约只消费 `ui/src/api/generated/services/` 中已生成的方法）。
- 只读画布、前向构建器、build 配置、`inspectedTask` 点击行为、唯一 IR 类型的**首次落地** — 属 sub-spec 03（本子规约在其之上增量开启编辑并复用其前向构建器作为逆变换的对称面）。
- IR 形状本身（task_configs/links/kind/position 定义）— 由 01 拥有；本子规约只消费。
- 撤销/重做、多用户协同编辑、画布上的实时校验/执行触发 — 未来工作，本子规约只覆盖「编辑 → 反向序列化 → 写回 → 重载一致」最小闭环。
