---
title: flowgram-workflow-canvas-03-read-canvas — flowgram 只读画布与 xyflow 全树清除
status: code-complete
created: 2026-06-03
---

# flowgram-workflow-canvas-03-read-canvas — flowgram 只读画布与 xyflow 全树清除

## Summary

把 molexp UI 工作流可视化的**只读路径**从 `@xyflow/react` + `elkjs` + `@dagrejs/dagre` 彻底重写为基于 `@flowgram.ai/free-layout-editor@1.0.11` 的自由布局只读画布，并从整个 `ui/src/` 树删除全部 xyflow/elk/dagre 代码与依赖，不保留任何向后兼容垫片。用户在工作流文件、Run 视图、Workflow 视图里看到的拓扑改由 flowgram document 渲染；点击节点仍经 `inspectedTask` 打开右栏 `TaskViewer`。客户端只保留**单一规范 IR 类型**（`ui/src/types/task_graph_ir.ts`，对齐 01 的 `task_configs` + 带 `kind` 的 `links` + `position`），删除 `types.ts` 里 `{nodes, edges}` 的 `WorkflowGraph` 旧形状与 `buildWorkflowGraph`/`getLayoutedElements` 旧归一化层。本子规格只做只读渲染；编辑/写回是 04。

## Design

依赖 `flowgram-workflow-canvas-01-ir` 已落地的新 IR 形状（`task_configs` + 带 `kind` 的 `links`，节点带 `position`）作为前向构建器的唯一输入契约。

实体与符号变更：

- **唯一客户端 IR 类型** — `ui/src/types/task_graph_ir.ts` 重塑为 01 的形状：`TaskNodeJson { id; type; position?; config?; ... }`、`EdgeJson { from; to; kind }`、`TaskGraphJson { task_configs; links; ... }`（字段名对齐 01；保留 `TaskGraphJson/TaskNodeJson/EdgeJson` 导出名）。这是全树**唯一** IR 接口——`types.ts` 的 `WorkflowGraph { nodes; edges }` / `WorkflowNodeMetadata` / `WorkflowGraphEdge`、`WorkflowGraph.tsx` 内联的 `WorkflowIR`、各 viewer 私有的 `DisplayWorkflowGraph` / `WorkflowFilePayload` 局部 IR 全部收敛到此或被删除。
- **前向构建器** `buildFlowgramDocument(ir: TaskGraphJson): FlowgramDocument` — 取代/重写 `buildWorkflowGraph` 与 `lib/workflow-utils.ts` 的 `getLayoutedElements`。输入新 IR，输出 flowgram free-layout document（`nodes` 携带 `position`、`data`；`edges` 携带 `sourceNodeID`/`targetNodeID`）。规则：每个 `task_config` 一个 document 节点；丢弃引用未知 `id` 的非法 link；缺 `position` 时用确定性回退布局（按入度分层，与现有 longest-path 等价）；遇环不崩溃，每个节点仍得到 position。无 dagre/elk 调用。
- **flowgram 只读画布渲染器** — `WorkflowGraphViewer.tsx`（保持 `*Viewer.tsx` 命名）重写为消费 `buildFlowgramDocument` 输出、用 `@flowgram.ai/free-layout-editor` 渲染的只读组件；`readonly` 模式，无连线/拖拽写回。节点点击经 `useInspectedTask().inspectTask(nodeId, …)` 打开右栏 `TaskViewer`。`WorkflowGraph.tsx` 同步重写为同一画布上的内联/plan-preview 变体，复用 `buildFlowgramDocument` + `parseWorkflowIr`，导出供测试用的纯函数（取代旧 `buildElements`）。
- **装饰器运行时** — flowgram 依赖 `reflect-metadata` 与 legacy 装饰器元数据；`index.tsx` 顶部 `import 'reflect-metadata'`（在任何 flowgram import 之前），`rsbuild.config.ts` 启用 legacy decorator + `emitDecoratorMetadata`。
- **删除项** — `ElkEdge.tsx` / `elkLayout.ts` / `dagreLayout.ts` 三个模块整体删除；`index.tsx` 移除 `@xyflow/react/dist/style.css`；`package.json` 移除三依赖、新增两依赖。

不变量：registry/`registerRenderers.ts`/`AppShell`/`CenterPanel`/导航/`RunViewer.tsx`/`WorkflowViewer.tsx` 接线保持；`inspectedTask` 点击链路保持；plugins（molq/molvis/metrics/tensorboard）不受影响；画布周边新 chrome 优先 shadcn/ui。

## Files to create or modify

- `ui/package.json` — 移除 `@xyflow/react` / `elkjs` / `@dagrejs/dagre`；新增 `@flowgram.ai/free-layout-editor@1.0.11` + `reflect-metadata`。
- `ui/rsbuild.config.ts` — 启用 legacy decorator + `emitDecoratorMetadata`（rspack/swc 配置）。
- `ui/src/index.tsx` — 顶部 `import 'reflect-metadata'`；删除 `import "@xyflow/react/dist/style.css"`（按需引入 flowgram 样式）。
- `ui/src/types/task_graph_ir.ts` — 重塑为 01 的 `task_configs` + 带 `kind` 的 `links` + `position`，作为唯一 IR 类型。
- `ui/src/app/state/api.ts` — 重写 `buildWorkflowGraph` 为 `buildFlowgramDocument`（前向 IR→flowgram document 构建器），更新 `mapWorkflows` 的 summary 统计到新返回形状。
- `ui/src/lib/workflow-utils.ts` — 删除 `getLayoutedElements` / `autoLayoutNodes` 等 xyflow `{Node,Edge}` 工具及 `@xyflow/react` 类型 import；保留与 flowgram 无关的纯图算法或折叠进构建器。
- `ui/src/app/renderers/WorkflowGraphViewer.tsx` — 重写为 flowgram 只读画布（保持命名）。
- `ui/src/app/renderers/WorkflowGraph.tsx` — 重写为复用同一画布的内联变体，导出纯构建函数供测试。
- `ui/src/app/renderers/WorkflowFileViewer.tsx` — 去 xyflow 化，改用 `buildFlowgramDocument` + flowgram 画布。
- `ui/src/app/types.ts` — 删除 `{nodes, edges}` 的 `WorkflowGraph` / `WorkflowNodeMetadata` / `WorkflowGraphEdge` 旧 IR 形状及 `WorkflowSummary.graph` 引用（或改指唯一 IR）。
- `ui/src/app/renderers/ElkEdge.tsx` — 删除。
- `ui/src/app/renderers/elkLayout.ts` — 删除。
- `ui/src/app/renderers/dagreLayout.ts` — 删除。
- `ui/src/app/renderers/__tests__/WorkflowGraph.test.ts` — 改测 `buildFlowgramDocument` 的 flowgram-document 输出。

## Tasks

- [ ] Reshape `ui/src/types/task_graph_ir.ts` to the 01 IR (task_configs + typed links + position) as the sole IR type, and delete the `{nodes,edges}` WorkflowGraph / WorkflowNodeMetadata / WorkflowGraphEdge shapes from `ui/src/app/types.ts`
- [ ] Update `ui/package.json` (drop @xyflow/react, elkjs, @dagrejs/dagre; add @flowgram.ai/free-layout-editor@1.0.11 + reflect-metadata) and enable legacy decorator + emitDecoratorMetadata in `ui/rsbuild.config.ts`; add `import 'reflect-metadata'` and drop the xyflow css import in `ui/src/index.tsx`
- [ ] Delete `ui/src/app/renderers/{ElkEdge.tsx,elkLayout.ts,dagreLayout.ts}` and purge every `@xyflow|elkjs|dagre` import/type from `WorkflowFileViewer.tsx` and `lib/workflow-utils.ts`
- [ ] Rewrite `WorkflowGraph.test.ts` to assert the flowgram-document output (well-formed doc from typed-links+position IR, invalid links dropped, cyclic IR survives) — RED before the builder lands
- [ ] Implement `buildFlowgramDocument` forward builder in `ui/src/app/state/api.ts` (new IR → flowgram free-layout document), replacing `buildWorkflowGraph`; update `mapWorkflows` summary stats to the new shape
- [ ] Implement the flowgram read-only canvas in `WorkflowGraphViewer.tsx` (keep the *Viewer name) consuming `buildFlowgramDocument`, wiring node click to `inspectedTask`
- [ ] Rewrite `WorkflowGraph.tsx` as the inline canvas variant reusing the builder, and re-point `WorkflowFileViewer.tsx` to the flowgram canvas
- [ ] Verify zero `@xyflow|elkjs|dagre` matches anywhere under `ui/src/` and that registry/AppShell/CenterPanel/RunViewer/WorkflowViewer wiring + inspectedTask click path still resolve
- [ ] Run full check + test suite (npm typecheck + test + build green with decorator metadata)

## Testing strategy

- Happy path：`buildFlowgramDocument` 用代表性新 IR（多个 `task_config` + 带 `kind` 的 `links` + 显式 `position`）产出 well-formed flowgram document——节点数等于 task_configs 数，每节点带 position/data，每条合法 link 对应一条 document edge。
- Edge cases：引用未知 `id` 的 link 被丢弃；缺 `position` 时回退布局给每节点确定性坐标；空 IR 产出空 document；**环形 IR 不崩溃**，每节点仍得到 position（沿用旧 `WorkflowGraph.test.ts` 的 linear/diamond/cyclic 用例，retarget 到 document 输出）。
- 单一 IR 不变量：`grep -R "interface .*WorkflowGraph"` 不再出现第二个 `{nodes,edges}` IR；`task_graph_ir.ts` 是唯一被 import 的 IR 类型。
- 全树 grep bar：`ui/src/` 对 `@xyflow|elkjs|dagre` 零匹配；`ElkEdge.tsx`/`elkLayout.ts`/`dagreLayout.ts` 文件不存在。
- UI runtime（hint mol:web）：flowgram 画布在浏览器渲染出节点/连线；点击节点经 `inspectedTask` 打开右栏 `TaskViewer`；plugins 面板不受影响。
- Runtime：`npm run typecheck && npm run test && npm run build` 全绿，装饰器元数据在构建产物中生效。

## Dependency boundary — canvas core only, no FlowGram materials

只引入 FlowGram 的**画布内核** `@flowgram.ai/free-layout-editor`（+ `reflect-metadata`）。**明确不引入** `@flowgram.ai/form-materials` / `@flowgram.ai/form-antd-materials` / `@flowgram.ai/coze-editor` —— 它们建立在 **Semi Design**（`@douyinfe/semi-ui`）或 Ant Design 之上，会把第二套 UI 体系拖进 molexp（与 shadcn/ui + Tailwind 冲突、显著增重）。画布内的节点一律用 molexp 自己的 **shadcn/ui** 组件自定义渲染（node-registries / 自定义节点组件），不使用 FlowGram 物料。

**Monaco 不动。** 代码编辑面（`ui/src/app/renderers/TextEditor.tsx` 通用文件编辑、`WorkflowSourceViewer.tsx` 工作流源只读 Source 标签）全部保留现状，继续用 `@monaco-editor/react` / `monaco-editor`。本次迁移只替换**图渲染栈**（xyflow/ELK/dagre → flowgram 画布内核），不触碰任何 Monaco 代码编辑器。

## Out of scope

- 编辑、连线、节点拖拽、写回（IR 回存）——属于 `flowgram-workflow-canvas-04`。本子规格画布严格 `readonly`。
- 任何 Monaco / 代码编辑器改动（`TextEditor` / `WorkflowSourceViewer` / "Source" 标签）——保留现状，不替换。
- 引入 FlowGram 的 form-materials / Semi Design / Ant Design 物料——明确不做。
- 修改服务端 IR / OpenAPI / `schema/workflow.json`——由 01 拥有；本子规格只消费其形状。
- plugins（molq / molvis / metrics / tensorboard）渲染逻辑——不触碰，仅验证其不回归。
- 节点状态着色 / 边语义状态推导的视觉细节调优——保留功能等价即可，不在本规格做新设计。
