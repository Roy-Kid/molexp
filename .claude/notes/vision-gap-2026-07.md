# 愿景差距审查 — 2026-07-03

愿景基准：**以 Workspace 为边界、以 Workflow/Experiment/Run 为执行骨架、以 Knowledge 为长期上下文、以 Agent 为操作界面的科学工作系统**——研究人员一站式管理任务、模拟数据、发现的知识，并借 AI 创建、执行和回收任务。

方法：四路并行只读审查（每支柱一路，全部结论有 file:line 证据，以 dev 分支活代码为准，commit b3281ec 之后）。

## 总评

| 支柱 | 评分 | 一句话 |
|---|---|---|
| 执行骨架（Workspace/Workflow/Run） | **8/10** | 四根里最扎实：三面动词等价、sweep 闭环、HPC 全线真实 |
| Knowledge——存储面 | **7/10** | 类型/边/来源约束/编辑器齐备，检索薄 |
| Knowledge——上下文面 | **2.5/10** | 只写不读：知识→Agent 零通道，执行→知识仅一条窄缝 |
| Agent 操作界面 | **4.5/10** | 创建深而窄（PlanMode 13 步真通），执行 CLI 独占，回收半残 |
| 一站式整合（UI/server 缝合） | **6/10** | 各 tab 有真后端，但四支柱之间的缝断了三条半 |

**核心诊断：四根柱子都立起来了，屋顶还没盖。** 单支柱内部（尤其执行骨架）成熟度高；愿景独有的价值——四支柱**之间**的闭环——正是断的地方。距愿景整体约 55-60%。

## 六条断缝（三路独立交叉印证）

1. **知识→Agent 零通道**（最致命）。InteractiveLoop 只有 read_file/list_directory/search_code 三个通用文件工具（`agent/loops/interactive/tools.py:92-176`）；PlanMode 九步无一 stage 读知识；自述"hand to agents"的 `WorkspaceContext.knowledge`（`workspace_context.py:1-5`）实际只有 HTTP 端点和 CLI 展示在消费。agent 规划下一个实验时看不到上一个实验的 Finding/FailureAnalysis。
2. **执行→知识窄通道**。唯一回流：`services/plan_runtime/record.py:207-262` 把 experiment_report 写成 Decision 型 KnowledgeItem（带 SourceRef + typed 边——机制是好的）。但 execute 尾链的真正科学结论 final_report/audit_report 不回流；普通 run.execute/sweep 零回流；9 类 KnowledgeKind 生产代码只写 1 类。
3. **Agent→执行 CLI 独占**。服务端 plan 任务硬编码 `PlanMode()`（execute=False，`services/plan_runtime/task.py:107`），`PlanTaskCreateRequest` 无 execute 字段——UI 永远到不了真实执行。run/resume/rerun/cancel/prune 五个 lifecycle 动词零 agent 可达路径（无 ToolCapability、无 loop tool）。
4. **安全：服务端破坏性操作自动放行**（两路独立发现）。`ApprovalGate` 默认 `auto_grant_approver`（`approval_gate.py:44,75`），服务端 curate 任务显式 `approve=None`（`curate_runtime/task.py:86-93`）→ UI 发起的任务可无人审批 `delete_folder`。审批目前只在 CLI TTY 上真实成立；UI 无任何审批交互面。
5. **UI 缝合断裂**。① Settings 模型配置打到 `agent_admin.py:26-33` 的 503 catch-all stub——纯 UI 用户第一次用 AI 前必然跌回 CLI（`molexp config set agent.model`）；② plan↔run 双向断（run 不知 plan 出身，`relations.ts:145-150` 明写 agent/knowledge 无边；DeliverablesPanel 的 run id 是纯文本 `DeliverablesPanel.tsx:837`）；③ knowledge 未织入实体页与 ⌘K（entity 页无 Notes 区，catalog 只索引六类）；④ curate 路由+生成客户端都在，UI 零消费。
6. **产物血缘是可选注解，不是引擎事实**。`Producer.inputs` 只在任务作者显式传 `consumed=[...]` 时落账（`assets/accessors.py:86-87`）；workflow 引擎从不自动记录 task 输入→输出 asset 边（grep `consumed=` 在 `src/molexp/workflow` 零命中）。FAIR 溯源的核心承诺尚未由骨架自动兑现。另：检索面三面不对称（CLI asset 只有 list；`Bundle.search` 仅 path/title 子串、不搜 body、无 server/UI 暴露）。

## 建议路线（按杠杆排序）

### P0 — 合闭环（高杠杆，多为接线活）
1. **UI 审批成一等公民 + plan execute 字段**：plan-tasks 请求加 `execute`；审批从 auto-grant 改为 server 挂起→UI 批复（pending approval 资源 + SSE + 批复路由）。同时修安全洞与打通 AI→执行。
2. **Settings 模型配置接线**：`services/operator_config` 已存在，把 `agent_admin.py` 503 stub 换成真读写——纯 UI 动线第一断点即除。
3. **知识→Agent 通道**：`WorkspaceContext.knowledge` 注入 plan 系统提示词；InteractiveLoop 加 `knowledge_search`/`read_note` 工具（懂 Bundle/backlinks，非裸 grep）。
4. **执行→知识回流补全**：final_report→Finding、失败→FailureAnalysis、audit_report→records 边；普通 run/sweep 提供 opt-in 收割。

### P1 — 操作面加宽
5. run lifecycle 动词（cancel/resume/rerun/prune）做成带 `side_effects` 的 ToolCapability，复用现成闸门。
6. UI 缝合：plan↔run 双向链接、entity 页 Notes 区、⌘K 收 knowledge 文档、agent prose 里 run/plan id linkify。
7. 检索：`Bundle.search` 搜 body + server `/knowledge/search` + UI 搜索框；CLI `asset info/lineage` 子命令。

### P2 — 骨架深化
8. 引擎自动血缘：workflow 引擎在 values-on-edges 层自动落 task 输入→输出 asset 边（FAIR 承诺自动化）。
9. 挂载点上下文：挂在 run 上的 AgentSession 自动注入 run 的 params/status/artifacts 摘要。
10. workspace 级事件流消费端（全局"最近发生了什么"视图）；`rehome_asset` 补实现或删除（现为名存实亡，选中即 `CurationArgumentError`）。

## 各支柱亮点（不要在重构中丢掉）

- 执行骨架：retryable 域单一来源、reaper 三面共用、`--fresh` 跨调度器 marker、RemoteTarget→molq SshTransport→monitor TUI 全线真实。
- Knowledge 存储：KnowledgeItem 强制 ≥1 SourceRef（构造即验证）、EdgeRole 五词表读写同格式、doc_embed 实体嵌入。
- Agent：PlanMode 13 步真实 DeepSeek 贯通（含审批+血缘审计）业界少见；curation NL→gate→mutation 骨架成立。
- UI：`relations.ts` 单点声明类型边 + RelatedPanel、⌘K、verb 铁律镜像到按钮级、agent SSE。

已知小债（审查中顺带记录）：`record.py:236` Bundle 深嵌套 path-doubling bug（带 workaround 上线）；`record.py:69-70` 知识写入失败仅 warning 吞掉；`assets.scan` 全树遍历在大 workspace 的规模瓶颈（有意取舍，暂不动）。
