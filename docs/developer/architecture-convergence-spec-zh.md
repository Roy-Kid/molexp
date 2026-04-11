# Molexp 架构收敛与易用性修复 Spec

> 状态：Draft v1
> 日期：2026-04-10
> 范围：`workflow` / `workspace` / `server` / `agent` 公共模型、API 契约、文档与可用性
> 目标：把当前“多套架构并存”的状态收敛为一套可解释、可运行、可测试、可文档化的产品架构

实施拆分见 [architecture-convergence-work-breakdown-zh.md](./architecture-convergence-work-breakdown-zh.md)。

---

## 1. 背景

当前仓库已经完成了部分新架构落地：

- `workflow` 层以 `WorkflowSpec` / `Step` / `WorkflowRuntime` 为中心
- `agent` 层以 `AgentService` / `AgentRuntime` / `PydanticAI` 封装为方向
- `workspace` 层仍承担 `Workspace / Project / Experiment / Run / Asset` 产品语义

但仓库的公共表面并未完成收敛，导致以下现象同时存在：

- README、Quick Start、Core 文档仍在描述旧 `Task / IR / Engine` 架构
- 新旧 import path 在文档中混用，且大量路径已不存在
- server 请求模型、响应模型、核心领域模型互相不一致
- 部分路由与核心对象的方法签名不匹配
- 顶层 import 会隐式拉起远程执行基础设施
- 多个“还未完成”的能力以 stub 或伪成功形式暴露给用户

这不是单点 bug，而是产品架构边界没有收口。

---

## 2. 问题定义

### 2.1 架构问题

1. 没有唯一公共模型
2. 领域模型与接口契约断裂
3. workspace 持久化存在双重真相源
4. 可选基础设施依赖污染默认导入路径

### 2.2 易用性问题

1. 首屏文档不可运行
2. API 文档不能代表真实行为
3. 不完整能力以“看起来可用”的形式暴露
4. 用户无法判断什么是正式支持，什么是迁移残留

---

## 3. 设计目标

### 3.1 产品目标

- 用户只面对一套 Molexp 公共模型
- README、Quick Start、Python API、HTTP API、测试样例描述同一套行为
- 本地 workspace/server 使用场景不依赖远程执行或外部数据库初始化
- 未完成能力要显式失败，不允许伪成功

### 3.2 工程目标

- `workflow`、`workspace`、`server`、`agent` 之间边界稳定
- 请求模型、响应模型、领域模型一一对应
- 文档、代码、测试由同一 contract 驱动
- 清理旧架构残留，降低未来维护成本

### 3.3 成功标准

- 新用户按 README/Quick Start 可在干净环境中完成最小闭环
- `import molexp.server.app` 不触发远程基础设施副作用
- 所有公开路由至少有 contract test 覆盖
- 不再出现文档存在、实现不存在的公共概念

---

## 4. 非目标

- 不在本阶段重做 UI 视觉或前端交互
- 不在本阶段引入复杂迁移工具链
- 不保证对旧文档和旧 import path 的长期兼容
- 不在本阶段扩展新的 workflow 能力面

说明：
若为降低重构风险需要短期 shim，可以存在于内部实现，但不能继续作为公共文档承诺。

---

## 5. 核心设计决策

## 5.1 选择唯一公共模型

公共架构统一为三层：

1. Product Layer：`workspace / assets / project / experiment / run / server`
2. Workflow Layer：`WorkflowSpec / WorkflowBuilder / workflow() / Step / Actor / WorkflowRuntime`
3. Agent Layer：`Goal / AgentSession / AgentService / Tool / ApprovalPolicy`

明确决策：

- 旧 `Task / IR / WorkflowEngine / WorkflowCompiler` 不再作为公共文档入口
- 若仓库内仍保留过渡性实现，必须降级为内部历史残留，不得出现在 README、Quick Start、OpenAPI 示例中
- `molexp.__init__` 不再通过 eager import 扩散整个子系统依赖图

## 5.2 Experiment 和 Run 恢复产品语义

当前产品语义要求：

- `Experiment` 是“可重复执行的 workflow 定义”
- `Run` 是“某次具体执行及其快照”

因此领域模型必须显式承载这些信息：

- `ExperimentMetadata`
  - `workflow_source`
  - `workflow_type`
  - `git_commit`
  - `parameter_space`
  - `description`
  - `tags`
  - `config`
- `RunMetadata`
  - `parameters`
  - `status`
  - `finished_at`
  - `error`
  - `workflow_snapshot`
  - 必要的执行上下文引用

若不准备支持这些概念，则必须同步从请求模型、响应模型、文档中删除，不能继续悬空。

## 5.3 以“少参数默认可用”为 API 原则

面向用户的创建接口统一采用最小必要输入：

- `create_project(name, ...)`
- `project.create_experiment(name, workflow_source, ...)`
- `experiment.create_run(parameters, ...)`

用户可选提供自定义 `id`，但不是必填。自动生成策略必须一致，且文档、CLI、HTTP API 完全一致。

## 5.4 父子关系以目录结构为真相源

对于 `Workspace -> Project -> Experiment -> Run`：

- 子对象存在性和成员关系以目录结构加子对象 metadata 文件为真相源
- 父 metadata 中不再维护 `projects` / `experiments` / `assets` 这类重复列表
- 如保留索引文件，只能作为缓存，必须允许重建

这样可以消除删除、恢复、手工修复时的双重真相源问题。

## 5.5 可选能力必须惰性加载

远程执行、HPC、外部 transfer、agent runtime 等都属于可选能力。

约束如下：

- 默认 import 路径不能初始化它们
- 只有在显式调用相关能力时才加载对应依赖
- 如果环境不满足，返回明确错误，而不是污染基础使用路径

## 5.6 未完成能力必须显式 gated

以下能力在未完成前必须统一策略：

- agent session 执行
- workflow plan/execution API
- registry 查询
- 远程执行

统一行为：

- 返回明确的 `501 Not Implemented` 或领域错误
- 不返回“pending session”一类伪成功对象
- 不在 README/Quick Start 中展示未完成能力

## 5.7 文档与测试是产品表面的一部分

文档和测试不是附属物，而是 contract 的一部分。

必须建立三类 smoke/contract 校验：

1. import smoke
2. quick start smoke
3. API contract smoke

任一失败都视为产品表面损坏。

---

## 6. 目标架构

## 6.1 包边界

### `molexp.workspace`

- 负责工作区目录模型、持久化、资产作用域、Project/Experiment/Run 生命周期
- 不直接依赖 server、agent、远程执行基础设施

### `molexp.workflow`

- 负责 workflow 定义、编译适配、执行 runtime
- 不污染顶层 import
- 远程执行通过惰性 adapter 接入

### `molexp.agent`

- 负责 goal/session/tooling/approval
- 若 runtime 不可用，服务端显式返回能力未启用

### `molexp.server`

- 只暴露已实现并受测试保护的 contract
- 路由不得直接依赖“猜测式字段”或历史属性名

## 6.2 顶层导入策略

`molexp.__init__` 只暴露稳定符号，不执行重型导入。

建议策略：

- 保留 `__version__`
- 使用 lazy import 或显式子模块导入
- 禁止在包初始化时 import `remote`、`molq`、数据库或网络资源

---

## 7. Roadmap

## Phase 0：冻结公共表面并建立清单

目标：
停止继续扩散新旧架构混用，先把“当前支持什么”说清楚。

交付物：

- 公共 API 清单
- 历史残留 import path 清单
- server 路由到领域模型映射清单
- README / docs / tests 的冲突矩阵

退出条件：

- 每个公开入口都标记为 `supported`、`internal`、`remove`
- 未完成能力全部标注 gating 策略

## Phase 1：公共 API 收敛

目标：
只保留一套对外叙事，移除旧架构文档入口。

交付物：

- README 改写到新公共模型
- Quick Start 改写为可运行的新示例
- 删除或降级旧 `Task / IR / Engine` 文档入口
- `molexp.__init__` 去除 eager import

退出条件：

- README 中所有 import path 均真实存在
- Quick Start 能在测试中跑通最小样例
- `python -c "from molexp.server.app import create_app"` 在本地环境可成功导入

## Phase 2：领域模型与持久化修复

目标：
让 Experiment/Run 的产品语义重新落到真实模型上，同时消除双重真相源。

交付物：

- 重构 `ExperimentMetadata` 与 `RunMetadata`
- 移除父 metadata 中重复 child 列表
- 统一创建 API 的参数模型
- 统一 asset library 的查找和索引语义

退出条件：

- HTTP 请求模型与 Python API 使用同一组核心字段
- 删除/列举/读取行为不再依赖重复状态
- run/experiment 的响应字段均来自真实模型，而不是 `getattr(..., None)` 猜测

## Phase 3：Server Contract 修复

目标：
把 server 从“原型路由集合”修到“可依赖 contract”。

交付物：

- 修复所有方法签名不匹配的路由
- 移除 dead code、注释式补丁和不可达重试逻辑
- 未完成端点统一返回 `501`
- 生成稳定的 OpenAPI 文档

退出条件：

- 所有已暴露路由至少有一个成功路径测试
- 所有未完成路由有明确失败测试
- 资产上传、下载、run start、project/experiment CRUD 可端到端运行

## Phase 4：Agent 与 Execution 能力显式化

目标：
把 agent/execution 从“部分可见、部分 stub”改为“已实现则可用，未实现则显式关闭”。

交付物：

- agent runtime availability 检测
- session 持久化与事件流 contract 固化
- execution/plan 能力按实现状态分层暴露
- server 文档中标注 capability matrix

退出条件：

- 不再返回 stub session
- session/events 行为在单进程与恢复场景下有明确测试
- execution 入口不再误导用户以为已完整可用

## Phase 5：回归清理与发布门禁

目标：
防止问题再次回流。

交付物：

- import smoke tests
- doc smoke tests
- API contract tests
- “公共表面回归检查” CI 门禁

退出条件：

- 新 PR 无法再引入不存在的 import path 到 README/Quick Start
- 新增路由必须附带 contract test
- 顶层导入副作用被测试锁住

---

## 8. 验收标准

达到以下标准，视为本 spec 完成：

1. README、Quick Start、HTTP API、Python API 共享同一套术语和对象模型
2. `Project / Experiment / Run` 的创建、读取、执行、状态更新路径不再依赖历史字段猜测
3. server 不再存在签名错配、不可达修复逻辑和伪成功返回
4. 顶层导入不要求远程执行环境、数据库路径或外部服务
5. 未实现能力全部显式 gated
6. 文档示例具备 smoke 测试

---

## 9. 风险与缓解

### 风险 1：重构期间新旧代码并存导致继续扩散

缓解：

- 先执行 Phase 0 清单化
- 所有公共入口先分类，再决定保留/删除

### 风险 2：为快速修复 server 而继续堆积 `getattr` 和特判

缓解：

- 规定 response model 只能映射真实领域字段
- 不允许通过“猜字段”维持表面兼容

### 风险 3：删除旧文档后短期信息缺口

缓解：

- 在新 README 和 Quick Start 中补最小闭环
- 历史设计提案保留，但明确标为历史/迁移文档

### 风险 4：可选依赖拆分影响已有测试

缓解：

- 先建立 import smoke
- 再做惰性加载改造

---

## 10. 决策摘要

本 spec 的核心结论只有三条：

1. Molexp 必须只有一套公共架构叙事
2. 领域模型、接口契约、文档示例必须从同一个真相源生成
3. 未完成能力和可选能力必须显式化，不能再通过 stub、历史残留和导入副作用泄漏到产品表面

如果这三条不先完成，继续叠加 workflow、agent、server 功能只会放大维护成本和首日使用摩擦。
