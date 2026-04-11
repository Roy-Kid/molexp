# Molexp 架构收敛实施拆分

> 关联文档：[architecture-convergence-spec-zh.md](./architecture-convergence-spec-zh.md)
> 用途：把收敛 spec 拆成可以直接排期和实施的工作包
> 状态：Draft v1

---

## 1. 使用方式

本文件不重复解释为什么要做，而是回答三个问题：

1. 先做什么
2. 每一步改哪些东西
3. 做到什么程度算完成

建议按 workstream + issue 的方式执行。

- workstream 用来分责任边界
- issue 用来落具体变更
- milestone 用来控制顺序和依赖

---

## 2. 实施顺序

建议按以下顺序推进：

1. 冻结公共表面和导入边界
2. 修复核心领域模型和持久化真相源
3. 修复 server contract 与路由实现
4. 清理未完成能力的暴露方式
5. 重写文档与示例
6. 加上 smoke/contract 门禁

原因：

- 不先冻结公共表面，后续每一步都会返工
- 不先修领域模型，server 修复会继续堆 `getattr` 和特判
- 不先修导入边界，测试和本地使用都会继续被可选依赖污染

---

## 3. Workstreams

## WS1：公共 API 与包边界收敛

目标：

- 明确 Molexp 只支持一套公共模型
- 清理顶层 import 副作用
- 把历史残留 API 从文档和入口层拿掉

涉及模块：

- `src/molexp/__init__.py`
- `src/molexp/workflow/__init__.py`
- `src/molexp/workflow/remote.py`
- `README.md`
- `docs/get-started/*`
- `docs/core/*`

完成标准：

- 顶层 import 不初始化远程执行基础设施
- README 中所有 import path 真实存在
- Quick Start 描述的新用户路径能运行

## WS2：领域模型与持久化修复

目标：

- 让 `Experiment` 和 `Run` 承载真实产品语义
- 消除 parent metadata 与目录结构的双重真相源
- 统一 Python API 和 HTTP API 的字段模型

涉及模块：

- `src/molexp/workspace/metadata.py`
- `src/molexp/workspace/workspace.py`
- `src/molexp/workspace/project.py`
- `src/molexp/workspace/experiment.py`
- `src/molexp/workspace/run.py`
- `src/molexp/workspace/asset.py`

完成标准：

- `ExperimentMetadata` 与 `RunMetadata` 字段能支撑当前对外文案
- 删除/列举逻辑不再依赖重复索引
- asset 语义统一，不再混用按 name 查找和按 asset_id 查找

## WS3：Server Contract 修复

目标：

- 请求模型、响应模型、路由实现、领域模型完全对齐
- 清除不可达代码、签名错配和伪修复逻辑

涉及模块：

- `src/molexp/server/schemas/requests.py`
- `src/molexp/server/schemas/responses.py`
- `src/molexp/server/routes/*.py`
- `src/molexp/server/app.py`
- `src/molexp/server/dependencies.py`

完成标准：

- 所有公开路由都能解释成真实模型操作
- 所有 response 字段都来自真实字段，不靠 `getattr` 猜测
- OpenAPI 能代表真实可用行为

## WS4：未完成能力 gating

目标：

- agent、execution、registry、remote 这类未完成能力显式化
- 未启用和未实现是明确错误，不是伪成功

涉及模块：

- `src/molexp/server/routes/agent.py`
- `src/molexp/server/routes/execution.py`
- `src/molexp/server/routes/registry.py`
- `src/molexp/agent/*`
- `src/molexp/workflow/runtime.py`

完成标准：

- 不再返回 stub session
- 未实现端点统一返回明确错误
- 文档不再宣称这些能力已可用

## WS5：文档和示例重写

目标：

- README、Quick Start、Core 文档全部切到唯一公共模型
- 删除失效 import path 和旧架构叙事

涉及模块：

- `README.md`
- `docs/index.md`
- `docs/get-started/*`
- `docs/core/*`
- `docs/workspace/*`

完成标准：

- 文档不再引用不存在模块
- 文档中的创建/执行路径与当前代码一致
- 旧 `Task / IR / Engine` 文档要么删除，要么明确标历史

## WS6：测试与门禁

目标：

- 把公共表面收敛后的行为锁住
- 防止 README、导入边界、server contract 再次漂移

涉及模块：

- `tests/server/*`
- `tests/workspace/*`
- `tests/workflow/*`
- 新增 smoke/contract tests

完成标准：

- 顶层 import smoke 测试通过
- Quick Start smoke 测试通过
- server contract tests 覆盖所有公开路由

---

## 4. Issue 拆分

下面的 issue 顺序已经按依赖排好，适合直接开工。

## I0：公共表面清单化

目标：

- 产出一份当前公开 API、历史残留 API、待删除 API 的清单

主要动作：

- 梳理 README、docs、server routes、`__all__`
- 标注每个入口是 `supported`、`internal`、`remove`
- 标注每个未完成能力的现状和 gating 策略

依赖：

- 无

产物：

- 一份清单文档或 issue comment

Definition of Done：

- 后续所有实现都以这份清单为准

## I1：顶层 import 去副作用

目标：

- `import molexp` 与 `from molexp.server.app import create_app` 不触发 `molq` 初始化

主要动作：

- 改造 `src/molexp/__init__.py`，避免 eager import 子模块
- 审查 `src/molexp/workflow/__init__.py` 的 import 树
- 将 `remote` 相关能力改为按需加载

涉及文件：

- `src/molexp/__init__.py`
- `src/molexp/workflow/__init__.py`
- `src/molexp/workflow/remote.py`

依赖：

- I0

Definition of Done：

- server tests 至少可以完成 import 和 app 初始化
- 不再出现因远程执行依赖导致的导入失败

## I2：README 与 Quick Start 收敛到新模型

目标：

- 用户第一眼看到的内容就是唯一支持模型

主要动作：

- README 全部切换到 `molexp.workflow` / `molexp.workspace` 当前实际 API
- 删除或替换旧 `TaskEngine`、`workflow.node`、`workspace.core` 等路径
- Quick Start 改写为可运行的最小例子

涉及文件：

- `README.md`
- `docs/index.md`
- `docs/get-started/overview.md`
- `docs/get-started/quick-start.md`

依赖：

- I1

Definition of Done：

- 文档中的 import path 全部真实存在
- 样例代码可以被 smoke 测试执行

## I3：Experiment 模型恢复产品语义

目标：

- `Experiment` 真正表示“可重复执行的 workflow 定义”

主要动作：

- 为 `ExperimentMetadata` 增加 workflow 相关字段
- 调整 `Project.create_experiment()` 参数签名
- 统一 CLI、server request、response 的 experiment 字段

涉及文件：

- `src/molexp/workspace/metadata.py`
- `src/molexp/workspace/project.py`
- `src/molexp/workspace/experiment.py`
- `src/molexp/server/schemas/requests.py`
- `src/molexp/server/schemas/responses.py`
- `src/molexp/cli/__init__.py`

依赖：

- I0

Definition of Done：

- experiment create/read 的 workflow 字段来自真实 metadata
- 不再通过 `workflow_template` 一类悬空属性拼响应

## I4：Run 模型恢复执行快照语义

目标：

- `Run` 真正表示一次具体执行及其快照

主要动作：

- 明确 `RunMetadata` 中的 snapshot 和 execution 相关字段
- 统一 `create_run()`、状态更新、执行上下文落盘逻辑
- 修复 `RunResponse` 对 `workflow_snapshot`、`finished_at` 的映射

涉及文件：

- `src/molexp/workspace/metadata.py`
- `src/molexp/workspace/run.py`
- `src/molexp/server/schemas/responses.py`
- `src/molexp/server/routes/run.py`

依赖：

- I3

Definition of Done：

- run read/status/update 使用一致字段
- run 响应不再依赖不存在的属性

## I5：移除双重真相源

目标：

- parent metadata 不再维护重复 child 列表

主要动作：

- 去掉 `WorkspaceMetadata.projects`
- 去掉 `ProjectMetadata.experiments`
- 评估 `ProjectMetadata.assets` 是否保留为可重建缓存，否则删除
- 调整 create/delete/list 流程

涉及文件：

- `src/molexp/workspace/metadata.py`
- `src/molexp/workspace/workspace.py`
- `src/molexp/workspace/project.py`
- `src/molexp/workspace/base.py`

依赖：

- I3

Definition of Done：

- create/list/delete 只依赖目录结构和子 metadata
- 删除后不会留下静默脏索引

## I6：统一 AssetLibrary contract

目标：

- 资产操作语义统一，不再混用 name 和 asset_id

主要动作：

- 确定 `get_asset(name)` 与 `get_by_id(asset_id)` 的正式接口
- 若 server 需要按 ID 读取，就显式补 API，而不是假定存在 `get()`
- 明确下载 payload 的正式接口

涉及文件：

- `src/molexp/workspace/asset.py`
- `src/molexp/server/routes/asset.py`
- `src/molexp/server/routes/project.py`

依赖：

- I5

Definition of Done：

- asset 上传、查询、下载的路由都调用真实存在的方法
- 不再有按 name/index 与按 id/path 混乱切换

## I7：修复 server 路由签名错配

目标：

- 去掉当前路由层里的假设式调用和不可达修复代码

主要动作：

- 修 `project` asset upload 路由
- 修 `run.start` 路由对 `Run` 上下文 API 的调用
- 对所有 CRUD 路由做一次签名一致性检查

涉及文件：

- `src/molexp/server/routes/project.py`
- `src/molexp/server/routes/run.py`
- `src/molexp/server/routes/experiment.py`
- `src/molexp/server/routes/asset.py`

依赖：

- I4
- I6

Definition of Done：

- 路由代码中不再出现不可达补丁逻辑
- 所有调用都能直接映射到真实对象方法

## I8：请求/响应模型重对齐

目标：

- 请求模型不要求实现不存在的数据
- 响应模型不暴露实现不存在的数据

主要动作：

- 重写 `requests.py`
- 重写 `responses.py`
- 为每个 endpoint 明确字段来源

涉及文件：

- `src/molexp/server/schemas/requests.py`
- `src/molexp/server/schemas/responses.py`
- 对应 routes

依赖：

- I3
- I4
- I6

Definition of Done：

- API schema 与真实代码一致
- OpenAPI 不再包含误导性必填字段

## I9：未完成能力统一 gating

目标：

- agent、execution、registry 的未完成状态可解释、可预测

主要动作：

- agent route 不再返回 stub session
- execution/plan/registry 明确返回 `501` 或能力未启用错误
- 文档同步降级这些能力表述

涉及文件：

- `src/molexp/server/routes/agent.py`
- `src/molexp/server/routes/execution.py`
- `src/molexp/server/routes/registry.py`
- `src/molexp/agent/*`

依赖：

- I0

Definition of Done：

- 用户不会收到“看起来成功但实际没做事”的响应

## I10：文档全量重写与历史文档降级

目标：

- 文档完全以当前唯一公共模型为准

主要动作：

- 重写 `docs/core/*` 和 `docs/workspace/*`
- 对旧架构文档加“历史/迁移文档”标记，或直接移除导航入口
- 清理 README 中失效链接

涉及文件：

- `README.md`
- `docs/index.md`
- `docs/core/*`
- `docs/workspace/*`

依赖：

- I2
- I8
- I9

Definition of Done：

- 搜索文档中不再出现已删除的公共 import path

## I11：smoke 与 contract tests

目标：

- 把本次收敛后的产品表面锁住

主要动作：

- 增加 import smoke tests
- 增加 Quick Start smoke tests
- server route contract tests 覆盖 CRUD、asset、status、gating 行为

建议测试集：

- `tests/smoke/test_imports.py`
- `tests/smoke/test_quickstart_examples.py`
- `tests/server/test_contract_projects.py`
- `tests/server/test_contract_experiments.py`
- `tests/server/test_contract_runs.py`
- `tests/server/test_contract_assets.py`
- `tests/server/test_contract_capability_gating.py`

依赖：

- I1
- I7
- I8
- I9

Definition of Done：

- 文档、导入、路由 contract 漂移会被 CI 阻止

---

## 5. 建议里程碑

## M1：公共表面冻结

包含：

- I0
- I1
- I2

可见成果：

- 新用户至少能 import 和看懂正确入口

## M2：领域模型收口

包含：

- I3
- I4
- I5
- I6

可见成果：

- `Project / Experiment / Run / Asset` 语义与真实数据结构一致

## M3：Server 可依赖

包含：

- I7
- I8
- I9

可见成果：

- API 可用性不再靠运气

## M4：文档与门禁闭环

包含：

- I10
- I11

可见成果：

- 文档、测试、实现同构

---

## 6. 推荐开工方式

如果要立即实施，我建议按下面节奏：

第 1 轮：

- I1 顶层 import 去副作用
- I2 README / Quick Start 收敛

第 2 轮：

- I3 Experiment 模型
- I4 Run 模型
- I5 双重真相源移除

第 3 轮：

- I6 AssetLibrary contract
- I7 路由签名修复
- I8 请求/响应模型重对齐

第 4 轮：

- I9 能力 gating
- I10 文档全量重写
- I11 smoke/contract tests

这样做的好处是：

- 每轮都能形成稳定增量
- 不会先修文档再推翻
- 能较早恢复最关键的开发和测试路径

---

## 7. 实施准备清单

在真正开始改代码前，建议先确认：

- 是否接受“旧 API 不再对外承诺”
- 是否接受 `Experiment` / `Run` 恢复 workflow snapshot 语义
- 是否接受未完成能力统一返回错误而不是 stub
- 是否接受本轮优先修 contract，不扩新功能

如果以上四点都接受，就可以按本拆分直接开工。
