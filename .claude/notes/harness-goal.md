# Molcrafts Scientific Workflow Harness — Goal / North Star

**Status**: target architecture. Today's `architecture.md` describes current state; this file describes what we are building toward. Reconcile incrementally as each Phase (§16) lands.

## 0. 目标

本系统的目标不是做一个"会写实验流程的 agent"，而是做一个 **provenance-first scientific workflow harness**。

它接收用户口语化的实验计划，将其转化为规范实验报告、结构化 workflow 中间表示、绑定到 Molcrafts 生态的可执行任务图、可运行测试和可审计执行记录。

系统中 agent 只负责生成候选内容。harness 拥有状态、校验、审批、执行、日志、版本、测试、回放和追踪。

一句话定义：

> Molcrafts Harness is a provenance-first scientific workflow harness that uses agents to transform informal experimental intent into validated reports, workflow IRs, Molcrafts-bound executable tasks, test specifications, and auditable execution records.

中文定义：

> Molcrafts Harness 是一个以可追踪性为中心的科学工作流 harness，用 agent 将口语化实验意图转成规范实验报告、可校验 workflow IR、绑定到 Molcrafts 生态的可执行任务图、测试规范和完整审计记录。

---

## 1. 核心原则

### 1.1 Agent proposes, harness disposes

Agent 只能生成 proposal，不能直接改变系统状态。

错误设计：

```text
agent 生成 workflow → agent 写文件 → agent 执行命令 → agent 总结结果
```

正确设计：

```text
agent 生成 proposal
    ↓
harness schema validation
    ↓
harness policy validation
    ↓
harness approval gate
    ↓
harness stores artifact
    ↓
harness executes controlled task
    ↓
harness records provenance
```

### 1.2 Everything is an artifact

所有中间产物都必须变成 artifact。

包括：

```text
用户原始计划
规范实验报告
workflow IR
Molcrafts-bound workflow
测试规范
执行配置
命令 stdout/stderr
模拟输入文件
模拟输出文件
分析结果
图像
最终报告
审计报告
```

不能让关键状态只存在于 prompt、chat history、临时变量或 agent 的自然语言回复里。

### 1.3 Every artifact has provenance

每个 artifact 都必须能回答：

```text
它由谁生成？
什么时候生成？
由哪些 artifact 派生？
使用了哪个 agent / tool / package / command？
用了哪些参数？
哪些参数来自用户？
哪些参数来自 agent 推断？
哪些参数来自默认值？
哪些参数来自项目配置？
哪些参数经过人工修改？
```

### 1.4 Event log is append-only

系统中发生的每个关键事件都要进入 append-only event log。

artifact 表示"产物是什么"。
event log 表示"发生过什么"。
provenance graph 表示"因果关系是什么"。

三者不能混在一起。

### 1.5 Workflow IR and Bound Workflow must be separate

Workflow IR 描述科学意图和逻辑结构。

Bound Workflow 描述具体 Molcrafts 工具、函数、CLI、参数、环境和执行后端。

不要让 agent 从自然语言直接生成最终命令。那样会让系统失去可验证性、可移植性和可审计性。

### 1.6 Defaults must be explicit

可审计系统里不能有隐式默认值。

任何参数都必须标明来源：

```text
user_provided
agent_inferred
project_default
package_default
literature_default
manual_override
runtime_detected
```

如果一个 NEMD workflow 里出现 `field_strength = 1e6 V/cm`，系统必须能说明这个值从哪里来，为什么被接受，是否需要人工确认。

### 1.7 Tests are first-class artifacts

测试不是自然语言附录，而是结构化对象。

每个 task 和 workflow 都应有测试规范，并能产生测试结果 artifact。

测试至少覆盖：

```text
schema correctness
input/output connection
dry-run execution
artifact existence
numerical tolerance
provenance completeness
regression behavior
```

### 1.8 Human review is part of the harness

可审计不等于全自动。

对于科学意图、关键参数、长时间模拟、大资源任务、不可逆写操作、最终报告，harness 应当支持人工审批点。

审批本身也必须进入 event log。

---

## 2. 系统边界

### 2.1 本系统负责什么

系统负责：

```text
理解用户实验意图
扩写成规范实验报告
抽取 workflow IR
校验 workflow IR
将 IR 绑定到 Molcrafts 工具生态
生成测试规范
执行 dry-run 或 full-run
记录所有 artifact
记录所有事件
生成 provenance graph
支持 replay / resume / audit
生成最终实验报告和审计报告
```

### 2.2 本系统不应该负责什么

系统不应该让 agent 任意完成以下事情：

```text
直接写入最终文件
直接执行 shell
直接提交 HPC 作业
直接修改项目配置
直接覆盖已有结果
直接选择未注册工具
直接使用未校验参数
直接把推断当事实
```

这些动作必须由 harness 控制。

---

## 3. 端到端流程

> **Shipped (reconciled):** the implemented pipeline is the **single 9-step
> `PlanMode`** — idea → (1) draft proposal → (2) concrete spec → (3) resolve
> capabilities → (4) workflow IR → (5) tasks + per-task tests → (6) input set →
> (7) compile/dry-run → (8) review → (9) execution report. The nine steps end
> at a descriptive execution report (never submits); real scientific execution
> is the **opt-in `--execute` tail** (`ExecuteWorkflow → GenerateFinalReport →
> ApprovalGate → GenerateAuditReport`). The earlier separate `RunMode` is
> retired/folded into that tail. The conceptual pipeline below still holds; the
> shipped stage names + the spec/capabilities/input-set steps refine it.

完整 pipeline：

```text
UserPlan
  ↓
NormalizeIntent
  ↓
ExperimentReport
  ↓
ExtractWorkflowIR
  ↓
WorkflowIR
  ↓
ValidateWorkflowIR
  ↓
BindMolcraftsTasks
  ↓
BoundWorkflow
  ↓
GenerateTestSpec
  ↓
TestSpec
  ↓
ValidateBoundWorkflowAndTests
  ↓
ApprovalGate
  ↓
DryRunExecution
  ↓
DryRunResult
  ↓
ApprovalGate
  ↓
FullExecution
  ↓
ExecutionResult
  ↓
AnalysisAndValidation
  ↓
FinalReport
  ↓
AuditReport
```

每一步都是 stage。每个 stage 都必须：

```text
读取输入 artifact
记录 stage_started event
调用 agent/tool/executor 或 validator
生成 proposal 或 result
由 harness 校验
写入输出 artifact
写入 artifact_created event
写入 provenance edge
记录 stage_completed 或 stage_failed event
```

---

## 4. 核心数据模型

### 4.1 ArtifactRef

Artifact 是系统里的基本状态单元。

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


ArtifactKind = Literal[
    "user_plan",
    "experiment_report",
    "workflow_ir",
    "bound_workflow",
    "test_spec",
    "execution_plan",
    "execution_result",
    "test_result",
    "analysis_result",
    "final_report",
    "audit_report",
    "stdout",
    "stderr",
    "log",
    "input_file",
    "output_file",
    "plot",
    "dataset",
    "checkpoint",
]


class ArtifactRef(BaseModel):
    id: str
    kind: ArtifactKind
    uri: str
    sha256: str
    created_at: datetime
    created_by: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

要求：

```text
id 全局唯一
uri 指向 artifact store 中的位置
sha256 对内容做 hash
created_by 必须明确是 user / agent:name / harness / executor:name / tool:name
parent_ids 必须记录直接来源
metadata 可以包含 package version、git commit、backend、task id 等
```

### 4.2 HarnessEvent

Event 是系统审计的时间线。

```python
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


EventType = Literal[
    "run_created",
    "run_completed",
    "run_failed",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "artifact_created",
    "artifact_validated",
    "validation_passed",
    "validation_failed",
    "agent_called",
    "agent_completed",
    "agent_failed",
    "tool_called",
    "tool_completed",
    "tool_failed",
    "task_started",
    "task_completed",
    "task_failed",
    "test_started",
    "test_completed",
    "test_failed",
    "approval_requested",
    "approval_granted",
    "approval_rejected",
    "policy_checked",
    "policy_passed",
    "policy_failed",
    "artifact_edge_created",
]


class HarnessEvent(BaseModel):
    id: str
    run_id: str
    seq: int
    type: EventType
    actor: str
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
```

要求：

```text
seq 在同一个 run 内递增
event log append-only
不能覆盖已有 event
失败也必须记录
agent prompt 和 response 可以作为 artifact 保存，event 中只放引用和摘要
```

### 4.3 ParameterValue

所有参数都必须带来源。

```python
from typing import Any, Literal
from pydantic import BaseModel


ParameterSource = Literal[
    "user_provided",
    "agent_inferred",
    "project_default",
    "package_default",
    "literature_default",
    "manual_override",
    "runtime_detected",
]


class ParameterValue(BaseModel):
    value: Any
    source: ParameterSource
    reason: str | None = None
    confidence: float | None = None
    citation: str | None = None
    approved: bool = False
```

要求：

```text
关键科学参数不能没有 source
agent_inferred 参数默认需要进入 review summary
manual_override 必须记录操作者和原因
literature_default 应该记录参考来源
package_default 必须说明来自哪个 package/version
```

### 4.4 UserPlan

用户原始计划不要被覆盖。规范化之前，先保存原始输入。

```python
class UserPlan(BaseModel):
    raw_text: str
    user_id: str | None = None
    submitted_at: datetime
    attachments: list[ArtifactRef] = []
    metadata: dict[str, Any] = {}
```

### 4.5 ExperimentReport

ExperimentReport 是人类可读的规范实验说明。

```python
class ExperimentReport(BaseModel):
    title: str
    objective: str
    background: str | None = None
    system_description: str
    scientific_hypothesis: str | None = None
    experimental_design: str
    variables: list[str] = []
    controlled_conditions: list[str] = []
    expected_outputs: list[str] = []
    assumptions: list[str] = []
    risks_or_uncertainties: list[str] = []
    user_questions: list[str] = []
```

要求：

```text
必须区分用户明确目标和 agent 补全内容
assumptions 必须在后续 IR 中可追踪
不允许把未确认推断写成事实
```

### 4.6 WorkflowIR

WorkflowIR 是科学逻辑层，不直接包含具体 shell 命令。

```python
class WorkflowIR(BaseModel):
    id: str
    name: str
    objective: str
    inputs: dict[str, ParameterValue]
    tasks: list["TaskIR"]
    edges: list["DependencyEdge"]
    expected_outputs: list["ExpectedOutput"]
    assumptions: list[str] = []
    review_flags: list[str] = []


class TaskIR(BaseModel):
    id: str
    name: str
    purpose: str
    task_type: str
    inputs: dict[str, ParameterValue]
    outputs: dict[str, str]
    constraints: dict[str, ParameterValue] = {}
    suggested_capabilities: list[str] = []
    acceptance_criteria: list[str] = []
    review_flags: list[str] = []


class DependencyEdge(BaseModel):
    source_task_id: str
    target_task_id: str
    relation: str = "requires"


class ExpectedOutput(BaseModel):
    name: str
    kind: str
    description: str
    required: bool = True
```

WorkflowIR validator 必须检查：

```text
task id 唯一
edges 引用存在
dependency graph acyclic
每个 required output 有生产 task
每个 task input 可以被用户输入或上游 output 满足
关键参数有 source
所有 agent_inferred 关键参数进入 review_flags
没有直接 shell 命令
没有未注册 capability 被当作已绑定工具
```

### 4.7 BoundWorkflow

BoundWorkflow 是执行绑定层。

```python
class BoundWorkflow(BaseModel):
    id: str
    workflow_ir_id: str
    tasks: list["BoundTask"]
    edges: list[DependencyEdge]
    execution_backend: str
    environment: "ExecutionEnvironment"
    resource_policy: "ResourcePolicy"
    review_flags: list[str] = []


class BoundTask(BaseModel):
    id: str
    ir_task_id: str
    capability_id: str
    package: str
    callable: str
    version: str | None = None
    parameters: dict[str, ParameterValue]
    inputs: dict[str, str]
    outputs: dict[str, str]
    command_template: list[str] | None = None
    side_effects: list[str] = []
    tests: list[str] = []
    provenance: dict[str, str] = {}


class ExecutionEnvironment(BaseModel):
    python_version: str | None = None
    packages: dict[str, str] = {}
    git_commit: str | None = None
    container_image: str | None = None
    env_vars: dict[str, str] = {}
    platform: str | None = None


class ResourcePolicy(BaseModel):
    backend: str
    max_runtime_s: int
    max_memory_gb: float | None = None
    max_gpu_count: int | None = None
    allowed_paths: list[str] = []
    denied_paths: list[str] = []
    allow_network: bool = False
```

BoundWorkflow validator 必须检查：

```text
capability_id 在 registry 中存在
参数符合 capability input_schema
输出符合 capability output_schema
后端支持该 capability
资源请求符合 policy
side_effects 已声明
所有 command 都由 harness 生成，不由 agent 直接自由拼接
所有输出路径位于 run workspace 内
```

### 4.8 TestSpec

测试是一等 artifact。

```python
from typing import Literal


TestKind = Literal[
    "schema_test",
    "unit_test",
    "dry_run_test",
    "integration_test",
    "regression_test",
    "numerical_tolerance_test",
    "artifact_existence_test",
    "provenance_test",
    "resource_policy_test",
]


class TestSpec(BaseModel):
    id: str
    name: str
    kind: TestKind
    target_task_id: str | None = None
    target_workflow_id: str | None = None
    description: str
    inputs: dict[str, ParameterValue] = {}
    command: list[str] | None = None
    expected_artifacts: list[str] = []
    expected_metrics: dict[str, ParameterValue] = {}
    tolerance: dict[str, float] = {}
    required: bool = True


class TestResult(BaseModel):
    id: str
    test_spec_id: str
    status: Literal["passed", "failed", "skipped", "error"]
    metrics: dict[str, float] = {}
    produced_artifacts: list[ArtifactRef] = []
    stdout: ArtifactRef | None = None
    stderr: ArtifactRef | None = None
    reason: str | None = None
```

最低测试要求：

```text
每个 WorkflowIR 必须有 schema_test
每个 BoundWorkflow 必须有 binding validation test
每个 executable workflow 必须有 dry_run_test
每个关键输出必须有 artifact_existence_test
每个数值分析任务必须有 numerical_tolerance_test 或明确说明不能做
每个最终结果必须有 provenance_test
```

---

## 5. Capability Registry

### 5.1 为什么需要 registry

Agent 不能凭空猜 Molcrafts API。

错误设计：

```python
molpy.build_polymer(...)
molpack.pack(...)
molvis.plot(...)
```

如果这些函数不存在，系统应该在 binding 阶段失败，而不是等执行阶段炸掉。

正确设计是让 agent 从 capability registry 中选择工具。

### 5.2 Capability 数据模型

```python
class ToolCapability(BaseModel):
    id: str
    package: str
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    callable_path: str | None = None
    cli_template: list[str] | None = None
    side_effects: list[str] = []
    supported_backends: list[str] = ["local"]
    examples: list[dict] = []
    version: str | None = None
    tags: list[str] = []
```

Registry 来源：

```text
Python function signatures
Pydantic models
Typer CLI schemas
MolExp task definitions
MolPack CLI schemas
MolVis plotting functions
MolQ backend definitions
project-local adapters
MCP/code graph discovery results
```

### 5.3 Registry 接口

```python
class CapabilityRegistry:
    def list_capabilities(self) -> list[ToolCapability]: ...
    def get(self, capability_id: str) -> ToolCapability: ...
    def search(self, query: str, tags: list[str] | None = None) -> list[ToolCapability]: ...
    def validate_call(self, capability_id: str, parameters: dict) -> None: ...
```

### 5.4 Binding 原则

Agent 可以推荐 capability。

Harness 必须检查：

```text
capability 是否存在
版本是否兼容
schema 是否匹配
输入 artifact 是否存在
输出 artifact 是否声明
side effects 是否允许
执行后端是否支持
是否需要人工审批
```

---

## 6. Executor 设计

### 6.1 统一执行接口

所有副作用都必须通过 executor。

```python
class CommandSpec(BaseModel):
    cmd: list[str]
    cwd: str
    env: dict[str, str] = {}
    timeout_s: int = 3600
    expected_outputs: list[str] = []
    metadata: dict[str, str] = {}


class CommandResult(BaseModel):
    exit_code: int
    started_at: datetime
    ended_at: datetime
    stdout_artifact: ArtifactRef
    stderr_artifact: ArtifactRef
    output_artifacts: list[ArtifactRef]
    metadata: dict[str, str] = {}
```

### 6.2 LocalExecutor

LocalExecutor 负责本地执行。

必须实现：

```text
workspace 隔离
cwd 限制
环境变量过滤
timeout
stdout/stderr 捕获
输出文件扫描
文件 hash
exit code 记录
失败事件记录
```

### 6.3 SlurmExecutor

SlurmExecutor 负责 HPC 提交。

必须实现：

```text
生成 sbatch 脚本 artifact
记录 sbatch 命令
记录 job id
记录 submission time
记录 queue status
记录 output/error 文件
记录 resource request
记录 module/environment 信息
支持 job polling
支持 cancellation
支持 resume 查询
```

### 6.4 Executor 不应该做什么

Executor 不应该理解科学逻辑。

它只负责安全、可追踪、可回放地执行命令。

科学 task 如何变成 command，是 BoundTask adapter 的职责。

---

## 7. Policy 设计

### 7.1 Policy 类型

系统至少需要四类 policy。

```text
PathPolicy
ToolPolicy
ResourcePolicy
ApprovalPolicy
```

### 7.2 PathPolicy

控制路径访问。

```python
class PathPolicy(BaseModel):
    workspace_root: str
    allowed_read_paths: list[str]
    allowed_write_paths: list[str]
    denied_paths: list[str] = ["/", "/etc", "/usr", "~/.ssh"]
```

### 7.3 ToolPolicy

控制可执行工具。

```python
class ToolPolicy(BaseModel):
    allowed_commands: list[str]
    denied_commands: list[str] = ["rm -rf", "sudo", "chmod -R 777"]
    allow_network: bool = False
    max_runtime_s: int = 3600
    max_output_mb: int = 1024
```

注意：`denied_commands` 不能只用字符串匹配。后续应升级为命令 AST 或 token-level policy。

### 7.4 ApprovalPolicy

控制哪些步骤需要人工确认。

```python
class ApprovalPolicy(BaseModel):
    require_for_agent_inferred_scientific_parameters: bool = True
    require_for_full_execution: bool = True
    require_for_hpc_submission: bool = True
    require_for_large_resource_request: bool = True
    require_for_overwrite: bool = True
    require_for_final_report: bool = True
```

### 7.5 必须触发 approval 的场景

```text
关键科学参数由 agent 推断
任务需要大量计算资源
任务会提交 HPC
任务会覆盖已有 artifact
workflow 包含未验证 custom script
workflow 从 dry-run 进入 full-run
最终报告将作为正式结果输出
```

---

## 8. Stage 和 Orchestrator

### 8.1 Stage 抽象

```python
class Stage:
    name: str

    async def run(self, context: "RunContext") -> ArtifactRef:
        raise NotImplementedError
```

### 8.2 RunContext

```python
class RunContext(BaseModel):
    run_id: str
    workspace: str
    artifact_store: object
    event_log: object
    provenance_store: object
    capability_registry: object
    agent_gateway: object
    executor: object
    policy_engine: object
```

### 8.3 StageRunner

所有 stage 必须通过 StageRunner 执行。

```python
class StageRunner:
    def __init__(self, context: RunContext):
        self.context = context

    async def run_stage(self, stage: Stage) -> ArtifactRef:
        self.context.event_log.append(
            run_id=self.context.run_id,
            type="stage_started",
            actor="harness",
            payload={"stage": stage.name},
        )

        try:
            artifact = await stage.run(self.context)
            self.context.event_log.append(
                run_id=self.context.run_id,
                type="stage_completed",
                actor="harness",
                payload={"stage": stage.name},
                artifact_ids=[artifact.id],
            )
            return artifact
        except Exception as exc:
            self.context.event_log.append(
                run_id=self.context.run_id,
                type="stage_failed",
                actor="harness",
                payload={"stage": stage.name, "error": repr(exc)},
            )
            raise
```

### 8.4 标准 Stage 列表

```text
CreateRun
SaveUserPlan
GenerateExperimentReport
ReviewExperimentReport
ExtractWorkflowIR
ValidateWorkflowIR
RepairWorkflowIR
BindMolcraftsTasks
ValidateBoundWorkflow
GenerateTestSpec
ValidateTestSpec
ApprovalBeforeDryRun
ExecuteDryRun
EvaluateDryRun
ApprovalBeforeFullRun
ExecuteFullRun
AnalyzeResults
RunValidationTests
GenerateFinalReport
GenerateAuditReport
CloseRun
```

### 8.5 Repair 机制

Repair 也必须可追踪。

错误设计：

```text
validator 失败 → 重新问 agent → 覆盖原文件
```

正确设计：

```text
workflow_ir_v1 artifact
validation_failed event
repair_prompt artifact
workflow_ir_v2 artifact
parent_ids = [workflow_ir_v1, validation_report]
validation_passed event
```

---

## 9. Storage 设计

### 9.1 最小可行存储

初期建议：

```text
filesystem artifact store
SQLite event log
SQLite provenance graph
```

不要一开始上复杂图数据库。

### 9.2 文件布局

```text
.runs/
  run_2026_05_26_xxxxx/
    artifacts/
      user_plan/
      experiment_report/
      workflow_ir/
      bound_workflow/
      test_spec/
      execution_result/
      final_report/
      audit_report/
      stdout/
      stderr/
      plots/
      datasets/
    workspace/
    logs/
    events.sqlite
    provenance.sqlite
    manifest.json
```

### 9.3 SQLite schema

```sql
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE artifact_edges (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    artifact_ids_json TEXT NOT NULL
);

CREATE UNIQUE INDEX idx_events_run_seq ON events(run_id, seq);
```

### 9.4 ArtifactStore 接口

```python
class ArtifactStore:
    def put_json(self, kind: str, obj: dict, created_by: str, parent_ids: list[str]) -> ArtifactRef: ...
    def put_text(self, kind: str, text: str, created_by: str, parent_ids: list[str]) -> ArtifactRef: ...
    def put_file(self, kind: str, path: str, created_by: str, parent_ids: list[str]) -> ArtifactRef: ...
    def get(self, artifact_id: str) -> bytes: ...
    def get_ref(self, artifact_id: str) -> ArtifactRef: ...
    def list_by_kind(self, kind: str) -> list[ArtifactRef]: ...
    def latest_by_kind(self, kind: str) -> ArtifactRef: ...
```

### 9.5 EventLog 接口

```python
class EventLog:
    def append(self, run_id: str, type: str, actor: str, payload: dict, artifact_ids: list[str] | None = None) -> HarnessEvent: ...
    def list_events(self, run_id: str) -> list[HarnessEvent]: ...
    def get_timeline(self, run_id: str) -> list[HarnessEvent]: ...
```

### 9.6 ProvenanceStore 接口

```python
class ProvenanceStore:
    def add_edge(self, parent_id: str, child_id: str, relation: str) -> None: ...
    def trace_backward(self, artifact_id: str) -> list[ArtifactRef]: ...
    def trace_forward(self, artifact_id: str) -> list[ArtifactRef]: ...
    def lineage_graph(self, artifact_id: str) -> dict: ...
```

---

## 10. Agent Gateway

### 10.1 统一 agent 调用接口

```python
class AgentCallSpec(BaseModel):
    agent_name: str
    input_artifact_ids: list[str]
    prompt_artifact_id: str | None = None
    output_schema: dict
    temperature: float = 0.2
    metadata: dict[str, str] = {}


class AgentCallResult(BaseModel):
    output_artifact: ArtifactRef
    raw_response_artifact: ArtifactRef
    model: str
    usage: dict[str, int] = {}
```

### 10.2 Agent 调用必须记录

每次 agent 调用必须记录：

```text
agent name
model name
input artifacts
prompt artifact
raw response artifact
parsed output artifact
schema validation result
usage metadata
```

### 10.3 Agent 不允许返回自由副作用

Agent 不应该返回：

```text
我已经运行了命令
我已经修改了文件
我已经提交了作业
```

Agent 只能返回结构化 proposal。

---

## 11. Validation 体系

### 11.1 Validator 类型

```text
SchemaValidator
WorkflowGraphValidator
ParameterSourceValidator
CapabilityBindingValidator
PolicyValidator
TestSpecValidator
ProvenanceValidator
ExecutionResultValidator
```

### 11.2 WorkflowIR Validator

必须检查：

```text
Pydantic schema 合法
task id 唯一
edge 合法
DAG 无环
required output 有生产者
required input 有来源
关键参数 source 合法
agent_inferred 参数进入 review_flags
没有 shell command
没有具体 backend 细节泄漏到 IR
```

### 11.3 BoundWorkflow Validator

必须检查：

```text
每个 task 对应一个 IR task
每个 capability 存在
参数匹配 capability schema
输入输出 artifact 引用合法
backend 支持 capability
resource policy 合法
输出路径在 workspace 内
side effects 被声明
```

### 11.4 TestSpec Validator

必须检查：

```text
每个 required task 有测试或明确豁免
每个 expected output 有 artifact existence test
数值任务有 tolerance 或明确说明
provenance test 存在
测试命令不越权
测试输出路径合法
```

### 11.5 Provenance Validator

必须检查：

```text
每个 final artifact 可回溯到 user_plan
每个 execution result 可回溯到 bound workflow
每个 bound task 可回溯到 workflow IR task
每个 parameter 有 source
每个 manual override 有 approval event
每个 agent_inferred critical parameter 有 review event 或显式 waiver
```

---

## 12. Audit Report

### 12.1 AuditReport 内容

AuditReport 不是普通总结，而是审计入口。

必须包括：

```text
Run ID
用户原始计划 artifact
最终报告 artifact
workflow IR artifact
bound workflow artifact
test spec artifact
执行结果 artifact
所有关键参数来源
所有 agent-inferred 参数
所有人工审批记录
所有失败和 repair 记录
所有工具和 package 版本
所有命令摘要
所有输出 artifact hash
provenance graph 摘要
测试结果摘要
已知限制和未解决问题
```

### 12.2 AuditReport schema

```python
class AuditReport(BaseModel):
    run_id: str
    summary: str
    root_artifact_id: str
    final_artifact_ids: list[str]
    critical_parameters: dict[str, ParameterValue]
    approvals: list[dict]
    validation_results: list[dict]
    test_results: list[str]
    failures: list[dict]
    repairs: list[dict]
    software_versions: dict[str, str]
    command_summaries: list[dict]
    provenance_summary: dict
    limitations: list[str]
```

---

## 13. CLI 设计

### 13.1 基础命令

```text
molharness init
molharness run plan.txt
molharness inspect <run_id>
molharness timeline <run_id>
molharness artifacts <run_id>
molharness trace <artifact_id>
molharness approve <approval_id>
molharness resume <run_id>
molharness replay <run_id>
molharness validate <artifact_id>
molharness export-audit <run_id>
```

### 13.2 inspect 输出

```text
Run status
Current stage
Last event
Artifacts summary
Pending approvals
Failed validations
Test summary
```

### 13.3 trace 输出

`molharness trace mobility.json` 应输出：

```text
mobility.json
  derived from analysis_task_result
  derived from trajectory.dcd
  derived from nemd_task
  derived from bound_workflow
  derived from workflow_ir
  derived from experiment_report
  derived from user_plan
```

### 13.4 replay 输出

Replay 不一定重新执行昂贵模拟。

应支持三种模式：

```text
metadata replay: 只重放 event/artifact lineage
validation replay: 重新跑 schema/policy/provenance tests
execution replay: 重新执行 selected tasks
```

---

## 14. 推荐代码结构

```text
molcrafts_harness/
  core/
    artifact.py
    event.py
    provenance.py
    run.py
    stage.py
    context.py

  schemas/
    user_plan.py
    experiment_report.py
    workflow_ir.py
    bound_workflow.py
    test_spec.py
    audit_report.py
    parameters.py

  store/
    artifact_store.py
    file_artifact_store.py
    event_log.py
    sqlite_event_log.py
    provenance_store.py
    sqlite_provenance_store.py

  agents/
    gateway.py
    prompts/
      report_writer.md
      ir_extractor.md
      task_binder.md
      test_writer.md
      final_report_writer.md
    report_writer.py
    ir_extractor.py
    task_binder.py
    test_writer.py

  registry/
    capability.py
    registry.py
    python_signature_loader.py
    typer_loader.py
    molpy.py
    molpack.py
    molexp.py
    molvis.py
    molq.py

  validators/
    base.py
    schema_validator.py
    workflow_graph_validator.py
    parameter_source_validator.py
    capability_binding_validator.py
    policy_validator.py
    test_spec_validator.py
    provenance_validator.py

  policies/
    path_policy.py
    tool_policy.py
    resource_policy.py
    approval_policy.py
    policy_engine.py

  executors/
    base.py
    command_spec.py
    local_executor.py
    slurm_executor.py
    result_collector.py

  stages/
    save_user_plan.py
    generate_experiment_report.py
    extract_workflow_ir.py
    validate_workflow_ir.py
    bind_molcrafts_tasks.py
    generate_test_spec.py
    validate_test_spec.py
    approval_gate.py
    execute_dry_run.py
    execute_full_run.py
    analyze_results.py
    generate_final_report.py
    generate_audit_report.py

  engine/
    orchestrator.py
    scheduler.py
    resume.py
    replay.py
    repair.py

  cli/
    main.py
    commands_run.py
    commands_inspect.py
    commands_trace.py
    commands_approve.py

  adapters/
    molpy_adapter.py
    molpack_adapter.py
    molexp_adapter.py
    molvis_adapter.py
    molq_adapter.py

  tests/
    unit/
    integration/
    fixtures/
```

关键分层：

```text
schemas 只定义数据结构
store 只负责持久化
agents 只负责 agent 调用
registry 只负责能力发现和工具描述
validators 只负责判断是否合法
executors 只负责受控执行
stages 只负责编排一个语义步骤
engine 负责编排整个 run
cli 只是用户界面
```

---

## 15. 现有代码最可能存在的缺憾检查表

因为还没有看到你的代码，这里按完全体 harness 反推最常见缺口。

### 15.1 如果你的代码现在是 prompt pipeline

症状：

```text
多个 agent 函数串起来
上一步输出字符串直接传给下一步
中间结果没有 artifact id
没有 hash
没有 event log
失败后覆盖旧结果
没有参数来源
没有 approval gate
```

需要重构：

```text
引入 ArtifactStore
引入 EventLog
每一步变成 Stage
每个 Stage 输入输出都是 ArtifactRef
agent output 先变成 proposal，再由 harness validate/store
```

### 15.2 如果 workflow IR 已经有了，但 BoundWorkflow 没分开

症状：

```text
IR 里直接出现 shell command
IR 里直接出现 slurm 参数
IR 里直接出现具体 Python 函数调用
IR 既描述科学任务又描述执行细节
```

需要重构：

```text
WorkflowIR 保持科学语义
BoundWorkflow 负责工具绑定
ExecutionPlan 负责命令和资源
```

### 15.3 如果有日志但没有 event sourcing

症状：

```text
使用 logging.info
只有文本日志
无法 query 某个 run 的所有事件
无法知道某个 artifact 怎么生成
无法按 seq 重放
```

需要重构：

```text
保留普通 log，但新增结构化 HarnessEvent
所有 stage/tool/agent/test/approval 都 append event
```

### 15.4 如果有文件输出但没有 provenance

症状：

```text
文件放在 output/ 目录
文件名包含时间戳
没有 parent-child 关系
不知道哪个输入生成哪个输出
```

需要重构：

```text
每个输出注册为 ArtifactRef
记录 parent_ids
写入 artifact_edges
提供 trace_backward/trace_forward
```

### 15.5 如果 agent 能直接执行工具

症状：

```text
agent 决定运行什么命令
agent 拼 shell string
agent 读写任意路径
agent 失败后自己 retry
```

需要重构：

```text
agent 只能返回 CommandProposal 或 BoundTask proposal
harness policy check 后由 Executor 执行
```

### 15.6 如果测试只是自然语言

症状：

```text
最终报告里写"建议测试 xxx"
没有 TestSpec schema
没有 TestResult artifact
测试不进入 event log
```

需要重构：

```text
把测试结构化
测试绑定 target_task_id 或 target_workflow_id
测试结果作为 artifact 保存
测试事件进入 event log
```

### 15.7 如果参数没有 source

症状：

```text
参数只是普通 dict
不知道参数是用户给的还是 agent 猜的
默认值在代码里静默填充
```

需要重构：

```text
所有重要参数改成 ParameterValue
默认值显式化
agent_inferred 参数进入 review_flags
```

### 15.8 如果 registry 不存在

症状：

```text
agent 根据名字猜 Molcrafts API
binding 阶段没有 schema 检查
直到执行时才发现函数不存在
```

需要重构：

```text
建立 CapabilityRegistry
用 Pydantic schema / signature / Typer metadata 描述工具
BoundTask 只能引用 capability_id
```

### 15.9 如果执行环境没有记录

症状：

```text
结果能生成，但不知道 Python/package/git/container/slurm 信息
```

需要重构：

```text
ExecutionEnvironment artifact
记录 package versions
记录 git commit
记录 command
记录 env 摘要
记录 backend metadata
```

### 15.10 如果没有 resume/replay

症状：

```text
中途失败只能重跑
不能从某个 stage 继续
不能重放 validation
不能重建 audit timeline
```

需要重构：

```text
Stage 幂等化
输出 artifact 不覆盖
run state 从 event log 推导
支持 resume from failed stage
支持 metadata replay 和 validation replay
```

---

## 16. 重构路线

### Phase 1：把状态抓住

先不要动 agent 能力。先把状态从内存里拿出来。

要做：

```text
ArtifactRef
FileArtifactStore
HarnessEvent
SQLiteEventLog
ProvenanceStore
RunContext
StageRunner
```

完成标准：

```text
任何一次 run 都能看到 timeline
任何中间输出都有 artifact id 和 hash
任何最终输出都能 trace 回 user plan
```

### Phase 2：把 pipeline 变成 stage

把现有函数重构成 stage。

例如原来：

```python
report = write_report(user_text)
ir = extract_ir(report)
bound = bind_tasks(ir)
```

改成：

```python
await runner.run_stage(SaveUserPlan())
await runner.run_stage(GenerateExperimentReport())
await runner.run_stage(ExtractWorkflowIR())
await runner.run_stage(BindMolcraftsTasks())
```

完成标准：

```text
所有 stage 有输入 artifact 和输出 artifact
所有失败会写 stage_failed event
所有 repair 生成新 artifact，不覆盖旧 artifact
```

### Phase 3：拆分 IR 和 BoundWorkflow

如果现在混在一起，必须拆。

要做：

```text
WorkflowIR schema
BoundWorkflow schema
TaskIR
BoundTask
DependencyEdge
ExpectedOutput
```

完成标准：

```text
WorkflowIR 不包含 shell/backend/package 细节
BoundWorkflow 不丢失 IR task 的来源
每个 BoundTask 都有 ir_task_id
```

### Phase 4：引入 capability registry

要做：

```text
ToolCapability schema
CapabilityRegistry interface
MolPy adapter
MolPack adapter
MolExp adapter
MolVis adapter
MolQ adapter
```

完成标准：

```text
BoundTask 只能引用 capability_id
非法 capability 在 validation 阶段失败
参数 schema mismatch 在 validation 阶段失败
```

### Phase 5：引入测试规范

要做：

```text
TestSpec
TestResult
TestSpecValidator
DryRunExecutor
ProvenanceValidator
```

完成标准：

```text
每个 workflow 有 dry-run test
每个 expected output 有 existence test
最终报告有 provenance test
测试结果作为 artifact 保存
```

### Phase 6：引入 policy 和 approval

要做：

```text
PathPolicy
ToolPolicy
ResourcePolicy
ApprovalPolicy
ApprovalRequest
ApprovalEvent
```

完成标准：

```text
HPC/full-run/overwrite/agent-inferred critical parameter 都能触发 approval
approval/rejection 进入 event log
```

### Phase 7：支持 replay/resume/audit

要做：

```text
Run state reconstruction
Resume engine
Replay engine
AuditReport generator
CLI inspect/timeline/trace
```

完成标准：

```text
失败后可 resume
任意 artifact 可 trace
任意 run 可导出 audit report
```

---

## 17. 最小完全体验收标准

一个 run 结束后，系统必须能回答以下问题。

### 17.1 科学意图

```text
用户最初说了什么？
系统扩写了哪些内容？
哪些是用户明确给的？
哪些是 agent 推断的？
哪些地方需要人工确认？
```

### 17.2 Workflow

```text
workflow 有哪些 task？
task 之间如何依赖？
每个 task 的输入来自哪里？
每个 task 的输出给谁用？
DAG 是否合法？
```

### 17.3 工具绑定

```text
每个 task 绑定到了哪个 Molcrafts capability？
为什么选择这个 capability？
参数是否符合 schema？
该 capability 来自哪个 package/version？
```

### 17.4 执行

```text
执行了哪些命令？
在哪个目录执行？
环境是什么？
stdout/stderr 在哪里？
exit code 是什么？
生成了哪些文件？
文件 hash 是什么？
```

### 17.5 测试

```text
跑了哪些测试？
哪些通过？
哪些失败？
失败原因是什么？
数值容差是什么？
是否有 dry-run？
```

### 17.6 审计

```text
最终结果能否追溯到用户原始计划？
所有关键参数是否有来源？
所有人工修改是否有记录？
所有失败和 repair 是否保留？
是否可以重放 validation？
是否可以从失败处 resume？
```

如果这些问题回答不了，就还不是完全意义上的 harness。

---

## 18. 反模式

### 18.1 把 prompt 当架构

Prompt 可以改进输出质量，但不能承担系统可靠性。

如果系统依赖"让 agent 自觉写清楚"，那就不是 harness。

### 18.2 把日志当 provenance

普通日志不能替代 provenance。

日志是时间文本。provenance 是 artifact 之间的因果图。

### 18.3 把最终报告当唯一产物

最终报告是人看的，不是系统状态。

真正的系统状态是 artifact + event + provenance。

### 18.4 让 agent 写最终命令

Agent 可以建议工具，但不能直接拼接和执行最终命令。

最终命令必须由 BoundTask adapter 和 Executor 生成。

### 18.5 隐式默认值

隐式默认值会破坏科学可追踪性。

默认值可以存在，但必须显式进入 ParameterValue。

### 18.6 覆盖旧结果

任何 repair、rerun、manual override 都应该生成新 artifact。

不要覆盖旧 artifact。

---

## 19. 推荐开发顺序

最务实的顺序：

```text
1. 定义 schemas
2. 实现 FileArtifactStore
3. 实现 SQLiteEventLog
4. 实现 StageRunner
5. 把现有 pipeline 包成 stages
6. 加 WorkflowIR validator
7. 加 BoundWorkflow validator
8. 加 TestSpec/TestResult
9. 加 LocalExecutor
10. 加 AuditReport
11. 加 CapabilityRegistry
12. 加 ApprovalGate
13. 加 SlurmExecutor
14. 加 replay/resume
15. 加 CLI 和 dashboard
```

先做 core，不要先做 UI。

没有 artifact/event/provenance 的 UI 只是漂亮壳子。

---

## 20. 对现有代码的重构判断标准

你可以用下面这几个问题直接审查你的代码。

### 20.1 State ownership

```text
系统状态是 harness 拥有，还是 agent/prompt/function call 临时拥有？
```

如果状态在 agent 输出字符串里，重构。

### 20.2 Side-effect ownership

```text
副作用是 harness executor 执行，还是 agent/tool 随便执行？
```

如果 agent 可以直接执行 shell，重构。

### 20.3 Schema ownership

```text
数据结构由 Pydantic/schema 约束，还是靠自然语言约定？
```

如果靠自然语言，重构。

### 20.4 Tool ownership

```text
工具来自 registry，还是 agent 猜函数名？
```

如果 agent 猜，重构。

### 20.5 Provenance ownership

```text
最终结果能否自动 trace 到 user plan？
```

如果不能，重构。

### 20.6 Test ownership

```text
测试是结构化 TestSpec，还是报告里的一段建议？
```

如果只是建议，重构。

---

## 21. 第一版可以牺牲什么

为了快速落地，第一版可以暂时不做：

```text
复杂 dashboard
分布式执行
复杂图数据库
自动代码修复
多用户权限系统
完整容器化
完整 HPC lifecycle
复杂语义搜索
```

但第一版不能牺牲：

```text
ArtifactRef
EventLog
Provenance edge
WorkflowIR / BoundWorkflow 分离
Parameter source
StageRunner
Validation
TestSpec
AuditReport
```

这些是 harness 的骨架。

---

## 22. 结论

你的系统要成为完全意义上的 harness，关键不是让 agent 更强，而是把 agent 降级为 proposal generator。

真正的中心应该是：

```text
ArtifactStore
EventLog
ProvenanceStore
WorkflowIR
BoundWorkflow
CapabilityRegistry
Validator
Executor
TestSpec
AuditReport
```

系统的硬边界是：

```text
agent 不能拥有状态
agent 不能拥有副作用
agent 不能绕过 schema
agent 不能绕过 policy
agent 不能绕过 provenance
```

只要这些边界守住，Molcrafts Harness 就会从一个高级 prompt pipeline 变成真正的科学工作流基础设施。
