# 统一 pydantic-graph 调度: Sweep 与 Backend 并入外层图

**状态**: Draft · **作者**: @RoyKid · **日期**: 2026-04-17

## 实施状态

| Phase | 状态 | 落地日期 | 备注 |
|-------|------|---------|------|
| 1 — Sweep pydantic-graph 化(仅 `--local` + `-j N`) | ✅ 已完成 | 2026-04-17 | 新增 `molexp.sweep` 包、`-j/--jobs` CLI flag、profile `jobs:` 字段约定;`molexp run --local -j N` 启用 sweep 级并发 |
| 2 — Backend 统一入口(`--backend`, `--block`) | ⏳ 待开始 | — | 需要先让 `molq.JobHandle` 支持 `wait_async` 或包 `asyncio.to_thread` |
| 3 — Per-node backend(`@wf.task(backend=..., resources=...)`) | ⏳ 待开始 | — | 依赖 Phase 2 |
| 4 — molq 原生 async API | ⏳ 独立推进 | — | 在 molq 仓完成 `monitor.py` async 化 |

**Phase 1 变更清单**:
- `src/molexp/sweep/__init__.py`、`src/molexp/sweep/graph.py`(`SweepReplica`, `SweepState`, `SweepRoot`, `run_sweep`)
- `src/molexp/cli/run_cmd.py` 拆分 `_execute_sweep`:`_discover_runs` + `_dispatch_local`(pydantic-graph)+ `_dispatch_cluster`(保留旧路径);新增 `-j/--jobs` CLI 参数和 `_resolve_jobs` 辅助
- `tests/test_sweep/test_graph.py`(9 个单测)+ `tests/test_cli_run_jobs.py`(6 个 E2E 测试)

**当前 profile 约定**:`jobs: <int>` 是用户数据(schemaless,见 §2 及 `docs/spec/molcfg-profiles.md`)。CLI `-j` 优先级高于 profile,profile 值 > 1 触发并行,默认 1(向后兼容)。

## 1. 动机

当前 `molexp run` 存在两层完全独立的调度:

| 层次 | 位置 | 并行模型 | 快照/恢复 |
|------|------|----------|-----------|
| **Sweep**(跨 experiment × replica) | `cli/run_cmd.py:_execute_sweep` 的 `for` 循环 | `--local` 强串行;`--slurm` 交给外部调度器 | 仅靠 run-id 去 store 里匹配,无 in-process 状态 |
| **Workflow**(task DAG) | `workflow/_pydantic_graph/` | 同 level 内 `asyncio.gather` | 有 pydantic-graph snapshot |

这带来四个问题:

1. **`--local` 不能并行**: 明明多个 experiment 互相无依赖,`for` 循环 + 阻塞
   `asyncio.run()` 强制串行(见 `cli/run_cmd.py:206-209, 467-472`)。
2. **两套 dispatch 路径**: `_local_handler` 和 `SubmitHandler` 是两套代码,同样的
   "把一个 run 跑起来"有两个完全不同的分支(`run_cmd.py:452-482` vs
   `run_cmd.py:484-543`)。
3. **`--slurm` fire-and-forget 不可观测**: 父进程提交完立即退出,sweep 层没有
   graph 状态,`molexp watch` 必须单独实现数据库轮询逻辑来反推进度。
4. **无法 per-node 选后端**: 想让 `prepare_data` 跑在本地、`train_model` 上 GPU
   集群是常见需求,当前架构不支持——`--slurm` 是整个脚本层级的开关。

**目标**: 把"sweep"也建模成一个 pydantic-graph,让 local / molq / 未来的其他
backend 都成为**某个 node 内部的 await 操作**,而不是顶层的两种 dispatch。

**非目标**:
- 不改 molq 内部模型(job store / reconciler / monitor 保持不变)。
- 不改内层 workflow 的 task 协议(`@wf.task` 语义保持)。
- 不为本地引入子进程池——local 就是 in-process async 并发。GPU 训练的真并行
  交给 molq 的 local scheduler 或远程 scheduler。

## 2. 设计原则

1. **pydantic-graph 是唯一编排器**: 所有执行语义——串/并、本地/远程、
   snapshot/resume——都通过 graph 节点 + async body 表达。不存在 graph 之外的
   调度循环。
2. **Backend 是 node body 的实现细节**: `@wf.task` 不关心自己跑在哪;跑哪儿由
   decorator 参数、profile 配置和 CLI 覆盖在运行时决定。
3. **默认零改动**: 现有脚本(如 `examples/train_allegro_qm9.py`)**不需要任何
   源码修改**即可在新架构下运行。所有新能力通过 profile / decorator 参数开启。
4. **Block 语义由用户控制**: `block=True` 节点 `await` 到作业终止,父进程必须
   存活;`block=False` 节点返回 `job_id` 立刻完成,后续状态交给 molq store。
5. **Account/resources 三级继承**: profile 兜底 → decorator 默认 → CLI/ctx
   覆盖。用户只在"特殊"节点显式覆盖。
6. **旧 CLI flag 作为别名保留**: `--local` / `--slurm` / `--pbs` / `--lsf` 映射
   到新的 `--backend` + profile,不做 deprecation 期强制迁移。

## 3. 核心架构

### 3.1 两层 Graph

```
SweepGraph(外层)
  节点: 每个 (experiment × replica)
  边:   无(所有 replica 互相独立)
  并发: asyncio.Semaphore(jobs) 节流同 level 并发
  Body: await _run_one_replica(run, exp, profile_cfg)

       ├─ backend=local:
       │    inner = exp.workflow.execute(run_context=ctx)
       │    await inner                              # 进程内直跑
       │
       └─ backend=slurm/pbs/lsf:
            handle = submitor.submit(
                argv=[python, -m, worker, script, run_dir, --task <name>?],
                resources=..., execution=...)
            if block:
                record = await handle.wait_async()   # 父进程守候
                return {"state": record.state.name, "job_id": handle.job_id}
            else:
                return {"job_id": handle.job_id}     # fire-and-forget

WorkflowGraph(内层,保持现状)
  节点: @wf.task
  边:   depends_on
  Body: 保持现有 async def 协议
```

### 3.2 Per-node Backend(进阶路径)

默认"一个 experiment 一个远程 job"(整个内层 workflow 打包提交)。当用户希望
混合执行(e.g. `prepare_data` 本地、`train_model` 上 GPU),在内层某个 `@wf.task`
上标注 `backend=`,框架把这个 task 作为独立 molq job 提交:

```python
@wf.task(depends_on=["prepare_data"],
         backend="slurm",
         resources=TaskResources(gpus=1, time="12h", mem="64G"))
async def train_model(ctx): ...
```

远端用同一份 `worker.py` 入口,加 `--task <name>` 参数来选择"只跑一个 node"
而不是整个 workflow(见 §5.3)。

### 3.3 并发度控制

```
molexp run script.py -j 4                    # sweep 级 4 并发
molexp run script.py -j 4 --backend slurm    # 4 路同时提交 slurm + await
molexp run script.py                         # -j 默认 1(= 当前行为)
```

`-j N` 的实现就是外层 `SweepGraph` 节点执行器里的
`asyncio.Semaphore(N)`——直接复用 `workflow/_pydantic_graph/node.py:88-124` 的
`_execute_parallel` 模式。

## 4. 用户 API

### 4.1 CLI

```bash
# 推荐新写法
molexp run script.py --backend {local,slurm,pbs,lsf,...} \
                     [-j N] [--block/--no-block] \
                     [-c molcfg.yaml] [--profile NAME]

# 旧 flag 保留为别名(zero-cost 迁移)
molexp run script.py --local        # ≡ --backend local -j 1
molexp run script.py --slurm        # ≡ --backend slurm --no-block (保持 fire-and-forget 行为)
```

**默认值**:
- `--backend`: `local`
- `-j`: `1`(向后兼容,显式设 `-j auto` = CPU 核数)
- `--block`: `local` 下无意义(本就是 await);`slurm` 下默认 `--no-block`
  匹配旧行为。未来版本可切换成 `--block` 默认,按社区反馈。

### 4.2 Profile(molcfg.yaml)

```yaml
default: &base
  smoke: false

profiles:
  local_quick:
    <<: *base
    backend: local
    jobs: 4                        # -j 等价写法

  prod_gpu:
    <<: *base
    backend: slurm
    block: true                    # 父进程守候
    jobs: 8                        # 同时提交/守候 8 个 job
    slurm:
      cluster: "perlmutter"
      account: "mycompchem"
      qos: "regular"
      partition: "gpu"

  mixed:
    <<: *base
    backend: local                 # 外层 sweep 本地并发
    jobs: 16
    per_node_backend: true         # 允许内层 task 上的 backend= 生效
    slurm:                         # 内层 task 用到时继承
      cluster: "perlmutter"
      account: "mycompchem"
```

### 4.3 Decorator(脚本内)

```python
from molexp import TaskResources

@wf.task(
    depends_on=["prepare_data"],
    backend="slurm",                                        # per-node 覆盖
    resources=TaskResources(gpus=1, time="12h", mem="64G"),
)
async def train_model(ctx): ...
```

**决策链**: CLI 覆盖 > profile > decorator > 框架默认。`account` / `cluster` /
`qos` 这类机构级字段**永远**从 profile 取,不进脚本(便于跨机器复用)。

### 4.4 TaskContext 扩展

```python
@wf.task
async def some_task(ctx: TaskContext):
    # 新增: 手动提交(高阶用户)
    handle = await ctx.molq.submit(
        argv=[...], resources=..., execution=...)
    record = await handle.wait_async()
    ...
```

`ctx.molq` 在非 molq 后端下可用(fallback 到 `local` scheduler),使用户代码不
依赖 CLI 选的 backend。

## 5. 内部实现

### 5.1 SweepGraph

新文件 `molexp/sweep/graph.py`:

```python
from pydantic_graph import BaseNode, End, GraphRunContext
import asyncio

@dataclass
class ReplicaNode(BaseNode[SweepState, SweepDeps, SweepResult]):
    mol_run: Run
    experiment: Experiment

    async def run(self, ctx) -> "ReplicaNode | End[SweepResult]":
        async with ctx.deps.semaphore:             # -j N 节流
            backend = _resolve_backend(
                ctx.deps.profile_cfg,
                self.experiment)
            if backend == "local":
                await self.experiment.workflow.execute(
                    run=self.mol_run,
                    profile_config=ctx.deps.profile_cfg)
            else:
                await _submit_and_maybe_wait(
                    backend=backend,
                    mol_run=self.mol_run,
                    profile_cfg=ctx.deps.profile_cfg)
        # ReplicaNode 是叶子,直接 End
        return End(...)

def build_sweep_graph(workspaces, profile_cfg, jobs: int) -> Graph:
    # 汇总所有 (project, experiment, replica),构建单级并行图
    ...
```

### 5.2 `_execute_sweep` 重构

`cli/run_cmd.py` 现有的 94 行 `_execute_sweep` 大幅简化:

```python
async def _execute_sweep_async(script, profile_cfg, resume, workspace, jobs):
    workspaces = load_workspaces(script)
    runs = _discover_runs(workspaces, profile_cfg, resume)   # 保留当前 resume 逻辑
    graph = build_sweep_graph(runs, profile_cfg, jobs=jobs)
    await graph.run(...)

def _execute_sweep(...):
    asyncio.run(_execute_sweep_async(...))
```

`_local_handler` 和 `SubmitHandler` 被删除(其逻辑迁移进 `ReplicaNode.run`)。

### 5.3 `worker.py` 扩展

`molexp/plugins/submit_molq/worker.py` 增加 `--task <name>` 选项:

```python
# 旧: python -m worker <script> <run_dir>
#     (跑整个 workflow)
# 新: python -m worker <script> <run_dir> [--task <node_name>]
#     不带 --task 保持旧行为(整个 workflow)
#     带 --task 只执行指定 node(per-node backend 用)
```

### 5.4 molq async 接口

molq 的 `JobHandle.wait()` 是 blocking 轮询(`monitor.py:40-96`)。过渡期:

```python
# molexp 侧临时方案
async def wait_async(handle, *, timeout=None):
    return await asyncio.to_thread(handle.wait, timeout=timeout)
```

长期方案:molq 本身加 `JobHandle.wait_async()`,把 `monitor.py` 的
`threading.Event.wait(interval)` 轮询循环改写成 `asyncio.sleep(interval)`。
独立 PR,不阻塞本 spec。

### 5.5 Snapshot / Resume

pydantic-graph 的 snapshot 机制天然支持 `SweepGraph`。新增能力:

- **父进程崩溃恢复**: 快照包含每个 replica 的 `job_id`(如已提交)。重启后
  对还在 running 的作业,节点从 `handle.wait_async()` 重新 await——molq store
  按 `job_id` 幂等查询状态。
- **`--resume` 语义**: 复用现有 `run_cmd.py:179-198` 的 run-id 匹配逻辑,只是
  进入点从 `for` 循环变成 `SweepGraph.resume(snapshot)`。

## 6. 迁移路径

### Phase 1: Sweep 层 pydantic-graph 化(仅 `--local`)

- 新增 `SweepGraph`,仅支持 `backend=local`
- `--local` 下的 `-j N` 生效
- 其他 backend 暂时继续走旧 `SubmitHandler`(保留两套代码)
- **用户可见变化**: `molexp run --local -j 4` 开始工作
- **风险**: 低。纯新增代码,老路径不动。

### Phase 2: Backend 统一入口

- 删除 `_local_handler` 和 `SubmitHandler`,所有 backend 走 `SweepGraph`
- `ReplicaNode.run` 根据 `backend` 分支(local 直跑 / molq 提交)
- 加入 `--block/--no-block` 支持
- `worker.py` 行为不变
- **用户可见变化**: `--backend slurm --block` 出现;`--slurm` 仍可用,语义等价
- **风险**: 中。slurm 路径重写,需要回归集群侧真机测试。

### Phase 3: Per-node backend

- `@wf.task(backend=..., resources=...)` 生效
- `worker.py` 加 `--task` 选项
- profile 加 `per_node_backend: true` 开关(默认 false,避免误启用)
- **用户可见变化**: 进阶用户可以混合执行
- **风险**: 中。需要 node 输入输出落盘约定的最佳实践文档。

### Phase 4: molq async API

- molq 原生 `wait_async()` 上线,molexp 去掉 `asyncio.to_thread` wrapper
- 独立推进,和 Phase 1/2/3 解耦

### 时间估算

| Phase | 预估工作量 | 依赖 |
|-------|-----------|------|
| 1 | 2-3 天 | 无 |
| 2 | 3-4 天 | Phase 1 |
| 3 | 3-5 天 | Phase 2 |
| 4 | 2-3 天(molq 侧) | 独立 |

## 7. 对现有脚本的影响

以 `molnex/examples/train_allegro_qm9.py` 为参照:

| 代码段 | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| `experiments = [...]` (73-85) | 不变 | 不变 | 不变 |
| `@wf.task prepare_data` (174-232) | 不变 | 不变 | 可选:加 `backend="local"` 显式标注(非必需) |
| `@wf.task train_model` (235-416) | 不变 | 不变 | 可选:`backend="slurm", resources=...` 启用 per-node 提交 |
| `me.entry(ws)` (428) | 不变 | 不变 | 不变 |
| `molcfg.yaml` | 可新增 `jobs:` | 可新增 `backend: / block:` | 可新增 `per_node_backend: true` |

**结论**: 用户脚本代码永远保持现状。所有新能力开关在 profile 或 decorator。

## 8. 兼容性

### 8.1 CLI 别名表

| 旧 flag | 新等价写法 |
|---------|-----------|
| `--local` | `--backend local -j 1` |
| `--local --bg` | `--backend local -j 1` + `nohup` 包装(`--bg` 功能保留) |
| `--slurm` | `--backend slurm --no-block` |
| `--slurm --block` | `--backend slurm --block`(语义同旧) |
| `--scheduler local` | `--backend local`(如果是 molq 的 local scheduler,仍然是 `--backend slurm` 但用 molq local cluster) |

### 8.2 破坏性变更

**无**。所有现有调用方式保持。Phase 2 后 `SubmitHandler` 的 class API 消失,但
这是内部实现,不在公共 API 契约内。

### 8.3 配置文件兼容

- 旧 profile 不含 `backend:` / `jobs:` 字段时,默认 `backend: local, jobs: 1`
  = 旧行为。
- 新字段位置参见 §4.2。

## 9. 测试策略

### 9.1 新增单元测试

- `tests/test_sweep/test_graph.py`: `SweepGraph` 基本执行、并发度、snapshot
- `tests/test_sweep/test_backend_resolution.py`: CLI > profile > decorator 优先级
- `tests/test_sweep/test_resume.py`: 父进程重启从快照恢复 job

### 9.2 集成测试

- `molexp run --backend local -j 4` 实际并发度(通过 task 内的 sleep 计时)
- `molexp run --backend slurm --block` 在 molq `testing` scheduler(内存模拟)上
  端到端跑通
- 混合 backend:一个 task `local` + 一个 task `slurm`,检查输入输出交接

### 9.3 回归测试

- 现有 `tests/test_cli/test_run_cmd.py` 全部保持 pass
- `examples/train_allegro_qm9.py` 在 Phase 2 后用 `--backend local --smoke` 跑完

## 10. 开放问题

1. **`--no-block` 下 `molexp run` 的 exit code**: 作业已提交但未完成,返回 0
   表示"提交成功"还是非 0 表示"未验证完成"?倾向返回 0 + stdout 明确告知需要
   `molexp watch`。
2. **`-j auto` 含义**: CPU 核数?GPU 数量?对 local 合理,对 slurm 作为"并发守候
   上限"可能过大。建议 `auto` 仅 local 生效,slurm 要求显式 `-j N`。
3. **Per-node snapshot 粒度**: 一个 replica 内多个 node 各自提交 job,某个
   node 失败后 resume 是重跑失败 node 还是整个 replica?pydantic-graph 的默认
   是 per-node,沿用即可,但需要文档强调。
4. **TaskResources 的 schema**: 直接复用 molq 的 `JobResources`(`gpu_count`
   / `memory` / `time_limit`),还是包一个 molexp 自己的类型?倾向复用,减少
   转换层。
5. **ctx.molq 在 local backend 下的语义**: 应该 fallback 到 molq 的 `local`
   scheduler,还是 raise?倾向前者——用户代码保持 backend-agnostic。

## 11. 参考

- `cli/run_cmd.py:206-209, 452-543` 当前 sweep 双路径实现
- `workflow/_pydantic_graph/node.py:88-124` 内层 level 并发模板
- `plugins/submit_molq/submit.py:52-115` 当前 SubmitHandler
- `plugins/submit_molq/worker.py:18-56` 现有 worker 入口
- `molq/submitor.py:197-251, 1120-1154` JobHandle 现状
- `molq/monitor.py:40-96` 阻塞轮询,待 async 化
- 关联 spec: `docs/spec/molcfg-profiles.md`(profile 机制)
