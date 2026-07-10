# 工作流模型

工作流层是计算图。它回答*应该发生什么*——任务边界、依赖顺序和数据流。它**不**知道项目分组、运行目录或调度器传输。

## 编写、编译、执行

MolExp 将工作流的生命周期分为三个阶段：

| 阶段 | 工具 | 做什么 |
|---|---|---|
| **编写** | `WorkflowCompiler` | 声明任务和依赖 |
| **编译** | `.compile()` | 冻结为已验证的 `CompiledWorkflow` |
| **执行** | `WorkflowRuntime` 或 `Run.execute()` | 驱动图 |

可以用装饰器编写：

```python
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="demo")

@wf.task
def fetch() -> list[float]:
    return [1.0, 4.0, 9.0]

@wf.task(depends_on=["fetch"])
def summarize(data: list[float]) -> float:
    return sum(data)

compiled = wf.compile()
```

也可以用可复用的任务类——两者产生同一种 `CompiledWorkflow`。

## 执行独立于持久化

编译后的工作流可以纯粹在内存中运行：

```python
import asyncio
from molexp.workflow import WorkflowRuntime

result = asyncio.run(WorkflowRuntime().execute(compiled))
```

或者在追踪运行下以完整持久化运行：

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("baseline").add_run(params={})
result = run.execute(wf)
```

图是相同的。只有围绕它的生命周期变化了。

## 哪些不在此层

工作流层刻意**不**知道：

- 项目/实验分组（那是工作区的）
- 运行目录和执行历史（工作区）
- 调度器传输（插件）
- 共享数据集和派生资源（资产）

这个窄边界保持工作流可复用——同一个编译图可以在本地、在 `molexp run` 下、或从远程工作节点运行。

## 数据流：名字绑定

值通过**名字**在任务间传递。上游任务的返回值键绑定到下游任务同名的参数。没有上游匹配的参数回退到默认值。构建时配置（`params`）可以覆盖默认值。

```python
@wf.task
def source() -> dict:
    return {"x": 10, "y": 20}

@wf.task(depends_on=["source"])
def consumer(x: int, y: int, z: int = 0) -> int:
    return x + y + z  # x=10（来自 source），y=20（来自 source），z=0（默认值）
```

## 下一步

- 关于持久化方面，阅读 [工作区模型](workspace.md)。
- 关于可复用数据和溯源，阅读 [资产与可复现性](assets-and-reproducibility.md)。
