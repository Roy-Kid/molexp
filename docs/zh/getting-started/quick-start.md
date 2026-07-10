# 快速开始

本页让你在一分钟内从零到一次追踪运行。你将定义一个两任务的工作流，在磁盘上创建工作区，执行工作流，并读回结果。

## 脚本

将以下内容复制到 `demo.py`：

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

# 1. 定义工作流
wf = WorkflowCompiler(name="sum")

@wf.task
def fetch(scale: float = 1.0) -> dict:
    return {"values": [1.0, 4.0, 9.0], "scale": scale}

@wf.task(depends_on=["fetch"])
def summarize(values: list[float], scale: float = 1.0) -> float:
    return sum(values) * scale

# 2. 创建工作区层级
ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("sum").add_run(params={"scale": 2.0})

# 3. 执行并读取结果
result = run.execute(wf)
print(run.status, result.outputs["summarize"])
```

运行它：

```bash
python demo.py
```

输出为 `succeeded 28.0`。

## 发生了什么

**步骤 1 — 定义。** `WorkflowCompiler` 持有任务定义。`@wf.task` 将普通函数变为工作流节点。`depends_on=["fetch"]` 告诉引擎 `summarize` 在 `fetch` 之后运行并接收其输出。

**步骤 2 — 创建。** `Workspace("./lab")` 在磁盘上创建目录。流畅链式调用 `.project("demo").experiment("sum").add_run(params={"scale": 2.0})` 构建持久化层级：项目分组相关工作，实验命名一个可重复定义，运行记录一次具体执行及其参数。

**步骤 3 — 执行。** `run.execute(wf)` 包办一切：编译工作流，打开运行的追踪生命周期，将运行参数绑定到根任务后执行图，将每个任务的输出持久化到运行目录，并返回结果。

## 数据如何流动

秘密在于**名字绑定**。运行的 `params={"scale": 2.0}` 绑定到 `fetch` 的 `scale` 参数，因为它们同名。`fetch` 返回带有 `values` 和 `scale` 键的字典——这些键绑定到 `summarize` 的 `values` 和 `scale` 参数。没有上游匹配的参数回退到声明的默认值。

```
params={"scale": 2.0}
        │
        ▼
     fetch(scale=2.0)  →  {"values": [1,4,9], "scale": 2.0}
        │                      │            │
        │   按名字绑定 ─────────┘            │
        ▼                                   │
  summarize(values=[1,4,9], scale=2.0)  ←───┘
        │
        ▼
      28.0
```

## 结果留在磁盘上

脚本退出后，运行仍在。打开新的 Python 会话读回：

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
same_run = ws.project("demo").experiment("sum").get_run(run.id)
print(same_run.status)                     # succeeded
print(same_run.get_result("summarize"))    # 28.0
```

## 下一步

- 如果工作流定义是陌生部分，继续读 [第一个工作流](first-workflow.md)。
- 如果工作区层级是新的，读 [追踪运行](tracked-runs.md)。
- 如果想用 `molexp run` 来驱动而不是手动调用 `run.execute`，去 [CLI 与配置文件](cli-and-profiles.md)。
- 如果更喜欢点击而非脚本，试试 [从浏览器开始](start-from-ui.md)。

## 可运行示例

`examples/getting_started/01_quick_start.py` 是此脚本的独立版本。
