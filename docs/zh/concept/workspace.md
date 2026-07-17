# 工作区模型

工作区层是磁盘上的持久记录。它回答*执行后留下什么*。

## 四层层级

```
工作区 Workspace     ← 根目录（如 ./lab）
└── 项目 Project      ← 分组相关工作（如 qm9）
    └── 实验 Experiment ← 可重复定义（工作流 + 参数）
        └── 运行 Run       ← 一次具体执行尝试
```

| 层级 | 是什么 | 在磁盘上 |
|---|---|---|
| **工作区** | 一组工作的根 | `workspace.json` |
| **项目** | 分组相关实验 | `projects/<slug>/project.json` |
| **实验** | 一个工作流 + 参数空间 | `projects/<slug>/experiments/<slug>/experiment.json` |
| **运行** | 一次带状态和输出的执行 | `projects/<slug>/experiments/<slug>/runs/run-<id>/run.json` |

## 定义 vs. 结果

关键区别在于**实验**（你打算重复什么）和**运行**（实际发生了什么）。实验携带工作流引用、参数空间和溯源信息。运行携带每次尝试可变的东西：状态、时间戳、配置文件、结果、错误和执行历史。

没有这个分离，重试和比较很快变得模糊不清。

## 配置文件与元数据

配置文件（`molcfg.yaml`）位于工作流执行层和工作区持久层之间的边界。任务将配置文件字段作为命名参数读取。解析后的配置文件——名称、合并后的配置、`config_hash`——存储在运行记录上。你可以事后查看 `run.json`，恢复运行所使用的确切配置。

## 磁盘布局

```
workspace_root/
├── workspace.json          ← 实体元数据
├── project.json            ← 子级索引（项目列表）
├── meta.yaml               ← OKF 概念标记
├── index.md                ← 知识图谱叙述
└── projects/<project_id>/
    ├── project.json        ← 实体元数据
    ├── experiment.json     ← 子级索引（实验列表）
    └── experiments/<exp_id>/
        ├── experiment.json ← 实体元数据
        ├── run.json        ← 子级索引（运行列表）
        └── runs/run-<id>/
            ├── run.json    ← 身份和溯源
            ├── _ops/run.json ← 热状态（状态、所有权）
            ├── assets.json ← 运行作用域资产清单
            └── executions/<exec_id>/
                ├── execution.json
                ├── workflow.json  ← 逐任务输出
                ├── stdout.log
                └── stderr.log
```

## 下一步

- 具体的 Python API，见 [工作区 API](../guide/workspace-api.md)。
- 可复用数据和溯源，见 [资产与可复现性](assets-and-reproducibility.md)。
- 发现此层级的 CLI，见 [CLI 与配置文件](../getting-started/cli-and-profiles.md)。
