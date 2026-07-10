# 快速入门

本节是 MolExp 的实践路径。按真实工作中遇到的顺序组织：先跑起来，再理解追踪机制，最后加上 CLI 和配置文件。

## 学习路径

1. **[快速开始](quick-start.md)** — 一个脚本，两个任务，一次追踪运行。最快速度看完整个系统。
2. **[第一个工作流](first-workflow.md)** — 任务、依赖、编译、同步/异步混用。在挂载工作区之前先理解工作流。
3. **[追踪运行](tracked-runs.md)** — 持久化层级（工作区 → 项目 → 实验 → 运行）。参数扫描、恢复、重跑、CLI 注册。
4. **[CLI 与配置文件](cli-and-profiles.md)** — 用 `molexp run` 替代 `asyncio.run()`。用 `molcfg.yaml` 管理运行变体。
5. **[从浏览器开始](start-from-ui.md)** — 通过 UI 创建项目、实验和运行。无需写 Python。

## 准备好深入时

当你能够读懂调用 `wf.compile()` 和 `run.execute(wf)` 的脚本后，[核心概念](../concept/index.md) 会帮你巩固思维模型，[指南](../guide/index.md) 则涵盖更详细的话题。
