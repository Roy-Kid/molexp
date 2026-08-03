# 用 Agent Harness 做规划

`molexp plan` 把自然语言实验草案变成：

1. **任务板**（有序步骤 + 验收标准），
2. **人工审查**（审批收件箱 / TTY / `--yes`），
3. **冻结计划** + 可读 **计划报告**，
4. 默认再 **实现** 工作流（codegen + 仅编译 dry-run）。

生产入口是 `molexp.harness.PlanOrchestrator`。内部细节见
[Plan Mode 架构](../architecture/plan-mode.md)。

## 前置条件

```bash
pip install "molexp[agent]"
molexp config set agent.model anthropic:claude-sonnet-4-5
```

模型来自 `~/.molexp/config.json` 的 `agent.model`（CLI 与服务端同一加载器）。
可用 `--model` 覆盖。

## 规划（并实现）

```bash
molexp plan "筛选三种溶剂比例并报告电导率"
molexp plan --file draft.md
```

| 阶段 | 你会看到 |
|------|----------|
| 规划 | Agent 用工具往任务板上放任务；表单不完整则不能结束 |
| 审查 | 硬门禁 — 批准 / 拒绝 / 修订（keep_tasks、备注、优先级） |
| 实现 | 按任务生成代码与单测，再 compile-only |

无 TTY 授权且未 `--yes` 时，审查门禁会 **挂起** 到审批收件箱；授权后同一 run
store-first 恢复。默认项目/实验为 `plans` / `plan`。

```bash
molexp plan --file draft.md --yes   # 自动过审查门禁，仍会跑实现阶段
```

## UI

Agent 作曲器切到 **Plan**（模式胶囊或 `Shift+Tab`）。同一 `POST /plan-tasks`
驱动 `PlanOrchestrator`。

- **左轨** — 新阶段列表（任务板 → 审查 → 冻结 → 报告 → 绑定 → 源码 → 编译…）
- **右栏** — 当前阶段交付物
- **审批** — 结构化表单；**批准与修订都会提交 fieldValues**

## 产物

```text
runs/run-<id>/
├── plan/task_board.json
├── artifacts/
└── harness.sqlite
```

实现全绿时，工作流 IR 会投影到 experiment，供图查看器打开。

## 审批动作

| 动作 | 效果 |
|------|------|
| **批准** | 写入授权 + 可选 field_values；恢复后冻结并实现 |
| **修订** | 把 field_values 应用到任务板，重新进门禁 |
| **拒绝** | 计划任务标记失败 |

## 相关

- 架构：[Plan Mode 架构](../architecture/plan-mode.md)
- 跟踪运行：[Tracked runs](../getting-started/tracked-runs.md)
