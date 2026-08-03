---
title: MolExp
description: 面向 FAIR 研究的智能体辅助科学工作流平台
hide:
  - navigation
  - toc
hero:
  kicker: 手册
  title: MolExp
  description: 用 Python 构建可复现的科学工作流。将任务定义为普通函数，让引擎处理依赖图，并将每次运行持久化到磁盘——可选配 LLM 智能体来规划、生成和驱动实验。
  actions:
    - label: 快速开始
      href: "#start-here"
      style: primary
    - label: 核心概念
      href: "#core-concepts"
    - label: 完整指南
      href: "guide/"
  install:
    label: 安装
    methods:
      - { label: pip, command: pip install molexp }
      - { label: uv, command: uv add molexp }
  badges:
    - img: https://img.shields.io/pypi/v/molexp
      href: https://pypi.org/project/molexp/
      alt: PyPI version
    - img: https://img.shields.io/badge/python-3.12%2B-blue
      href: https://pypi.org/project/molexp/
      alt: Python 3.12+
---

<h1 class="molcrafts-sr-only">MolExp</h1>

<div class="molcrafts-manual-home" markdown>

<section id="start-here" class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">从这里开始</span>

## 找到合适的页面

把这一页当作手册的索引，而非营销概览。

</div>

<nav class="molcrafts-manual-index" aria-label="手册入口">
  <a href="getting-started/quick-start/">
    <span>01</span>
    <strong>运行快速开始</strong>
    <em>一个脚本，两个任务，一次追踪运行。30 秒看完整个生命周期。</em>
  </a>
  <a href="getting-started/first-workflow/">
    <span>02</span>
    <strong>编写第一个工作流</strong>
    <em>理解任务、依赖关系、编译，以及同步/异步混用。</em>
  </a>
  <a href="getting-started/tracked-runs/">
    <span>03</span>
    <strong>追踪运行到磁盘</strong>
    <em>工作区 → 项目 → 实验 → 运行。持久化、参数扫描、恢复、重跑。</em>
  </a>
  <a href="getting-started/cli-and-profiles/">
    <span>04</span>
    <strong>使用 CLI 与配置文件</strong>
    <em>用 molexp run 替代 asyncio.run()。用 molcfg.yaml 管理运行变体。</em>
  </a>
  <a href="getting-started/start-from-ui/">
    <span>05</span>
    <strong>从浏览器开始</strong>
    <em>通过 UI 创建项目、实验和运行。无需写 Python。</em>
  </a>
</nav>

</section>

<section id="representative-workflows" class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">一瞥</span>

## 长什么样

三种模式覆盖大多数用例。每个都是完整、可运行的脚本。

</div>

<div class="molcrafts-workflow-list" markdown>

<article markdown>

<div class="molcrafts-workflow-list__meta">模式 01 · 定义+运行</div>

### 两个任务，一次追踪运行

用装饰器定义工作流，创建工作区，执行运行。引擎处理依赖顺序、线程隔离和持久化。

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="sum")

@wf.task
def fetch(scale: float = 1.0) -> dict:
    return {"values": [1.0, 4.0, 9.0], "scale": scale}

@wf.task(depends_on=["fetch"])
def summarize(values: list[float], scale: float = 1.0) -> float:
    return sum(values) * scale

ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("sum").add_run(params={"scale": 2.0})
result = run.execute(wf)
print(run.status, result.outputs["summarize"])  # succeeded 28.0
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">模式 02 · 参数扫描</div>

### 一个工作流，多组参数

在实验上声明参数网格。扫描为每个参数组合物化一个内容寻址的运行，并全部执行。

```python
scan = (
    ws.project("demo")
    .experiment("lr-scan")
    .sweep(wf, {"scale": [1.0, 2.0, 4.0]})
)
summary = scan.execute()
for row in summary.to_records():
    print(row["scale"], row["status"], row["summarize"])
best = summary.min_by("summarize")
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">模式 03 · CLI 驱动</div>

### 注册实验，从终端运行

将编译好的工作流绑定到实验，让 `molexp run` 负责发现、配置文件、恢复标志和调度器执行。

```python
(
    me.Workspace("./lab", name="lab")
    .project("demo")
    .experiment("sum")
    .run(wf.compile(), params={"scale": [1.0, 2.0]})
)
```

```bash
molexp run train.py --profile smoke
molexp run train.py --profile smoke --override scale=4.0
molexp run train.py --resume            # 继续失败的运行
molexp run train.py --rerun --fresh     # 从头重新执行
```

</article>

</div>

</section>

<section id="core-concepts" class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">概念</span>

## 工作原理

四个层次，各司其职。松耦合组合。

</div>

<dl class="molcrafts-feature-matrix">
  <div>
    <dt>工作流 Workflow</dt>
    <dd>将计算定义为任务图。装饰器或类式编写。从依赖关系中自动推导并行。内容寻址缓存。</dd>
  </div>
  <div>
    <dt>工作区 Workspace</dt>
    <dd>持久化层级：工作区 → 项目 → 实验 → 运行。每次运行都是一个包含参数、状态和输出的持久记录。</dd>
  </div>
  <div>
    <dt>智能体 Agent</dt>
    <dd>LLM 对话层。规划实验、生成工作流代码、通过工具调用驱动运行——全部记录在磁盘会话中。</dd>
  </div>
  <div>
    <dt>流水线 Harness</dt>
    <dd>实验编排器。起草规格 → 解析能力 → 生成工作流代码 → 编译 → 测试 → 审查。九个可审计步骤。</dd>
  </div>
  <div>
    <dt>资产 Assets</dt>
    <dd>每个产物、日志和检查点都以内容寻址标识和溯源信息追踪到生成它的运行和任务。</dd>
  </div>
  <div>
    <dt>配置文件 Profiles</dt>
    <dd>YAML 中的运行变体。一个脚本，多种形态——smoke、production、dry-run——通过 CLI 标志切换。</dd>
  </div>
</dl>

</section>

<section id="ecosystem" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">生态</span>

## 向外扩展，保持轻量

`import molexp` 保持轻量——重依赖只在真正用到时加载。核心通过可选 extras
（`molexp[agent]`、`molexp[tensorboard]`）连接 molcrafts 栈，并通过两条独立
插件通道接入你自己的代码。

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <a href="guide/molq/">
    <strong>molq · 调度桥</strong>
    <em>让 <code>molexp run</code> 上集群：同一运行可提交到 Slurm、PBS 或 LSF。换的是传输，工作流与记录不变。</em>
  </a>
  <a href="getting-started/cli-and-profiles/">
    <strong>molcfg · 运行配置</strong>
    <em>支撑 profile 系统。<code>molcfg.yaml</code> 保存执行变体，一个 <code>--profile</code> 切换。</em>
  </a>
  <a href="architecture/plan-mode/">
    <strong>molmcp · 能力发现</strong>
    <em>为智能体 Harness 落地能力。PlanOrchestrator 经 molmcp 发现完整工具链，再绑定实验所需的最小子集。</em>
  </a>
  <a href="concept/plugins/">
    <strong>molvis · 可视化</strong>
    <em>内置 UI 插件在浏览器中渲染分子结构——与 <code>core</code>、<code>metrics</code> 同属树内包。</em>
  </a>
  <a href="plugins/">
    <strong>CLI 插件</strong>
    <em>任意 pip 包通过 <code>molexp.cli_plugins</code> 入口点注册 <code>molexp &lt;yourcmd&gt;</code> 子命令。</em>
  </a>
  <a href="plugins/">
    <strong>UI 插件</strong>
    <em>通过独立的 <code>molexp.ui_plugins</code> 通道向 SPA 注入动态加载的 React 包。</em>
  </a>
</div>

</section>

<section id="doc-map" class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">地图</span>

## 文档地图

下列分区与导航树对应，方便回头查阅。

</div>

<div class="molcrafts-doc-map">
  <section>
    <h3><a href="getting-started/">快速入门</a></h3>
    <p>快速开始、第一个工作流、追踪运行、CLI 与配置文件，以及基于浏览器的 UI 导览。</p>
  </section>
  <section>
    <h3><a href="concept/">核心概念</a></h3>
    <p>工作流与工作区模型、智能体层、资产可复现性，以及插件架构。</p>
  </section>
  <section>
    <h3><a href="guide/">指南</a></h3>
    <p>任务编写、控制流、参数扫描、工作区 API、持久化、配置文件、Plan Mode、服务生命周期与调度桥。</p>
  </section>
  <section>
    <h3><a href="architecture/">架构</a></h3>
    <p>层级边界、导入规则、智能体防火墙、Plan Mode 流水线，以及工作流引擎设计。</p>
  </section>
</div>

</section>

<section id="three-pillars" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">设计</span>

## 三大支柱

MolExp 不是功能的大杂烩。每个子系统服务于以下目标之一。

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <a href="concept/workflow/">
    <strong>可复现计算</strong>
    <em>内容寻址缓存、确定性运行 ID、逐任务快照——相同输入始终产生相同标识。</em>
  </a>
  <a href="concept/workspace/">
    <strong>持久记录</strong>
    <em>每次运行都是一个包含参数、溯源、输出和执行历史的目录。数据不只在内存中。</em>
  </a>
  <a href="concept/agent/">
    <strong>智能体辅助科学</strong>
    <em>LLM 智能体规划实验、生成代码、驱动运行。每个决策都可审计——智能体会话是一等工作区对象。</em>
  </a>
</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">速查</span>

## 关键 API

你最常接触的入口点。

</div>

<div class="molcrafts-manual-list">
  <a href="concept/workflow/">
    <strong>WorkflowCompiler</strong>
    <em>用 @wf.task 定义任务。编译为冻结图。</em>
  </a>
  <a href="getting-started/tracked-runs/">
    <strong>Workspace → Project → Experiment → Run</strong>
    <em>创建持久化层级。执行、扫描、恢复、重跑。</em>
  </a>
  <a href="guide/task-and-actor/">
    <strong>Task · Actor</strong>
    <em>两种任务形态：同步/异步函数（装饰器）或可复用类。</em>
  </a>
  <a href="guide/control-flow/">
    <strong>Branch · Loop · Parallel</strong>
    <em>控制流原语，组合成任意 DAG 形态。</em>
  </a>
  <a href="guide/sweeps/">
    <strong>Sweep · RunSet</strong>
    <em>网格搜索参数，并行执行，用 to_records() 汇总。</em>
  </a>
  <a href="guide/plan-mode/">
    <strong>PlanOrchestrator</strong>
    <em>两阶段规划流水线：任务板 → 审查冻结 → 实现编译。</em>
  </a>
</div>

</section>

</div>
