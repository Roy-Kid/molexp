# molcfg + Profiles: 配置文件与 Profile 机制

**状态**: Draft · **作者**: @RoyKid · **日期**: 2026-04-13

## 1. 动机

当前 `--dry-run` 是硬编码的一等执行模式，扎根于 CLI、RunStatus、RunMetadata、
ExecutionConfig、ctx、UI badge 等多层。进一步的需求（`--profile smoke`、
`--dataset-md17`、速度测试、debug 切片）会让 CLI 快速膨胀为 CMake 式的
flag 森林。

**目标**: 用单一机制——带命名 profile 的配置文件——覆盖所有参数变体
（dataset、epochs、batch_size、dry-run 是否跳过重计算、...）。

**Profile 语义对框架透明**: 框架只负责"加载配置 / 选中切片 / 注入 ctx /
记录 profile 名到 run metadata"。切片里写什么字段、task 如何解读它们
（包括"是否产生持久化副作用"），完全由用户自己定义。框架不内置任何
语义字段（没有 `side_effects`、没有 `real`、没有 `persist`）。

**非目标**: 向后兼容 `ctx.dry_run` / `RunStatus.DRY_RUN` / `--dry-run` 作为
独立字段。全部替换，无 deprecation 期。

## 2. 设计原则

1. **CLI 稳定**: `molexp run script.py [--config X.yaml] [--profile NAME]` 是
   唯一入口；新增变体只改 config，不改 CLI。
2. **Profile 是命名的配置切片**: profile 名本身就是 UI 展示标签、run
   metadata 字段、resume 过滤条件。
3. **框架对 profile 内容不可知**: 任何字段都是用户数据。Task 代码通过
   `ctx.config["epochs"]` 等自由读取。框架不解释。
4. **Profile 名归一化**: `-` 自动替换为 `_`（YAML 里 `dry-run` 等价于
   `dry_run`），避免 Python 标识符场景和 CLI 下划线/连字符混淆。
5. **格式**: YAML（默认）或 JSON。不支持 TOML（stdlib 只读）。
6. **Profile 可继承**: 显式 `extends: NAME`，不用 YAML anchor（JSON 写不出、
   魔法多）。

## 3. Schema 草案

### 3.1 配置文件

```yaml
# molcfg.yaml
version: 1

# 默认值（所有 profile 继承）
defaults:
  dataset: md17
  epochs: 100
  batch_size: 32
  seed_base: 42

profiles:
  dry-run:                # 加载时归一化为 "dry_run"
    extends: defaults
    epochs: 1

  smoke:
    extends: defaults
    epochs: 5
    batch_size: 8

  md22:
    extends: defaults
    dataset: md22

  prod:
    extends: defaults
    # 显式空，等同 defaults
```

### 3.2 Pydantic 模型

```python
# src/molexp/config/models.py
class ProfileConfig(Mapping[str, Any]):
    """Immutable merged config for one profile.

    Behaves like a read-only dict of user data; carries a ``name`` attr
    (normalized, "-" → "_"; ``None`` means "no profile, defaults only").
    Framework adds no semantic fields — every key comes from the YAML.
    """
    name: str | None
    # internal: frozen dict

    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default=None) -> Any: ...
    def hash(self) -> str: ...  # content-hash for RunMetadata.config_hash

class MolCfg(BaseModel, frozen=True):
    version: int = 1
    defaults: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def resolve(self, name: str | None) -> ProfileConfig:
        """Merge defaults + (optional) profile; normalize name; freeze."""
```

### 3.3 Context 注入

```python
class StepContext(Generic[StateT, DepsT, InputT]):
    config: ProfileConfig   # 新增
    # 移除 dry_run 属性
```

Task 作者的 migration（任何"跳过副作用"的判断都变成用户自己的 config 字段）：
```python
# 旧
if ctx.dry_run:
    return mock_result()

# 新 —— 用户自己在 YAML 里加字段，task 里自己读
# molcfg.yaml:
#   profiles:
#     dry-run:
#       skip_heavy_compute: true
if ctx.config.get("skip_heavy_compute"):
    return mock_result()

# 或纯参数变体
if ctx.config["epochs"] < 5:
    ...

# 也可以按 profile 名分支（不推荐，耦合命名）
if ctx.config.name == "dry_run":
    ...
```

### 3.4 RunMetadata

```python
class RunMetadata(BaseModel):
    id: str
    status: str = "pending"        # pending/running/succeeded/failed/cancelled
    profile: str | None = None     # 新增：activated profile name, e.g. "dry-run"
    config_hash: str | None = None # 新增：content-hash of resolved ProfileConfig
    # 删除: dry_run: bool
    ...
```

### 3.5 RunStatus

```python
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # 删除: DRY_RUN
```

Profile 与状态正交。Profile 为 `dry-run` 的 run 跑完了照样是 `SUCCEEDED`。

### 3.6 CLI

```
molexp run script.py [--config PATH] [--profile NAME] [--local|--slurm|...]
                     [--resume]

--config PATH    默认 ./molcfg.yaml，找不到则 profile 必须为空或报错
--profile NAME   激活的 profile 名，默认无（= 用 defaults）
--resume         重跑 profile 匹配、状态非 succeeded 的 run
```

**删除**: `--dry-run` flag。用 `--profile dry-run` 代替。

### 3.7 UI

- Run 列表：紧邻 status 的位置显示 `[{profile}]` 徽章（若 profile 非空）
- 颜色映射可配置，但默认：`dry-run`=yellow, `smoke`=cyan, 其它=dim
- 不再有 "dry_run" 状态徽章

### 3.8 Resume 语义

新规则：`--resume --profile X` 找所有 `profile == X and status != "succeeded"`
的 run，重新执行。旧的"只 resume dry-run 状态"逻辑泛化为"resume
任意 profile"。

## 4. Roadmap

**Phase 1** (本次): 核心替换
- 引入 `molexp.config` 模块（MolCfg/ProfileConfig/加载器）
- 替换 ExecutionConfig → ProfileConfig
- 删除 RunStatus.DRY_RUN、RunMetadata.dry_run
- CLI 改用 --config/--profile
- Server / schemas 同步
- 测试全绿

**Phase 2** (后续 PR): UI
- 新 profile 徽章组件
- 徽章颜色配置项

**Phase 3** (后续): 文档
- quick-start 改写
- README 更新

## 5. Todo List

见 TaskList；核心路径：

1. 建 `src/molexp/config/` 模块：models + loader + resolver
2. 改 `workspace/models.py`：ExecutionConfig → ProfileConfig，RunMetadata 字段
3. 改 `workspace/run.py`：删 DRY_RUN 枚举，ctx 暴露 config 而非 dry_run
4. 改 `workflow/context.py`：替换 dry_run property → config property
5. 改 `workflow/_pydantic_graph/state.py + node.py + runtime.py + compiler.py`
6. 改 `workflow/spec.py + runtime.py`：execute() 接受 profile
7. 改 `cli/__init__.py`：--config/--profile，删 --dry-run 及其分支
8. 改 `server/routes/run.py`：状态列表更新
9. 改 `monitor.py`：徽章逻辑
10. 迁移现有测试（tests/）
11. 添加新测试：config 加载、profile 解析、继承
12. 运行完整 pytest 确认绿
