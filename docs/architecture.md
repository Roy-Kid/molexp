# 分子建模平台运行与数据组织架构设计

## 设计哲学

本架构采用 **Hybrid Project-Experiment-Run + 全局 Asset Repository** 模式,核心理念：

- **可追踪性**：每次运行完全可重放，通过快照和引用记录完整上下文
- **资产复用**：Asset 全局去重和共享，避免数据冗余
- **层次清晰**：Project（课题）→ Experiment（实验设计）→ Run（单次执行）三层分离关注点
- **云端就绪**：Asset 仓库抽象化，未来可无缝迁移至对象存储/数据湖

---

## 1. 概念定位

### 1.1 Project（项目）

**定位**：长期研究/工程课题的顶层容器

- 代表一个持续的研究方向（如"PEO 聚电解质扫描"、"MolPack 核心开发"）
- 承载领域上下文、公共配置、团队信息
- 相对稳定，数量有限（通常几个到几十个）
- 包含多个 Experiment，提供组织边界

**职责**：
- 提供高层描述和元信息（owner、tags、README）
- 定义项目级别的默认配置（计算资源、环境变量）
- 作为权限和访问控制的边界

### 1.2 Experiment（实验）

**定位**：可重复的研究问题/参数空间定义

- 从属于某个 Project
- 定义一个 **workflow 模板**（TaskGraph）和 **参数空间**
- 可以是单一配置，也可以是参数扫描（parameter sweep）
- 一个 Experiment 可以产生多个 Run（不同参数/随机种子/输入）

**职责**：
- 存储 workflow 定义（模板级别，非快照）
- 定义参数空间和默认值
- 指定默认输入 Asset（通过 AssetRef）
- 记录实验目标和假设

### 1.3 Run（运行）

**定位**：workflow 的单次执行实例

- 在特定时间、特定 Experiment 下，使用具体参数执行 workflow
- **完全可重放单位**：保存 workflow 快照、参数快照、环境信息
- 通过 AssetRef 引用输入和输出，本身不存储大文件
- 记录执行状态（pending/running/succeeded/failed）和日志

**职责**：
- 保存当次执行的完整上下文（参数、环境、时间戳）
- 记录 workflow 图和代码的快照（git commit hash 或序列化定义）
- 引用输入 Assets（reads）和生成输出 Assets（writes）
- 存储小文件（logs、临时图表、中间状态）

### 1.4 Asset / AssetRepo / AssetRef

**Asset**：可复用的数字产物

- 文件或文件集合：结构、轨迹、拓扑、图像、表格、模型权重等
- 内容可寻址（content-addressable），支持基于 hash 去重
- 可被多个 Run/Experiment/Project 共享
- 不绑定到任何特定 Project，全局管理

**AssetRepo**：全局资产库

- 统一存储和管理所有 Assets
- 提供 hash-based 去重
- 支持后端切换（本地文件系统 → 对象存储 → 数据湖）

**AssetRef**：轻量级引用

- Run/Experiment/Project 中记录"使用了哪个 Asset"
- 包含 asset_id、角色（role）、用途说明
- 支持数据血缘追踪（producer_run_id）

---

## 2. 文件系统组织架构

### 2.1 目录结构

```
workspace_root/
├── projects/
│   ├── peo-electrolyte-scan/              # project_slug
│   │   ├── project.yaml                    # 项目元信息
│   │   ├── README.md                       # 项目文档
│   │   └── experiments/
│   │       ├── density-sweep/              # experiment_slug
│   │       │   ├── experiment.yaml         # 实验定义
│   │       │   └── runs/
│   │       │       ├── 20251129_173045_a3b2/  # run_id (timestamp_shortid)
│   │       │       │   ├── run.json        # Run 元信息
│   │       │       │   ├── context.json    # 执行上下文快照
│   │       │       │   ├── asset_refs.json # Asset 引用清单
│   │       │       │   ├── logs/
│   │       │       │   │   ├── stdout.log
│   │       │       │   │   └── stderr.log
│   │       │       │   └── artifacts/      # 小文件（图表、临时数据）
│   │       │       │       └── plot.png
│   │       │       └── 20251129_180122_k9f1/
│   │       │           └── ...
│   │       └── temperature-ramp/
│   │           └── ...
│   └── molpack-core/
│       └── ...
│
├── assets/
│   ├── a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b/  # asset_id (UUID or hash)
│   │   ├── meta.yaml                       # Asset 元信息
│   │   └── data/
│   │       └── structure.pdb               # 实际数据文件
│   ├── b4g3f9e0-5c2d-5f6g-0b3c-2d4e5f6g7h8i/
│   │   ├── meta.yaml
│   │   └── data/
│   │       ├── trajectory.xtc
│   │       └── topology.top
│   └── ...
│
└── .workspace.yaml                         # 全局配置
```

### 2.2 关键文件职责

#### `project.yaml`

```yaml
project_id: "peo-electrolyte-scan"
name: "PEO Electrolyte Parameter Scan"
description: "Systematic study of PEO polymer electrolyte behavior"
created_at: "2025-11-01T10:00:00Z"
owner: "research-team"
tags: ["polymer", "electrolyte", "MD"]
config:
  default_resources:
    cpu: 8
    memory: "16GB"
  environment:
    MD_ENGINE: "gromacs"
```

**包含**：project_id、name、description、owner、tags、created_at、default_config

#### `experiment.yaml`

```yaml
experiment_id: "density-sweep"
project_id: "peo-electrolyte-scan"
name: "Density Sweep Experiment"
description: "Scan system density from 0.8 to 1.2 g/cm³"
created_at: "2025-11-15T14:30:00Z"

workflow_template:
  type: "taskgraph_v1"
  source: "workflows/md_pipeline.py"
  git_commit: "a3b4c5d"

parameter_space:
  density: [0.8, 0.9, 1.0, 1.1, 1.2]
  temperature: 300
  pressure: 1.0

default_inputs:
  - asset_id: "a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b"
    role: "initial_structure"
  - asset_id: "c5h6j8k9-7d8e-9f0a-1b2c-3d4e5f6g7h8i"
    role: "force_field"
```

**包含**：experiment_id、project_id、workflow 定义、参数空间、默认输入 AssetRef

#### `run.json`

```json
{
  "run_id": "20251129_173045_a3b2",
  "project_id": "peo-electrolyte-scan",
  "experiment_id": "density-sweep",
  "created_at": "2025-11-29T17:30:45Z",
  "finished_at": "2025-11-29T18:45:12Z",
  "status": "succeeded",
  "parameters": {
    "density": 1.0,
    "temperature": 300,
    "pressure": 1.0,
    "random_seed": 42
  },
  "workflow_snapshot": {
    "git_commit": "a3b4c5d",
    "workflow_file": "md_pipeline.py",
    "serialized_graph": "..."
  },
  "executor_info": {
    "hostname": "compute-node-03",
    "platform": "linux-x86_64",
    "python_version": "3.11.5"
  },
  "working_dir": "projects/peo-electrolyte-scan/experiments/density-sweep/runs/20251129_173045_a3b2",
  "logs_dir": "logs/"
}
```

**包含**：run_id、关联 project/experiment、时间戳、状态、参数、workflow 快照、环境信息

#### `context.json`

```json
{
  "environment": {
    "PATH": "...",
    "MD_ENGINE": "gromacs",
    "CONDA_ENV": "molexp-v1.2"
  },
  "dependencies": {
    "gromacs": "2023.1",
    "numpy": "1.24.0",
    "molexp": "0.3.5"
  },
  "hardware": {
    "cpu_count": 8,
    "gpu_available": true,
    "total_memory_gb": 32
  }
}
```

**包含**：环境变量、依赖版本、硬件信息

#### `asset_refs.json`

```json
{
  "inputs": [
    {
      "asset_id": "a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b",
      "role": "initial_structure",
      "accessed_at": "2025-11-29T17:30:50Z"
    },
    {
      "asset_id": "c5h6j8k9-7d8e-9f0a-1b2c-3d4e5f6g7h8i",
      "role": "force_field",
      "accessed_at": "2025-11-29T17:30:51Z"
    }
  ],
  "outputs": [
    {
      "asset_id": "d7e8f9g0-1h2i-3j4k-5l6m-7n8o9p0q1r2s",
      "role": "output_trajectory",
      "produced_at": "2025-11-29T18:44:00Z",
      "size_bytes": 524288000
    },
    {
      "asset_id": "e9f0g1h2-3i4j-5k6l-7m8n-9o0p1q2r3s4t",
      "role": "final_structure",
      "produced_at": "2025-11-29T18:45:10Z",
      "size_bytes": 102400
    }
  ]
}
```

**包含**：inputs（读取的 Assets）、outputs（生成的 Assets），每项记录 asset_id、role、时间戳

#### `assets/<asset_id>/meta.yaml`

```yaml
asset_id: "d7e8f9g0-1h2i-3j4k-5l6m-7n8o9p0q1r2s"
type: "trajectory"
format: "xtc"
created_at: "2025-11-29T18:44:00Z"
producer_run_id: "20251129_173045_a3b2"
size_bytes: 524288000
content_hash: "sha256:a3b4c5d6e7f8..."
mime_type: "application/x-gromacs-xtc"
tags: ["MD", "production-run", "peo"]
metadata:
  n_frames: 10000
  timestep_ps: 2.0
  total_time_ns: 20.0
files:
  - path: "data/trajectory.xtc"
    size: 524288000
    hash: "sha256:a3b4c5d6e7f8..."
```

**包含**：asset_id、type、format、created_at、producer_run_id、content_hash、size、mime_type、tags、metadata、files

### 2.3 Asset 仓库分离的理由

#### 为何分离？

1. **跨项目共享**：同一个力场、初始结构可以被多个 Project/Experiment/Run 复用
2. **避免重复存储**：基于 content-hash 去重，相同内容只存一份
3. **简化迁移**：Asset 仓库可独立备份、迁移到云端对象存储
4. **权限解耦**：Asset 访问权限可独立于 Project 管理
5. **Blood Lineage**：通过 producer_run_id 追踪数据来源，但不依赖目录结构

#### 如何支持？

- **跨项目共享**：多个 Run 的 `asset_refs.json` 可引用同一 asset_id
- **去重**：上传 Asset 时先计算 hash，相同 hash 则复用已有 asset_id
- **云端迁移**：将 `assets/` 目录替换为 S3/GCS/MinIO 后端，AssetRef 引用逻辑不变

---

## 3. 元数据与对象模型

### 3.1 Python 对象模型（Pydantic 示例）

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ============ Project ============
class Project(BaseModel):
    project_id: str = Field(..., description="Unique project identifier (slug)")
    name: str
    description: str = ""
    owner: str
    tags: List[str] = []
    created_at: datetime
    config: Dict[str, Any] = {}
    
    # Computed
    @property
    def path(self) -> str:
        return f"projects/{self.project_id}"

# ============ Experiment ============
class WorkflowTemplate(BaseModel):
    type: str = "taskgraph_v1"
    source: str  # Path to workflow file
    git_commit: Optional[str] = None

class Experiment(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier (slug)")
    project_id: str
    name: str
    description: str = ""
    created_at: datetime
    workflow_template: WorkflowTemplate
    parameter_space: Dict[str, Any]
    default_inputs: List["AssetRef"] = []
    
    @property
    def path(self) -> str:
        return f"projects/{self.project_id}/experiments/{self.experiment_id}"

# ============ Run ============
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowSnapshot(BaseModel):
    git_commit: Optional[str] = None
    workflow_file: str
    serialized_graph: Optional[str] = None

class Run(BaseModel):
    run_id: str = Field(..., description="Unique run identifier (timestamp_shortid)")
    project_id: str
    experiment_id: str
    created_at: datetime
    finished_at: Optional[datetime] = None
    status: RunStatus = RunStatus.PENDING
    parameters: Dict[str, Any]
    workflow_snapshot: WorkflowSnapshot
    executor_info: Dict[str, Any] = {}
    working_dir: str
    logs_dir: str = "logs/"
    
    @property
    def path(self) -> str:
        return f"projects/{self.project_id}/experiments/{self.experiment_id}/runs/{self.run_id}"

# ============ Asset ============
class AssetType(str, Enum):
    STRUCTURE = "structure"
    TRAJECTORY = "trajectory"
    TOPOLOGY = "topology"
    FORCEFIELD = "forcefield"
    IMAGE = "image"
    TABLE = "table"
    MODEL = "model"
    OTHER = "other"

class AssetFile(BaseModel):
    path: str
    size: int
    hash: str

class Asset(BaseModel):
    asset_id: str = Field(..., description="UUID or content hash")
    type: AssetType
    format: str  # e.g., "pdb", "xtc", "png", "csv"
    created_at: datetime
    producer_run_id: Optional[str] = None
    size_bytes: int
    content_hash: str = Field(..., description="SHA256 hash of content")
    mime_type: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    files: List[AssetFile]
    
    @property
    def path(self) -> str:
        return f"assets/{self.asset_id}"

# ============ AssetRef ============
class AssetRef(BaseModel):
    asset_id: str
    role: str = Field(..., description="e.g., 'input_structure', 'output_trajectory'")
    producer_run_id: Optional[str] = None
    accessed_at: Optional[datetime] = None
    produced_at: Optional[datetime] = None
    extra: Dict[str, Any] = {}

# ============ AssetRepo ============
class AssetRepo:
    """Abstract interface for Asset repository"""
    
    def store(self, asset: Asset, local_path: str) -> str:
        """Store asset and return asset_id"""
        raise NotImplementedError
    
    def retrieve(self, asset_id: str, dest_path: str) -> None:
        """Retrieve asset to destination path"""
        raise NotImplementedError
    
    def get_meta(self, asset_id: str) -> Asset:
        """Get asset metadata"""
        raise NotImplementedError
    
    def exists(self, content_hash: str) -> Optional[str]:
        """Check if asset with given hash exists, return asset_id if found"""
        raise NotImplementedError
    
    def delete(self, asset_id: str) -> None:
        """Delete asset (with safety checks)"""
        raise NotImplementedError
```

### 3.2 对象与文件对应关系

| 对象 | 文件路径 |
|------|---------|
| `Project` | `projects/<project_id>/project.yaml` |
| `Experiment` | `projects/<project_id>/experiments/<experiment_id>/experiment.yaml` |
| `Run` | `projects/<project_id>/experiments/<experiment_id>/runs/<run_id>/run.json` |
| `Asset` | `assets/<asset_id>/meta.yaml` + `assets/<asset_id>/data/*` |
| `AssetRef` | 嵌入在 `asset_refs.json` 中（inputs/outputs 列表） |

### 3.3 引用关系

**关键点**：
- Run → Asset：单向引用（通过 AssetRef）
- Asset → Run：可选的反向追踪（producer_run_id），不强依赖
- 避免双向强耦合，Asset 可以独立于 Project 存在

---

## 4. ID 与命名约定

### 4.1 Project

- **project_id**：人类可读的 slug，如 `peo-electrolyte-scan`，`molpack-core`
  - 规则：小写字母、数字、连字符，长度 3-50
  - 全局唯一 within workspace
- **project_slug**：同 project_id（简化设计）

### 4.2 Experiment

- **experiment_id**：人类可读的 slug，如 `density-sweep`，`temperature-ramp`
  - 规则：小写字母、数字、连字符
  - 在 Project 内唯一

### 4.3 Run

- **run_id**：`YYYYMMDD_HHMMSS_<short_uuid>`
  - 时间戳：便于排序和浏览
  - short_uuid：4-6位随机字符（如 `a3b2`），避免同一秒内冲突
  - 示例：`20251129_173045_a3b2`
  - 在 Experiment 内唯一

### 4.4 Asset

- **asset_id**：UUID v4 或 content hash (SHA256 前缀)
  - **推荐**：UUID v4（如 `a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b`）
    - 优点：生成简单，与内容无关，支持先分配 ID 后上传
  - **备选**：Hash-based（如 `sha256-a3b4c5d6e7f8...`）
    - 优点：内容寻址，天然去重
    - 缺点：需要先计算完整文件 hash 才能确定 ID
  - 全局唯一

### 4.5 Session（可选）

如需支持"交互式会话"概念（如"2025-11-29 这一天的所有探索性运行"）：

- **session_id**：`YYYYMMDD` 或 `YYYYMMDD_HH`
- Session 可以是一个轻量级标签或分组概念，不一定需要实体对象
- Session 与 Run 的关系：
  - 在 Run 的 metadata 中添加 `session_id` 字段
  - 或在 Project/Experiment 级别维护 session → run_ids 映射

**建议**：初期可不实现 Session，用 tags 或 created_at 过滤即可

---

## 5. 生命周期与操作场景

### 5.1 创建 Project 和 Experiment

```python
# Step 1: 创建 Project
project = Project(
    project_id="peo-electrolyte-scan",
    name="PEO Electrolyte Parameter Scan",
    description="Systematic study...",
    owner="research-team",
    created_at=datetime.now(),
    tags=["polymer", "electrolyte"]
)

# 写入文件系统
project_dir = Path("projects") / project.project_id
project_dir.mkdir(parents=True, exist_ok=True)
(project_dir / "project.yaml").write_text(project.model_dump_yaml())
(project_dir / "experiments").mkdir(exist_ok=True)

# Step 2: 定义 Experiment
experiment = Experiment(
    experiment_id="density-sweep",
    project_id=project.project_id,
    name="Density Sweep Experiment",
    workflow_template=WorkflowTemplate(
        source="workflows/md_pipeline.py",
        git_commit="a3b4c5d"
    ),
    parameter_space={"density": [0.8, 0.9, 1.0, 1.1, 1.2]},
    default_inputs=[
        AssetRef(asset_id="a3f2e8d9-...", role="initial_structure")
    ],
    created_at=datetime.now()
)

# 写入文件系统
exp_dir = project_dir / "experiments" / experiment.experiment_id
exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / "experiment.yaml").write_text(experiment.model_dump_yaml())
(exp_dir / "runs").mkdir(exist_ok=True)
```

### 5.2 启动新 Run

```python
# Step 1: 生成 run_id
from uuid import uuid4
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
short_id = str(uuid4())[:4]
run_id = f"{timestamp}_{short_id}"  # "20251129_173045_a3b2"

# Step 2: 确定 run 目录
run_dir = Path(f"projects/{project_id}/experiments/{experiment_id}/runs/{run_id}")
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "logs").mkdir(exist_ok=True)
(run_dir / "artifacts").mkdir(exist_ok=True)

# Step 3: 创建 Run 对象并保存
run = Run(
    run_id=run_id,
    project_id=project_id,
    experiment_id=experiment_id,
    created_at=datetime.now(),
    status=RunStatus.PENDING,
    parameters={"density": 1.0, "temperature": 300, "random_seed": 42},
    workflow_snapshot=WorkflowSnapshot(
        git_commit="a3b4c5d",
        workflow_file="md_pipeline.py"
    ),
    working_dir=str(run_dir)
)
(run_dir / "run.json").write_text(run.model_dump_json(indent=2))

# Step 4: 准备 context 和 asset_refs
context = {
    "environment": dict(os.environ),
    "dependencies": get_installed_packages(),
    "hardware": get_hardware_info()
}
(run_dir / "context.json").write_text(json.dumps(context, indent=2))

asset_refs = {"inputs": [], "outputs": []}
(run_dir / "asset_refs.json").write_text(json.dumps(asset_refs, indent=2))

# Step 5: 执行 workflow
run.status = RunStatus.RUNNING
# ... 执行计算 ...

# Step 6: 生成输出 Asset
output_file = run_dir / "trajectory.xtc"
# ... 计算生成 output_file ...

# 计算 hash 并检查是否已存在
content_hash = compute_sha256(output_file)
existing_asset_id = asset_repo.exists(content_hash)

if existing_asset_id:
    # 复用已有 Asset
    asset_id = existing_asset_id
else:
    # 创建新 Asset
    asset_id = str(uuid4())
    asset = Asset(
        asset_id=asset_id,
        type=AssetType.TRAJECTORY,
        format="xtc",
        created_at=datetime.now(),
        producer_run_id=run_id,
        size_bytes=output_file.stat().st_size,
        content_hash=content_hash,
        mime_type="application/x-gromacs-xtc",
        files=[AssetFile(path="data/trajectory.xtc", size=..., hash=content_hash)]
    )
    asset_repo.store(asset, str(output_file))

# Step 7: 更新 asset_refs.json
asset_refs["outputs"].append({
    "asset_id": asset_id,
    "role": "output_trajectory",
    "produced_at": datetime.now().isoformat(),
    "size_bytes": output_file.stat().st_size
})
(run_dir / "asset_refs.json").write_text(json.dumps(asset_refs, indent=2))

# Step 8: 完成 Run
run.status = RunStatus.SUCCEEDED
run.finished_at = datetime.now()
(run_dir / "run.json").write_text(run.model_dump_json(indent=2))
```

### 5.3 重复利用已有 Asset

```python
# 场景：第二次运行要复用第一次运行的输出结构作为输入

# Step 1: 从 AssetRepo 检索
asset_id = "d7e8f9g0-1h2i-3j4k-5l6m-7n8o9p0q1r2s"
asset_meta = asset_repo.get_meta(asset_id)

# Step 2: 在新 Run 的 asset_refs 中记录
new_run_asset_refs["inputs"].append({
    "asset_id": asset_id,
    "role": "input_structure",
    "accessed_at": datetime.now().isoformat()
})

# Step 3: 获取 Asset 实际文件（如需本地访问）
temp_path = run_dir / "inputs" / "structure.pdb"
asset_repo.retrieve(asset_id, str(temp_path))

# 使用 temp_path 进行计算...
```

**关键点**：
- 不需要拷贝 Asset 数据到 run 目录
- AssetRef 中记录引用关系和访问时间
- 支持按 content_hash 查找相同内容的 Asset（去重）

---

## 6. 架构优势总结

### 6.1 科研可复现性

- **完整快照**：Run 保存 workflow、参数、环境、依赖的完整快照
- **血缘追踪**：通过 AssetRef 和 producer_run_id 追踪数据来源
- **时间锚定**：所有对象记录 created_at，支持时间旅行

### 6.2 资产复用与去重

- **全局 Asset 仓库**：跨项目共享力场、结构、模型权重
- **Content-addressable**：基于 hash 自动去重，节省存储
- **轻量引用**：Run 只存 AssetRef，不重复拷贝大文件

### 6.3 云端就绪与可扩展性

- **后端抽象**：AssetRepo 接口支持本地 / S3 / GCS / Azure Blob 切换
- **路径无关**：Asset 通过 asset_id 引用，不依赖文件系统路径
- **水平扩展**：Asset 存储可独立扩展（对象存储）

### 6.4 层次清晰与易维护

- **关注点分离**：Project（课题）、Experiment（设计）、Run（执行）各司其职
- **单向依赖**：Run → Asset，避免循环引用
- **人类友好**：slug + 时间戳命名，目录结构直观

---

## 7. 实现路径建议

### Phase 1: 核心功能（MVP）

1. 实现 Project / Experiment / Run 的基本 CRUD
2. 本地文件系统 AssetRepo（基于目录和 YAML）
3. 手动创建 Run，记录 inputs/outputs AssetRef
4. CLI 工具：`molexp project create`、`molexp run start` 等

### Phase 2: 自动化与集成

1. Workflow executor 集成：自动创建 Run、捕获 Asset
2. Hash-based 去重
3. Web UI 浏览 Projects/Experiments/Runs
4. Asset 搜索和过滤（by tags/type/producer）

### Phase 3: 云端与高级特性

1. S3/GCS AssetRepo 后端
2. 分布式执行（多节点 Run）
3. Asset 版本控制和增量更新
4. Provenance 可视化（血缘图）
