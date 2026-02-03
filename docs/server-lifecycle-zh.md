# 服务器生命周期管理

`molexp` 通过 `ServerManager` 类提供对服务器进程生命周期的灵活控制。

## 后台进程行为

当在后台模式启动服务器时，您可以控制在主进程退出后它们是否继续运行：

### 默认行为（守护进程模式）

```python
from molexp.server import ServerManager

manager = ServerManager()
pids = manager.start(
    background=True,
    kill_on_exit=False,  # 默认：进程继续运行
)
# 即使脚本退出，服务器仍然继续运行
```

在此模式下：
- 后台进程在新会话中启动（`start_new_session=True`）
- 它们成为独立于父进程的守护进程
- 主进程退出后仍然存在
- 必须手动使用 `manager.stop()` 停止

### 随主进程退出模式

```python
from molexp.server import ServerManager

manager = ServerManager()
pids = manager.start(
    background=True,
    kill_on_exit=True,  # 新功能：主进程退出时终止
)
# 脚本退出时服务器将自动被终止
```

在此模式下：
- 后台进程保持在同一进程组中
- 主进程退出时自动终止
- 注册清理处理程序以实现优雅关闭
- 适用于临时服务器或测试场景

## API 参考

### ServerManager.start()

```python
def start(
    self,
    host: str = "0.0.0.0",
    port: int = 8000,
    dev: bool = True,
    background: bool = False,
    ui: bool = False,
    sample_data: bool = False,
    kill_on_exit: bool = False,  # 新参数
) -> dict[str, int]:
```

#### 参数

- **host** (`str`): 绑定的主机地址
- **port** (`int`): API 服务器端口号
- **dev** (`bool`): 启用开发模式（自动重载）
- **background** (`bool`): 在后台运行服务器
- **ui** (`bool`): 同时启动 UI 开发服务器
- **sample_data** (`bool`): 启动前创建示例数据
- **kill_on_exit** (`bool`): **新功能** - 如果为 `True` 且 `background=True`，主进程退出时后台进程将被自动终止

#### 返回值

包含 PID 的字典：`{"api": int, "ui": int}`（仅在请求 ui 时）

## 实现细节

### 进程管理

当 `kill_on_exit=True` 时：

1. **进程组**：子进程保持在父进程的进程组中（不使用 `start_new_session`）
2. **PID 跟踪**：后台 PID 被跟踪在 `_background_pids` 列表中
3. **清理注册**：
   - 为正常退出注册 `atexit` 处理程序
   - 为 `SIGTERM` 和 `SIGINT` 注册信号处理程序
4. **优雅关闭**：首先尝试 `SIGTERM`，如有需要再使用 `SIGKILL`

### 信号处理

清理处理程序响应：
- 正常脚本退出（通过 `atexit`）
- `SIGTERM`（终止信号）
- `SIGINT`（Ctrl+C）

## 使用场景

### 1. 生产部署（默认）

```python
# 启动在部署脚本退出后仍然存在的服务器
manager = ServerManager()
manager.start(background=True, kill_on_exit=False)
```

### 2. 测试与开发

```python
# 测试后自动清理服务器
manager = ServerManager()
manager.start(background=True, kill_on_exit=True)
# 运行测试...
# 脚本结束时服务器自动被终止
```

### 3. 临时服务

```python
# 为特定任务启动临时服务器
with TemporaryServer():
    manager = ServerManager()
    manager.start(background=True, kill_on_exit=True)
    # 执行工作...
# 服务器自动清理
```

## 示例

查看 [`examples/server_lifecycle_demo.py`](../examples/server_lifecycle_demo.py) 获取完整演示：

```bash
# 演示默认行为
python examples/server_lifecycle_demo.py --mode default

# 演示随主进程退出行为
python examples/server_lifecycle_demo.py --mode kill-on-exit

# 快速测试
python examples/server_lifecycle_demo.py --mode test-exit
```

## 故障排查

### 退出后进程仍在运行

检查是否启用了 `kill_on_exit`：

```bash
# 检查服务器是否在运行
ps aux | grep uvicorn

# 如有需要手动停止
python -c "from molexp.server import ServerManager; ServerManager().stop()"
```

### 进程未被终止

可能原因：
1. 信号处理程序被阻塞或覆盖
2. 进程变为孤儿进程（检查进程组）
3. 硬崩溃阻止清理（作为最后手段使用 `kill -9`）

调试：
```bash
# 检查进程组
ps -o pid,pgid,command | grep uvicorn
```

## 最佳实践

1. **生产环境**：使用 `kill_on_exit=False`（默认）用于持久服务
2. **测试**：使用 `kill_on_exit=True` 避免孤儿进程
3. **开发**：考虑使用前台模式（`background=False`）便于调试
4. **监控**：启动后始终检查服务器状态：
   ```python
   if not manager.is_running():
       print("服务器启动失败！")
   ```

## 技术说明

### subprocess.Popen 参数

- **start_new_session=False**（当 `kill_on_exit=True`）
  - 子进程与父进程保持在同一会话中
  - 父进程接收到信号时，子进程也会收到
  - 允许父进程控制子进程的生命周期

- **start_new_session=True**（当 `kill_on_exit=False`，默认）
  - 子进程在新会话中启动
  - 成为会话领导者，独立于父进程
  - 父进程退出不影响子进程

### 清理机制

1. **atexit 处理程序**：捕获正常的脚本退出
2. **信号处理程序**：捕获 SIGTERM 和 SIGINT
3. **优雅关闭**：
   - 发送 SIGTERM（优雅终止）
   - 等待最多 5 秒
   - 如果仍在运行，发送 SIGKILL（强制终止）

## 完整代码示例

查看以下文件获取更多示例：
- [`examples/server_lifecycle_demo.py`](../examples/server_lifecycle_demo.py) - 完整演示脚本
- [`examples/server_usage_examples.py`](../examples/server_usage_examples.py) - 使用模式示例
- [`tests/test_server_lifecycle.py`](../tests/test_server_lifecycle.py) - 单元测试
