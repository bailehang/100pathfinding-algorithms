# 3D 群体无人机寻路与避让展示

这是一个独立的 Three.js 静态演示，用于展示 3D 群体无人机在复杂空域中的全局路径规划、局部轨迹优化、机间互避和群体协调。

## 交互结构

- 先按“方法族”划分。
- 每个方法族下的算法都有独立编号。
- 点击方法族会切换到该族的默认算法。
- 点击算法编号会单独切换该算法的展示参数或实现逻辑。
- 单位数量支持 `1 / 10 / 50 / 100 / 500 / 1000`。

## 实现进度

状态说明：

- 已实现：已有独立算法逻辑。
- 近似实现：已有实时浏览器近似版，适合展示，但不是论文级完整求解器。
- 部分实现：已有核心思想或核心可视化，但算法不完整。
- 参数展示：当前只是共用模拟框架，通过参数差异展示效果。
- 未实现：仅保留编号与说明。

| 编号 | 算法 | 方法族 | 当前状态 | 说明 |
|---|---|---|---|---|
| A01 | 3D A* | 搜索 / 采样 | 部分实现 | 当前是 3D 栅格多源距离场 + 贪心下降路径，不是严格 A* open/closed list |
| A02 | Hybrid A* | 搜索 / 采样 | 参数展示 | 尚无航向、曲率、运动学状态搜索 |
| A03 | JPS 3D | 搜索 / 采样 | 参数展示 | 尚无 3D Jump Point 剪枝，当前通过速度 / 前瞻参数展示效果 |
| A04 | RRT* | 搜索 / 采样 | 参数展示 | 尚无采样树、rewire、渐进优化 |
| A05 | Informed RRT* | 搜索 / 采样 | 参数展示 | 尚无椭球启发式采样域 |
| A06 | BIT* | 搜索 / 采样 | 参数展示 | 尚无批量采样图搜索 |
| A07 | Kinodynamic A* | 搜索 / 采样 | 参数展示 | 尚无速度/加速度状态空间搜索 |
| A08 | Motion Primitives | 搜索 / 采样 | 参数展示 | 尚无 motion primitive 库 |
| B01 | Minimum Snap | 轨迹优化 | 参数展示 | 尚无多项式 QP 轨迹优化 |
| B02 | Minimum Jerk | 轨迹优化 | 参数展示 | 尚无 jerk 目标函数优化 |
| B03 | SFC + Convex QP | 轨迹优化 | 部分实现 | 有安全走廊可视化，尚无凸优化求解 |
| B04 | B-Spline + ESDF | 轨迹优化 | 参数展示 | 尚无 B 样条控制点优化和 ESDF |
| B05 | Fast-Planner | 轨迹优化 | 参数展示 | 尚无 Fast-Planner 前后端 |
| B06 | GCOPTER / MINCO | 轨迹优化 | 参数展示 | 尚无 MINCO 稀疏参数化 |
| C01 | 3D ORCA | 分布式避让 | 近似实现 | 已加入 3D 速度障碍实时投影与互惠修正，未做完整线性规划 |
| C02 | RVO 3D | 分布式避让 | 参数展示 | 尚无完整 RVO 几何构造 |
| C03 | Buffered Voronoi Cells | 分布式避让 | 参数展示 | 尚无 Voronoi cell 约束 |
| C04 | DMPC | 分布式避让 | 参数展示 | 尚无滚动优化和预测轨迹约束 |
| C05 | MADER / RMADER | 分布式避让 | 参数展示 | 尚无 check-recheck 协议 |
| C06 | EGO-Swarm | 分布式避让 | 参数展示 | 尚无广播轨迹优化 |
| C07 | DCP / 错峰通过 | 分布式避让 | 部分实现 | 已有 startDelay 错峰效果，尚无冲突时空分配 |
| C08 | 3D HRVO | 分布式避让 | 近似实现 | 已加入 HRVO 风格混合 apex 与切向偏置，未做完整 HRVO 几何求解 |
| D01 | APF 3D | 反应式 / 场方法 | 部分实现 | 已有目标吸引、个体排斥、障碍排斥 |
| D02 | 3D Boids 群体模型 | 反应式 / 场方法 | 已实现 | 已独立实现 separation / alignment / cohesion 三规则 |
| D03 | Olfati-Saber Flocking | 反应式 / 场方法 | 参数展示 | 尚无 sigma-norm、邻接图和理论控制律 |
| D04 | Vasarhelyi Flocking | 反应式 / 场方法 | 参数展示 | 尚无论文参数模型 |
| D05 | 3D 社会力模型 | 反应式 / 场方法 | 已实现 | 已独立实现目标驱动力、个体指数排斥、障碍指数排斥 |
| E01 | GLAS | 学习类 | 参数展示 | 尚无学习策略模型 |
| E02 | PRIMAL / PRIMAL2 | 学习类 | 参数展示 | 尚无 RL/IL 网络或 MAPF 策略 |
| E03 | Neural CBF | 学习类 | 参数展示 | 尚无 CBF 约束求解 |
| E04 | RL + Safety Layer | 学习类 | 部分实现 | 仅有简单预测碰撞过滤 |
| E05 | End-to-End Swarm RL | 学习类 | 参数展示 | 尚无端到端策略模型 |
| F01 | MAPF 3D | 集中式协调 | 参数展示 | 尚无离散多智能体路径规划 |
| F02 | CBS / ECBS | 集中式协调 | 参数展示 | 尚无冲突树搜索，当前以分层 / 错峰参数展示 |
| F03 | PBS | 集中式协调 | 参数展示 | 尚无优先级约束搜索，当前以分层 / 错峰参数展示 |
| F04 | SCP 集中式轨迹 | 集中式协调 | 参数展示 | 尚无序列凸规划，当前以分层 / 错峰参数展示 |
| F05 | 匈牙利 / 拍卖分配 | 集中式协调 | 参数展示 | 尚无目标分配求解，当前以分层 / 错峰参数展示 |

## 算法编号

### 1. 搜索 / 采样类全局规划

- A01: 3D A*
- A02: Hybrid A*
- A03: JPS 3D
- A04: RRT*
- A05: Informed RRT*
- A06: BIT*
- A07: Kinodynamic A*
- A08: Motion Primitives

### 2. 轨迹优化后端

- B01: Minimum Snap
- B02: Minimum Jerk
- B03: SFC + Convex QP
- B04: B-Spline + ESDF
- B05: Fast-Planner
- B06: GCOPTER / MINCO

### 3. 分布式机间避让

- C01: 3D ORCA
- C02: RVO 3D
- C03: Buffered Voronoi Cells
- C04: DMPC
- C05: MADER / RMADER
- C06: EGO-Swarm
- C07: DCP / 错峰通过
- C08: 3D HRVO

### 4. 反应式 / 场方法

- D01: APF 3D
- D02: 3D Boids 群体模型
- D03: Olfati-Saber Flocking
- D04: Vasarhelyi Flocking
- D05: 3D 社会力模型

### 5. 学习类

- E01: GLAS
- E02: PRIMAL / PRIMAL2
- E03: Neural CBF
- E04: RL + Safety Layer
- E05: End-to-End Swarm RL

### 6. 集中式 / 编队与任务层

- F01: MAPF 3D
- F02: CBS / ECBS
- F03: PBS
- F04: SCP 集中式轨迹
- F05: 匈牙利 / 拍卖分配

## 运行

推荐直接双击：

```text
start_server.bat
```

它会自动打开：

```text
http://127.0.0.1:5173/
```

也可以手动在 `Search_3D` 目录下启动：

```powershell
C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m http.server 5173 --bind 127.0.0.1
```

页面使用 CDN 加载 Three.js，需要浏览器可以访问 jsDelivr。
