# 3D 群体无人机寻路与避让展示

这是一个独立的 Three.js 静态演示，用于展示 3D 群体无人机在复杂空域中的全局路径规划、局部轨迹优化、机间互避和群体协调。

## 交互结构

- 先按“方法族”划分。
- 每个方法族下的算法都有独立编号。
- 点击方法族会切换到该族的默认算法。
- 点击算法编号会单独切换该算法的展示参数或实现逻辑。
- 单位数量支持 `1 / 10 / 50 / 100 / 500 / 1000`。

## 算法说明卡

右上角是当前算法的学习卡片，切换算法时同步更新：

- **示意图**：手绘 SVG 概念图（栅格搜索波前、采样树与 informed 椭球、速度障碍锥、Voronoi 胞、MPC rollout、承诺轨迹 check/recheck、Boids 三规则、α-lattice、制动曲线、策略 + 安全滤波结构、时空甘特图、二分图分配等）。
- **原理**：2~4 句话讲清核心思想与关键机制，标注出处（Reynolds、Helbing、Olfati-Saber、Mellinger、MADER、EGO-Swarm 等）。
- **特点**：优势 / 局限 / 适用场景三条要点。
- **本演示的实现**：说明页面里实际运行的逻辑与论文完整版的差距，与下方状态表一一对应。
- 右上角 `i` 按钮可折叠卡片；窄屏（<1150px）默认折叠。

## 实现进度

状态说明：

- 已实现：已有独立算法逻辑。
- 近似实现：已有实时浏览器近似版，适合展示，但不是论文级完整求解器。
- 部分实现：已有核心思想或核心可视化，但算法不完整。
- 参数展示：当前只是共用模拟框架，通过参数差异展示效果。
- 未实现：仅保留编号与说明。

| 编号 | 算法 | 方法族 | 当前状态 | 说明 |
|---|---|---|---|---|
| A01 | 3D A* | 搜索 / 采样 | 已实现 | 3D 栅格 26 邻域 A*：开放/关闭列表 + 可采纳启发式 |
| A02 | Hybrid A* | 搜索 / 采样 | 近似实现 | 离散 8 航向状态搜索：航向约束 + 转向代价，非连续曲率 |
| A03 | JPS 3D | 搜索 / 采样 | 近似实现 | 3D 跳点搜索：直线跳跃剪枝 + 简化 forced-neighbor 规则 |
| A04 | RRT* | 搜索 / 采样 | 近似实现 | 实时 RRT*：随机采样树 + 最优父节点选择 + rewire，固定迭代预算 |
| A05 | Informed RRT* | 搜索 / 采样 | 近似实现 | RRT* + 椭球 informed 采样域（有解后聚焦采样） |
| A06 | BIT* | 搜索 / 采样 | 近似实现 | 批量 informed 采样 + 路线图图搜索，简化的 BIT* 边队列 |
| A07 | Kinodynamic A* | 搜索 / 采样 | 近似实现 | (体素 × 速度方向) 晶格搜索，加速度转向约束 |
| A08 | Motion Primitives | 搜索 / 采样 | 近似实现 | 两格弧线运动基元晶格搜索（中点碰撞校验） |
| B01 | Minimum Snap | 轨迹优化 | 已实现 | 闭式分段 7 次 minimum snap 多项式：位置/速度/加速度/jerk 约束 + ESDF 净空投影 |
| B02 | Minimum Jerk | 轨迹优化 | 已实现 | 闭式分段 5 次 minimum jerk 多项式：位置/速度/加速度约束 + ESDF 净空投影 |
| B03 | SFC + Convex QP | 轨迹优化 | 已实现 | 重叠 AABB 安全飞行走廊 + 投影梯度 QP，约束内平滑并保持 ESDF 净空 |
| B04 | B-Spline + ESDF | 轨迹优化 | 近似实现 | 三次均匀 B 样条控制点优化 + ESDF 梯度推离 |
| B05 | Fast-Planner | 轨迹优化 | 近似实现 | B 样条 + ESDF + 曲率自适应限速档 |
| B06 | GCOPTER / MINCO | 轨迹优化 | 近似实现 | 稀疏路标优化 + 样条重建 + 曲率限速（MINCO 思想简化） |
| C01 | 3D ORCA | 分布式避让 | 已实现 | 完整 3D ORCA：邻机 ORCA 半平面 + Dykstra 投影求最近可行速度 |
| C02 | RVO 3D | 分布式避让 | 已实现 | 完整 RVO 3D：截断互惠速度障碍 + 边界投影候选，选择最接近期望的安全速度 |
| C03 | Buffered Voronoi Cells | 分布式避让 | 已实现 | 缓冲 Voronoi 胞：半平面交集 + Dykstra 投影求最近可行速度 |
| C04 | DMPC | 分布式避让 | 已实现 | 滚动预测控制：warm-start 控制序列 + 坐标下降优化，显式处理邻机预测、加速度和障碍约束 |
| C05 | MADER / RMADER | 分布式避让 | 已实现 | 承诺轨迹协议：候选轨迹 check、异步 recheck、旧承诺回滚与制动承诺回退 |
| C06 | EGO-Swarm | 分布式避让 | 已实现 | 局部 B 样条控制点优化：广播轨迹惩罚、拓扑绕行梯度与 ESDF 障碍梯度 |
| C07 | DCP / 错峰通过 | 分布式避让 | 已实现 | 完整 DCP 错峰调度：体素-时间与边交换保留表，优先级追加延迟并二次消解冲突 |
| C08 | 3D HRVO | 分布式避让 | 已实现 | 完整 3D HRVO：混合 VO/RVO 锥顶 + 期望侧切向偏置，速度空间择优安全速度 |
| D01 | APF 3D | 反应式 / 场方法 | 已实现 | 独立人工势场：路径目标吸引 + 邻机二次排斥 + 障碍距离场排斥 |
| D02 | 3D Boids 群体模型 | 反应式 / 场方法 | 已实现 | 已独立实现 separation / alignment / cohesion 三规则 |
| D03 | Olfati-Saber Flocking | 反应式 / 场方法 | 已实现 | α-lattice flocking：σ-norm 梯度项、速度一致项、导航项与 beta-agent 障碍项 |
| D04 | Vasarhelyi Flocking | 反应式 / 场方法 | 已实现 | Vásárhelyi 控制律：短程排斥、制动曲线摩擦、各向异性权重与自推进项 |
| D05 | 3D 社会力模型 | 反应式 / 场方法 | 已实现 | 已独立实现目标驱动力、个体指数排斥、障碍指数排斥 |
| E01 | GLAS | 学习类 | 近似实现 | GLAS 结构：邻域聚合策略 + 危险度安全混合（无学习权重） |
| E02 | PRIMAL / PRIMAL2 | 学习类 | 近似实现 | PRIMAL 结构：27 离散动作按局部观测逐步择优（无网络权重） |
| E03 | Neural CBF | 学习类 | 近似实现 | 解析 CBF 速度滤波：邻机 + 障碍约束顺序投影（无神经网络） |
| E04 | RL + Safety Layer | 学习类 | 近似实现 | 启发式策略动作 + 候选速度安全层，按短时碰撞预测选择可行动作 |
| E05 | End-to-End Swarm RL | 学习类 | 近似实现 | 固定随机权重小型 MLP 策略前向 + 分离保护（非训练权重） |
| F01 | MAPF 3D | 集中式协调 | 已实现 | 3D MAPF 保留表：体素-时间顶点约束 + 边交换约束，按时序规划无冲突通行 |
| F02 | CBS / ECBS | 集中式协调 | 已实现 | CBS/ECBS 约束树：最早顶点/边冲突二分延迟约束，focal bound 控制搜索量 |
| F03 | PBS | 集中式协调 | 已实现 | PBS 优先级搜索：冲突触发优先级约束，拓扑排序后用保留表顺序规划 |
| F04 | SCP 集中式轨迹 | 集中式协调 | 已实现 | SCP：线性化机间分离约束 + 平滑/路径跟踪目标 + ESDF 障碍投影迭代求解 |
| F05 | 匈牙利 / 拍卖分配 | 集中式协调 | 已实现 | Hungarian 精确分配 + 大规模 ε-拍卖后备 + 到分配目标的单目标 A* 路由 |

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



