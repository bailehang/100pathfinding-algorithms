# 100pathfinding-algorithms

Based on the previous work of people, I am very interested in further summarizing all the pathfinding algorithms.


python3.12


Reference & Thanks

https://github.com/zhm-real/PathPlanning

https://github.com/ai-winter/ros_motion_planning





| 编号 | 缩写         | 英文全称                          | 中文名        | 发表时间 | 发表人                                       | 论文名                                                                    |
|------|--------------|---------------------------------|--------------|----------|--------------------------------------------|-------------------------------------------------------------------------|
| 001  | BFS          | Breadth-First Search            | 广度优先搜索     | 1959     | **E. F. Moore**                           | *The Shortest Path Through a Maze*                                     |
| 002  | DFS          | Depth-First Search              | 深度优先搜索     | 1972     | **R. E. Tarjan**                          | *Depth-First Search and Linear Graph Algorithms*                       |
| 003  | GBFS         | Greedy Best-First Search        | 贪婪最佳优先搜索   | 1966     | **B. W. Doran & D. Michie**               | *Experiments with the Graph Traverser Program*                         |
| 004  | Dijkstra     | Dijkstra’s Algorithm            | Dijkstra算法 | 1959     | **E. W. Dijkstra**                        | *A Note on Two Problems in Connexion with Graphs*                      |
| 005  | A*           | A* Search                      | A* 搜索     | 1968     | **P. E. Hart, N. J. Nilsson, B. Raphael** | *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* |
| 006  | Bi-A*        | Bidirectional A*                | 双向 A*     | 1969     | **I. Pohl**                               | *Bi-Directional Heuristic Search in Path Problems*                     |
| 007  | WA*          | Weighted A*                    | 加权 A*     | 1970     | **I. Pohl**                               | *Heuristic Path Algorithm (HPA) – Weighted A***                       |
| 008  | HPA*         | Hierarchical Path-Finding A*    | 分层 A*     | 2004     | **A. Botea, M. Müller, J. Schaeffer**     | *Near-Optimal Hierarchical Path-Finding (HPA*)*                       |
| 009  | PA*          | Parallel A*                    | 并行 A*     | 1987     | **V. N. Rao & V. Kumar**                  | *Parallel Best-First Search of State-Space Graphs*                     |
| 010  | Hybrid A*    | Hybrid A*                      | 混合 A*     | 2008     | **D. Dolgov, S. Thrun et al.**            | *Practical Search Techniques in Path Planning for Autonomous Driving*  |
| 011  | LRTA*        | Learning Real-Time A*           | 实时学习 A*   | 1990     | **R. E. Korf**                            | *Real-Time Heuristic Search (LRTA*)*                                  |
| 012  | Repair A*    | Repairing A*                   | 修复 A*     | 1994     | **A. Stentz**                             | *Optimal and Efficient Path Planning for Partially-Known Environments* |
| 013  | LPA*         | Lifelong Planning A*           | 终身规划 A*   | 2001     | **S. Koenig & M. Likhachev**              | *Lifelong Planning A**                                                |
| 014  | ARA*         | Anytime Repairing A*          | 任意时修复 A*  | 2003     | **M. Likhachev, G. Gordon, S. Thrun**     | *ARA*: Anytime A* with Provable Bounds on Sub-Optimality*            |
| 015  | RTAA*        | Real-Time Adaptive A*          | 实时自适应 A*  | 2006     | **S. Koenig & X. Sun**                    | *Real-Time Adaptive A**                                               |
| 016  | D*           | Dynamic A*                     | 动态 A*     | 1994     | **A. Stentz**                             | *Optimal and Efficient Path Planning for Partially-Known Environments* |
| 017  | Foc. D*      | Focused D*                     | 焦点 D*     | 1995     | **A. Stentz**                             | *The Focused D* Algorithm for Real-Time Replanning*                   |
| 018  | D* Lite     | D* Lite                       | D* Lite     | 2002     | **S. Koenig & M. Likhachev**              | *D* Lite*                                                             |
| 019  | Any-D*       | Anytime D*                     | 任意时 D*    | 2005     | **M. Likhachev & D. Ferguson**            | *Anytime Dynamic A* (AD*)*                                           |
| 020  | Field D*     | Field D*                       | 场域 D*      | 2006     | **D. Ferguson & A. Stentz**               | *Field D*: An Interpolation-Based Path Planner and Replanner*         |

规划中


21.Theta* Theta*: Any-Angle Path Planning on Grids
22.Lazy Theta* Lazy Theta*: Any-Angle Path Planning and Path Length Analysis in 3D
23.S-Theta* S-Theta: low steering path-planning algorithm
24.Enhanced Theta（增强型 Theta）
25.Multi - Agent Theta（多智能体 Theta）
26.Adaptive Theta（自适应 Theta）

27.JPS  jump point search 跳跃点寻路算法
28.JPS+
29.JPS++  (双向)
30.欧几里得 JPS（Euclidean JPS, EJPS
31.Hierarchical JPS, HJPS）
32.动态 JPS（Dynamic JPS
33.多代理 JPS
34.JPS-Lite
35.自适应 JPS（Adaptive JPS）
36.
37.Voronoi
38.Voronoi Field
39.Weighted Voronoi Diagram
40.Fuzzy Voronoi Diagram（模糊 Voronoi 图）
41.Adaptive Voronoi Field（自适应 Voronoi 场）
42.
43.RRT 快速扩展随机树 Rapidly-Exploring Random Trees: A New Tool for Path Planning
44.Goal-bias RRT    快速找到可行
45.RRT-Connect    双向扩展加速收敛
46.RRT*    渐近最优    离线规划
47.Informed RRT*    最优解收敛更
48.Dynamic RRT    动态环境适应性    
49.RRT-Dubins    考虑运动学约束
50.
51.
52.
53.ACO 蚁群优化 Ant Colony Optimization: A New Meta-Heuristic
54.GA 遗传算法 Adaptation in Natural and Artificial Systems
55.PSO 粒子群优化 Particle Swarm Optimization
56.DWA 动态窗口 The Dynamic Window Approach to Collision Avoidance
57.PID 单积分器动力 Mapping Single-Integrator Dynamics to Unicycle Control Commands 学
58.LQR 线性二次型调节器  Linear Quadratic Regulator 
59.APF  人工势场法 Real-time obstacle avoidance for manipulators and mobile robots
60.RPP 随机路径规划 Random Path Planning 基于随机采样思想
61.TEB 时间弹性带 Timed Elastic Band
62.MPC 模型预测控制 Model Predictive Control
63.Lattice  空间离散寻路

64.Polynomia 多项式函数来描述路径
65.Bezier 贝塞尔曲线路径
66.Cubic Spline 三次样条曲线
67.BSpline B 样条曲线
68.Dubins
69.Reeds-Shepp RS曲线
70. VRP Vehicle Routing Problem
