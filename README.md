# 100pathfinding-algorithms

I am very interested in further summarizing all the pathfinding algorithms.

## Quick Start  
python3.12 or 3.13
pip install matplotlib   
pip install scipy

## Thanks

ZJU Prof. FeiGao

Some code references

https://github.com/zhm-real/PathPlanning

https://github.com/ai-winter/ros_motion_planning

## Warning
AI assistance has been used in this article

Claude 3.7/4, Doubao, ChatGPT for in-depth research and Manus.

2026-05-17 iteration: updated with Codex and ChatGPT 5.5.

Manual review and inspection.

## License

This project is licensed under the Apache License, Version 2.0.

When redistributing this project or substantial portions of it, you must keep
the copyright notice, the `LICENSE` file, and the `NOTICE` attribution file as
required by Apache-2.0.

Required attribution: `100pathfinding-algorithms by bailehang`.

-------------------

# Contents

![Contents](images/Contents.png)

<details>
<summary><strong>一、图搜索算法 (Graph Search Algorithms)</strong></summary>

<details>
<summary>基础搜索 (Uninformed Search)</summary>

- 广度优先搜索 (Breadth-First Searching - BFS)
- 深度优先搜索 (Depth-First Search - DFS)
</details>

<details>
<summary>启发式搜索 (Informed/Heuristic Search)</summary>

- 最佳优先搜索 (Greedy Best-First Search - GBFS)
- Dijkstra 算法
- Flow Fields（流场寻路）
- A* 算法 (A* Algorithm)
  - 传统 A*
  - A* 变体与扩展
    - 加权 A* (Weighted A*)
    - 双向 A* (Bidirectional A*)
    - 分层 A* (Hierarchical A* - HPA*)
    - 并行 A* (Parallel A*)
</details>

<details>
<summary>实时/动态启发式搜索</summary>

- LRTA* (Learning Real-time A*)
- D* 家族
  - D* (Dynamic A*)
  - Focused D*
  - D* Lite
  - Anytime D*
- LPA* (Lifelong Planning A*)
- ARA* (Anytime Repairing A*)
- RTAA* (Real-time Adaptive A*)
</details>

<details>
<summary>任意角度路径规划</summary>

- Theta* 家族
  - Theta*
  - Lazy Theta*
  - S-Theta*
  - Enhanced Theta*
  - Adaptive Theta*
- Field D*
- 导航网格任意角规划 (Navmesh Any-Angle Pathfinding)
  - Polyanya
- JPS 家族
  - JPS (Jump Point Search)
  - JPS+
  - JPS++
  - 欧几里得 JPS (Euclidean JPS - EJPS)
  - 分层 JPS (Hierarchical JPS - HJPS)
  - 动态 JPS (Dynamic JPS)
  - JPS-Lite
  - 自适应 JPS (Adaptive JPS)
</details>

- 格网规划 (Lattice Planning)

</details>

<details>
<summary><strong>二、基于采样的路径规划 (Sampling-Based Path Planning)</strong></summary>

- 随机路径规划 (Random Path Planning - RPP)
- 快速扩展随机树 (Rapidly-Exploring Random Trees - RRT)
  - 基础 RRT (Basic RRT)
  - 目标偏向 RRT (Goal-bias RRT)
  - RRT-Connect
  - 动态 RRT (Dynamic RRT)
  - RRT-Dubins (考虑运动学约束)
- 最优快速扩展随机树 (Optimal RRTs)
  - RRT*
  - Informed RRT*

</details>

<details>
<summary><strong>三、智能优化算法 (Intelligent Optimization Algorithms)</strong></summary>

- 蚁群优化 (Ant Colony Optimization - ACO)
- 遗传算法 (Genetic Algorithm - GA)
- 粒子群优化 (Particle Swarm Optimization - PSO)

</details>


<details>
<summary><strong>四、反应式与几何规划 (Reactive & Geometric Planning)</strong></summary>

- 人工势场法 (Artificial Potential Field - APF)
- 动态窗口法 (Dynamic Window Approach - DWA)
- 向量场直方图 (Vector Field Histogram - VFH)
- Voronoi 图方法
  - 基础 Voronoi 图 (Basic Voronoi Diagram)
  - Voronoi 场 (Voronoi Field)
  - 加权 Voronoi 图 (Weighted Voronoi Diagram)
  - 模糊 Voronoi 图 (Fuzzy Voronoi Diagram)
  - 自适应 Voronoi 场 (Adaptive Voronoi Field)

</details>


<details>
<summary><strong>五、基于曲线与运动学的规划 (Curve-Based & Kinematic Planning)</strong></summary>

- 多项式曲线 (Polynomial Curves)
- 贝塞尔曲线 (Bezier Curves)
- 样条曲线 (Spline Curves)
  - 三次样条曲线 (Cubic Spline)
  - B样条曲线 (B-Spline)
- 时间弹性带 (Timed Elastic Band - TEB)
- Dubins 曲线 (Dubins Curves)
- Reeds-Shepp 曲线 (Reeds-Shepp Curves)
- 特定应用优化
  - Hybrid A*
- 车辆路径问题 (Vehicle Routing Problem - VRP)

</details>


<details>
<summary><strong>六、基于模型的控制与规划 (Model-Based Control & Planning)</strong></summary>

- PID 控制器 (PID Controller - for path following)
- 线性二次型调节器 (Linear Quadratic Regulator - LQR)
- 模型预测控制 (Model Predictive Control - MPC)

</details>



<details>
<summary><strong>七、多智能体路径规划 (Multi-Agent Path Finding - MAPF)</strong></summary>

<details>
<summary>基于速度障碍 (VO) 的方法</summary>

- 速度障碍 (VO)
- 相互速度障碍 (Reciprocal Velocity Obstacles - RVO)
- 混合相互速度障碍 (Hybrid Reciprocal Velocity Obstacles - HRVO)
- 最优相互碰撞避免 (Optimal Reciprocal Collision Avoidance - ORCA)
- 行人最优相互碰撞避免 (Pedestrian ORCA - PORCA)
- 椭圆相互速度障碍 (Elliptical Reciprocal Velocity Obstacles - ERVO / EORCA)
</details>

<details>
<summary>基于搜索的冲突解决</summary>

- 冲突驱动搜索 (Conflict-Based Search - CBS)
- 分层协作 A* (Hierarchical Cooperative A* - HCA*)
- 窗口化分层协作 A* (Windowed HCA* - WHCA*)
</details>

</details>



<details>
<summary><strong>八、其他规划方法 (Other Planning Methods)</strong></summary>

- 凸集图规划 (Graph of Convex Sets - GCS / GCS*)

</details>

---


## 一、图搜索算法 (Graph Search Algorithms)
-   **基础搜索 (Uninformed Search)**
    -   **001 广度优先搜索 (Breadth-First Searching - BFS)**
        -   路径长度：54.042；算法时间：13.400 ms（纯算法，不含动画/GIF 渲染）
        -   Moore (1959), Lee (1961)
        -   ![001_bfs](Search_2D/gif/001_bfs.gif)
        -   算法心得：
        逐层扩展的搜索策略，它从起始节点开始，逐层向外扩展，直到找到目标节点或遍历完整个搜索空间。
    -   **002 深度优先搜索 (Depth-First Search - DFS)**
        -   路径长度：89.213；算法时间：1.337 ms（纯算法，不含动画/GIF 渲染）
        -   Trémaux (1882), Hopcroft & Tarjan (1973)
        -   ![002_dfs](Search_2D/gif/002_dfs.gif)
        -   算法心得：
        从起始节点开始，沿着一条路径尽可能深入，直到无法继续前进，然后回溯到上一个节点，继续探索其他路径。
-   **启发式搜索 (Informed/Heuristic Search)**
    -   **003 贪婪最佳优先搜索 (Greedy Best-First Search - GBFS)**
        -   路径长度：67.255；算法时间：5.040 ms（纯算法，不含动画/GIF 渲染）
        -   Doran & Michie (1966), Pearl (1984)
        -   ![003_GBFS](Search_2D/gif/003_GBFS.gif)
        - 算法心得：
        从起始节点开始，每次选择与目标节点最近的节点进行扩展，直到找到目标节点或遍历完整个搜索空间。
    -   **004 Dijkstra 算法**
        -   路径长度：54.042；算法时间：9.098 ms（纯算法，不含动画/GIF 渲染）
        -   Dijkstra (1959)
        -   ![004_Dijkstra](Search_2D/gif/004_Dijkstra.gif)
        -   算法心得：
        Dijkstra 算法是一种用于寻找最短路径的算法，它通过维护一个距离表来记录从起始节点到其他节点的最短距离。
        算法从起始节点开始，每次选择距离表中距离最小的节点进行扩展，更新其相邻节点的距离表，直到找到目标节点或遍历完整个搜索空间。
    -   **005 Flow Fields（流场寻路）**
        -   路径长度：54.042；算法时间：9.807 ms（纯算法，不含动画/GIF 渲染）
        -   Treuille, Cooper, Popović (2006), 常用于 RTS/Game AI 群体寻路
        -   ![005_Flow_Fields](Search_2D/gif/005_Flow_Fields.gif)
        -   算法心得：
        Flow Fields 先从目标点反向传播总代价，得到 integration field，再让每个可行格子指向代价最低的邻居。多个智能体共享同一目标时，只需复用这张方向场即可快速前进。
    -   **006 A\* 算法 (A\* Algorithm)**
        -   路径长度：54.042；算法时间：4.956 ms（纯算法，不含动画/GIF 渲染）
        -   Hart, Nilsson, Raphael (1968)
        -   ![006_Astar](Search_2D/gif/006_Astar.gif)
        -   算法心得：
        A\* 算法是一种启发式搜索算法，它结合了广度优先搜索和贪婪最佳优先搜索的优点。
        算法从起始节点开始，每次选择距离表中距离最小的节点进行扩展，更新其相邻节点的距离表，直到找到目标节点或遍历完整个搜索空间。
    -   **A\* 变体与扩展 (A\* Variants & Extensions)**
        -   **007 双向 A\* (Bidirectional A\*)**
            -   路径长度：54.042；算法时间：3.558 ms（纯算法，不含动画/GIF 渲染）
            -   Pohl (1971)
            -   ![007_Bidirectional_Astar](Search_2D/gif/007_Bidirectional_a_star.gif)
            -   算法心得：
            它从起始节点和目标节点同时开始进行搜索，直到两个搜索方向相遇。
        
        -   **008 加权 A\* (Weighted A\*)**
            -   路径长度：57.355；算法时间：6.071 ms（纯算法，不含动画/GIF 渲染）
            -   Pohl (1970)
            -   ![008_Weighted_Astar](Search_2D/gif/008_Weighted_Astar_w2.0.gif)
            -   算法心得：
            加权 A\* 算法是 A\* 算法的一种变体，它通过对每个节点的代价进行加权来调整搜索过程。
        -   **009 分层 A\* (Hierarchical A\* - HPA\*)**
            -   路径长度：51.632；算法时间：2.374 ms（纯算法，不含动画/GIF 渲染）
            -   Botea, Müller, Schaeffer (2004)
            -   ![009_Hierarchical_Astar](Search_2D/gif/009_Hierarchical_Astar.gif)
            -   算法心得：
            分层 A\* 算法是 A\* 算法的一种变体，它将搜索空间划分为多个层次，每个层次使用 A\* 算法进行搜索。这里展示两个层次。
        -   **010 并行 A\* (Parallel A\*)**
            -   路径长度：total 142.083, avg 47.361, n=3；算法时间：60.805 ms（纯算法，不含动画/GIF 渲染）
            -   Zhou & Zeng (2015)
            -   ![010_Parallel_Astar](Search_2D/gif/010_Parallel_Astar.gif)
            -   算法心得：
            并行 A\* 算法的优点是可以利用多核处理器的并行计算能力，加快搜索速度。这里难以展示这种并行改为展示多个终点复用查询路径的情况
        -   **011 Hybrid A\***
            -   Dolgov, Thrun, Montemerlo, Diebel (2008)
            -   ![011_Hybrid_Astar](Search_2D/gif/011_Hybrid_Astar.gif)
            -   算法心得：
            混合 A\* 算法有运动约束的寻路，比如汽车的运动约束，
    -   **实时/动态启发式搜索 (Real-time/Dynamic Heuristic Search)**
        -   **012 LRTA\* (Learning Real-time A\*)**
            -   Korf (1990)
            -  ![012_LRTAstar](Search_2D/gif/012_LRTAstar.gif)
            -   算法心得：
            学习式 A\* 算法是 A\* 算法的一种变体，它通过学习来调整搜索过程。
        -   **013 Repairing A\***
            -  Stentz (1994)
            -  ![013_Repairing_Astar](Search_2D/gif/013_Repairing_Astar.gif)
            -   算法心得：
            修复 A\* 算法是 A\* 算法的一种变体，也是D\*，动态环境下它通过修复搜索过程中的错误来提高搜索效率。
        -   **014 LPA\* (Lifelong Planning A\*)**
            -   Koenig, Likhachev, Furcy (2004)
            -   ![014_LPAstar](Search_2D/gif/014_LPAstar.gif)
            -    算法心得：
            终身规划 A\* 算法是 A\* 算法的一种变体，它可以在不断变化的环境中进行搜索。
        -   **015 ARA\* (Anytime Repairing A\*)**
            -   Likhachev, Gordon, Thrun (2003)
            -  ![015_ARAstar](Search_2D/gif/015_ARAstar.gif)
            -   算法心得：
             anytime Repairing A\* 算法是 A\* 算法的一种变体，初始权重较大，路径接近 “次优解”，算法的 “修复” 机制多次调整，self.e -= 0.5。
        -   **016 RTAA\* (Real-time Adaptive A\*)**
            -   Koenig & Likhachev (2006)
            - ![016_RTAAStar](Search_2D/gif/016_RTAAStar.gif)
            -   算法心得：
            实时自适应 A\* 算法是 A\* 算法的一种变体，它可以在不断变化的环境中进行搜索。
        -   **D\* 家族 (D\* Family)**
            -   **017 D\* (Dynamic A\*)**
                -   Stentz (1994)
                -  ![017_Dstar](Search_2D/gif/017_D_star.gif)
                -   算法心得：
                也叫做动态 A\* 算法
            -   **018 Lazy D\***
                -   Koenig, Likhachev, Furcy (2004)
            -   **019 Focused D\***
                -   Stentz (1995)
            -   **020 D\* Lite**
                -   Koenig & Likhachev (2002)
            -   **021 Anytime D\***
                -   Likhachev, Ferguson, Gordon, Stentz, Thrun (2005)
                - ![021_Anytime_Dstar](Search_2D/gif/021_Anytime_D_star.gif)
            -   **022 Field D\***
                -   Ferguson & Stentz (2007)
  
    -   **任意角度路径规划 (Any-Angle Path Planning)**
        -   **ThetA\* 家族 (ThetA\* Family)**
            -   **023 ThetA\***
                -   Nash, Daniel, Koenig, Felner (2007)
                -   ![023_Theta_star](Search_2D/gif/023_Theta_star.gif)
            -   **024 Lazy ThetA\***
                -   Nash, Koenig, Tovey (2010)
                -  ![024_Lazy_Theta_star](Search_2D/gif/024_LazyTheta_star.gif)
            -   **025 S-ThetA\***
                -   Tang, Chen, Wu, Zhang, Chen (2021)
                -  ![025_S_Theta_star](Search_2D/gif/025_STheta_star.gif)
            -   **026 Enhanced ThetA\***
                -   Li, Wen, Wang, Zhang (2020)
                - ![026_Enhanced_Theta_star](Search_2D/gif/026_EnhancedTheta_star.gif)
            -   **027 Multi - Agent Theta**
                -   Li, Zhang, Wang, Zhang (2022)
            -   **028 Adaptive ThetA\***
                -   Ferguson & Stentz (2006)
        -   **导航网格任意角规划 (Navmesh Any-Angle Pathfinding)**
            -   **029 Polyanya**
                -   Cui, Harabor, Grastien (2017)
                -   ![029_Polyanya](Search_2D/gif/029_Polyanya.gif)
                -   算法心得：
                Polyanya 在导航网格上搜索“根点 + 边界区间”，用可见区间传播代替逐格扩展，可得到更接近连续空间最短路径的任意角路线。
        -   **JPS 家族 (Jump Point Search Family)**
            -   **030 JPS (Jump Point Search)**
                -   Harabor & Grastien (2011)
                -  ![030_JPS](Search_2D/gif/030_JPS.gif)
            -   **031 JPS+**
                -   Harabor & Grastien (2012)
            -   **032 JPS++**
                -   Pochter, Zohar, Rosenschein, Sturtevant (2012)
            -   **033 欧几里得 JPS (Euclidean JPS - EJPS)**
                -   Strasser, Botea, Harabor (2016)
            -   **034 分层 JPS (Hierarchical JPS - HJPS)**
                -   Harabor & Grastien (2014)
            -   **035 动态 JPS (Dynamic JPS)**
                -   Papadakis (2013)
            -   **036 JPS-Lite**
                -   Gong, Zhang, Wang, Wang (2019)
            -   **037 Multi - Agent ThetA\**
                -   Li, Zhang, Wang, Zhang (2022)
            -   **038 自适应 JPS (Adaptive JPS)**
                -   Su, Hsueh (2016)
    -   **039 格网规划 (Lattice Planning)**
        -   Pivtoraiko, Kelly (2005), Likhachev & Ferguson (2009)

## 二、基于采样的路径规划 (Sampling-Based Path Planning)
-   **040 随机路径规划 (Random Path Planning - RPP)**
    -   Barraquand & Latombe (1991)
-   **快速扩展随机树 (Rapidly-Exploring Random Trees - RRT)**
    -   **041 基础 RRT (Basic RRT)**
        -   LaValle (1998)
        -   ![041_rrt](Search_2D/gif/041_rrt.gif)
    -   **042 目标偏向 RRT (Goal-bias RRT)**
        -   LaValle & Kuffner (2001) 
        -   ![042_extended_rrt](Search_2D/gif/042_extended_rrt.gif)
    -   **043 RRT-Connect**
        -   Kuffner & LaValle (2000)
        -   ![043_rrt_connect](Search_2D/gif/043_rrt_connect.gif)
    -   **044 动态 RRT (Dynamic RRT)**
        -   Ferguson, Howard, Likhachev (2008)
    -   **045 RRT-Dubins (考虑运动学约束)**
        -   LaValle & Kuffner (2001) (Dubins with RRT)
-   **最优快速扩展随机树 (Optimal RRTs)**
    -   **046 rrt star**
        -   Karaman & Frazzoli (2011)
    -   **047 rrt start smart**
        -  Nasir, K., et al. (2013)
    -   **048 rrt sharp**
        -  Otte & Frazzoli (2014)
    -   **049 Informed RRT\***
        -  Gammell, Srinivasa & Barfoot (2014)
    -   **050 FMT\***
        -  Janson, Schmerling, Clark & Pavone (2015)
    -   **051 BIT\* batch informed trees**
        -  Gammell, Srinivasa & Barfoot (2015)
    -   **052 ABIT\* advanced batch informed trees**
        -  Strub & Gammell (2020)
    -   **053 AIT\* (Adaptively Informed Trees)**
        -  Strub & Gammell (2020)
    -   **054 Anytime-RRT\***
        -   Karaman, Walter, Perez, Frazzoli & Teller (2011)
    -   **055 Closed-loop RRT\* (CL-RRT\*)**
        -   Luders, Kothari & How (2010)
    -   **056 Spline-RRT\***
        -   Lee, Song & Shim (2014)
    -   **057 LQR-RRT\****
        -   Perez, Platt, Konidaris, Kaelbling & Lozano-Perez (2012)

## 三、智能优化算法 (Intelligent Optimization Algorithms)
-   **058 蚁群优化 (Ant Colony Optimization - ACO)**
    -   Dorigo, Maniezzo, Colorni (1991, 1996), Dorigo & Di Caro (1999)
-   **059 遗传算法 (Genetic Algorithm - GA)**
    -   Holland (1975/1992)
-   **060 粒子群优化 (Particle Swarm Optimization - PSO)**
    -   Kennedy & Eberhart (1995)

## 四、反应式与几何规划 (Reactive & Geometric Planning)
-   **061 人工势场法 (Artificial Potential Field - APF)**
    -   Khatib (1986)
-   **062 动态窗口法 (Dynamic Window Approach - DWA)**
    -   Fox, Burgard, Thrun (1997)
-   **063 向量场直方图 (Vector Field Histogram - VFH)**
    -   Borenstein & Koren (1991)
-   **Voronoi 图方法 (Voronoi Diagram Methods)**
    -   **064 基础 Voronoi 图 (Basic Voronoi Diagram)**
        -   Voronoi (1908), Shamos & Hoey (1975) 
        -   ![064_voronoi](Search_2D/gif/064_voronoi.gif)
    -   **065 Voronoi 场 (Voronoi Field)**
        -   Okabe, Boots, Sugihara, Chiu (2000) 
    -   **066 加权 Voronoi 图 (Weighted Voronoi Diagram)**
        -   Aurenhammer & Edelsbrunner (1984)
    -   **067 模糊 Voronoi 图 (Fuzzy Voronoi Diagram)**
        -   Jooyandeh, Mohades Khorasani (2008)
    -   **068 自适应 Voronoi 场 (Adaptive Voronoi Field)**
        -   Garrido, Moreno, Blanco, Medina (2010)

## 五、基于曲线与运动学的规划 (Curve-Based & Kinematic Planning)
-   **069 多项式曲线 (Polynomial Curves)**
    -   Richter, Bry, Roy (2013)
-   **070 贝塞尔曲线 (Bezier Curves)**
    -   Bezier (1960s), Forrest (1972)
-   **样条曲线 (Spline Curves)**
    -   **071 三次样条曲线 (Cubic Spline)**
        -   Ahlberg, Nilson, Walsh (1967)
    -   **072 B样条曲线 (B-Spline)**
        -   de Boor (1972), Cox (1972)
-   **073 时间弹性带 (Timed Elastic Band - TEB)**
    -   Rösmann, Hoffmann, Bertram (2012, 2017)
-   **074 Dubins 曲线 (Dubins Curves)**
    -   Dubins (1957)
-   **075 Reeds-Shepp 曲线 (Reeds-Shepp Curves)**
    -   Reeds & Shepp (1990)
-   **076 车辆路径问题 (Vehicle Routing Problem - VRP)**
    -   Dantzig & Ramser (1959)

## 六、基于模型的控制与规划 (Model-Based Control & Planning)
-   **077 PID 控制器 (PID Controller - for path following)**
    -   Minorsky (1922), Ziegler & Nichols (1942)
-   **078 线性二次型调节器 (Linear Quadratic Regulator - LQR)**
    -   Kalman (1960)
-   **079 模型预测控制 (Model Predictive Control - MPC)**
    -   Cutler & Ramaker (1980), Garcia, Prett, Morari (1989)

## 七、多智能体路径规划 (Multi-Agent Path Finding - MAPF)
-   **基于速度障碍 (Velocity Obstacle - VO) 的方法**
    -   **080 速度障碍 (VO)**
        -   Fiorini & Shiller (1998)
    -   **081 相互速度障碍 (Reciprocal Velocity Obstacles - RVO)**
        -   van den Berg, Lin, Manocha (2008)
    -   **082 混合相互速度障碍 (Hybrid Reciprocal Velocity Obstacles - HRVO)**
        -   Snape, van den Berg, Guy, Manocha (2011)
    -   **083 最优相互碰撞避免 (Optimal Reciprocal Collision Avoidance - ORCA)**
        -   van den Berg, Guy, Lin, Manocha (2008, 2011)
    -   **084 行人最优相互碰撞避免 (Pedestrian ORCA - PORCA)**
        -   Luo, Cai, Bera, Hsu, Lee, Manocha (2018)
    -   **085 椭圆相互速度障碍 (Elliptical Reciprocal Velocity Obstacles - ERVO / EORCA)**
        -   Best, Narang, Manocha (2016)
-   **基于搜索的冲突解决 (Search-Based Conflict Resolution)**
    -   **086 冲突驱动搜索 (Conflict-Based Search - CBS)**
        -   Sharon, Stern, Felner, Sturtevant (2012, 2015)
    -   **087 分层协作 A\* (Hierarchical Cooperative A\* - HCA\*)**
        -   Silver (2005)
    -   **088 窗口化分层协作 A\* (Windowed HCA\* - WHCA\*)**
        -   Silver (2005)
-   **基于社会力模型 (Social Force) 的方法**
    -   **089 UE5 AI Avoidance**
        -   UE5 MassAI MassAvoidanceProcessors (2023)
## 八、其他规划方法 (Other Planning Methods)
-   **090 凸集图规划 (Graph of Convex Sets - GCS / GCS\*)**
    -   Marcucci, Tedrake (2019), Chia, Jiang, Graesdal, Kaelbling, Tedrake (2024)
    -   ![090_GCS](Search_2D/gif/090_Graph_of_Convex_Sets.gif)
-   **091 多智能体凸集图规划 (Multi-Agent Graph of Convex Sets - MGCS / MGCS\*)**
    -   Marcucci, Tedrake (2019), Chia, Jiang, Graesdal, Kaelbling, Tedrake (2024)
-   **092 多智能体多目标规划 (Multi-Agent Multi-Objective Planning - MAMOP)**
    -   Chia, Jiang, Graesdal, Kaelbling, Tedrake (2024)

