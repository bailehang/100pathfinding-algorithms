# 100pathfinding-algorithms

Based on the previous work of people, I am very interested in further summarizing all the pathfinding algorithms.

## Env  python3.12
pip install matplotlib

## Thanks

zju Prof. FeiGao

https://github.com/zhm-real/PathPlanning

https://github.com/ai-winter/ros_motion_planning

## Warning
AI assistance has been used in this article, mainly using 
Claude 3.7, Doubao, ChatGPT for in-depth research and Manus.
Manual review and inspection.

-------------------

# Contents

![Contents](images/Contents.png)

## 一、图搜索算法 (Graph Search Algorithms)
-   **基础搜索 (Uninformed Search)**
    -   **广度优先搜索 (Breadth-First Searching - BFS)**
        -   Moore (1959), Lee (1961)
    -   **深度优先搜索 (Depth-First Search - DFS)**
        -   Trémaux (1882), Hopcroft & Tarjan (1973)
-   **启发式搜索 (Informed/Heuristic Search)**
    -   **最佳优先搜索 (Best-First Search - GBFS)**
        -   Doran & Michie (1966), Pearl (1984) (概念性)
    -   **Dijkstra 算法**
        -   Dijkstra (1959)
    -   **A\* 算法 (A\* Algorithm)**
        -   Hart, Nilsson, Raphael (1968)
    -   **A\* 变体与扩展 (A\* Variants & Extensions)**
        -   **双向 A\* (Bidirectional A\*)**
            -   Pohl (1971)
        -   **加权 A\* (Weighted A\*)**
            -   Pohl (1970)
        -   **分层 A\* (Hierarchical A\* - HPA\*)**
            -   Botea, Müller, Schaeffer (2004)
        -   **并行 A\* (Parallel A\*)**
            -   Zhou & Zeng (2015)
        -   **Hybrid A\* **
            -   Dolgov, Thrun, Montemerlo, Diebel (2008)
    -   **实时/动态启发式搜索 (Real-time/Dynamic Heuristic Search)**
        -   **LRTA\* (Learning Real-time A\*)**
            -   Korf (1990)
        -   **D\* 家族 (D\* Family)**
            -   **D\* (Dynamic A\*)**
                -   Stentz (1994)
            -   **Focused D\***
                -   Stentz (1995)
            -   **D\* Lite**
                -   Koenig & Likhachev (2002)
            -   **Anytime D\***
                -   Likhachev, Ferguson, Gordon, Stentz, Thrun (2005)
            -   **Field D\***
                -   Ferguson & Stentz (2007)
        -   **LPA\* (Lifelong Planning A\*)**
            -   Koenig, Likhachev, Furcy (2004)
        -   **ARA\* (Anytime Repairing A\*)**
            -   Likhachev, Gordon, Thrun (2003)
        -   **RTAA\* (Real-time Adaptive A\*)**
            -   Koenig & Likhachev (2006)
    -   **任意角度路径规划 (Any-Angle Path Planning)**
        -   **Theta\* 家族 (Theta\* Family)**
            -   **Theta\***
                -   Nash, Daniel, Koenig, Felner (2007)
            -   **Lazy Theta\***
                -   Nash, Koenig, Tovey (2010)
            -   **S-Theta\***
                -   Tang, Chen, Wu, Zhang, Chen (2021)
            -   **Enhanced Theta\***
                -   Li, Wen, Wang, Zhang (2020)
            -   **Adaptive Theta\***
                -   Ferguson & Stentz (2006)
        -   **JPS 家族 (Jump Point Search Family)**
            -   **JPS (Jump Point Search)**
                -   Harabor & Grastien (2011)
            -   **JPS+**
                -   Harabor & Grastien (2012)
            -   **JPS++**
                -   Pochter, Zohar, Rosenschein, Sturtevant (2012)
            -   **欧几里得 JPS (Euclidean JPS - EJPS)**
                -   Strasser, Botea, Harabor (2016)
            -   **分层 JPS (Hierarchical JPS - HJPS)**
                -   Harabor & Grastien (2014)
            -   **动态 JPS (Dynamic JPS)**
                -   Papadakis (2013)
            -   **JPS-Lite**
                -   Gong, Zhang, Wang, Wang (2019)
            -   **自适应 JPS (Adaptive JPS)**
                -   Su, Hsueh (2016)
    -   **格网规划 (Lattice Planning)**
        -   Pivtoraiko, Kelly (2005), Likhachev & Ferguson (2009)

## 二、基于采样的路径规划 (Sampling-Based Path Planning)
-   **随机路径规划 (Random Path Planning - RPP)**
    -   Barraquand & Latombe (1991)
-   **快速扩展随机树 (Rapidly-Exploring Random Trees - RRT)**
    -   **基础 RRT (Basic RRT)**
        -   LaValle (1998)
    -   **目标偏向 RRT (Goal-bias RRT)**
        -   LaValle & Kuffner (2001) 
    -   **RRT-Connect**
        -   Kuffner & LaValle (2000)
    -   **动态 RRT (Dynamic RRT)**
        -   Ferguson, Howard, Likhachev (2008)
    -   **RRT-Dubins (考虑运动学约束)**
        -   LaValle & Kuffner (2001) (Dubins with RRT)
-   **最优快速扩展随机树 (Optimal RRTs)**
    -   **RRT\***
        -   Karaman & Frazzoli (2011)
    -   **Informed RRT\***
        -   Gammell, Srinivasa, Barfoot (2014)

## 三、智能优化算法 (Intelligent Optimization Algorithms)
-   **蚁群优化 (Ant Colony Optimization - ACO)**
    -   Dorigo, Maniezzo, Colorni (1991, 1996), Dorigo & Di Caro (1999)
-   **遗传算法 (Genetic Algorithm - GA)**
    -   Holland (1975/1992)
-   **粒子群优化 (Particle Swarm Optimization - PSO)**
    -   Kennedy & Eberhart (1995)

## 四、反应式与几何规划 (Reactive & Geometric Planning)
-   **人工势场法 (Artificial Potential Field - APF)**
    -   Khatib (1986)
-   **动态窗口法 (Dynamic Window Approach - DWA)**
    -   Fox, Burgard, Thrun (1997)
-   **向量场直方图 (Vector Field Histogram - VFH)**
    -   Borenstein & Koren (1991)
-   **Voronoi 图方法 (Voronoi Diagram Methods)**
    -   **基础 Voronoi 图 (Basic Voronoi Diagram)**
        -   Voronoi (1908), Shamos & Hoey (1975) 
    -   **Voronoi 场 (Voronoi Field)**
        -   Okabe, Boots, Sugihara, Chiu (2000) 
    -   **加权 Voronoi 图 (Weighted Voronoi Diagram)**
        -   Aurenhammer & Edelsbrunner (1984)
    -   **模糊 Voronoi 图 (Fuzzy Voronoi Diagram)**
        -   Jooyandeh, Mohades Khorasani (2008)
    -   **自适应 Voronoi 场 (Adaptive Voronoi Field)**
        -   Garrido, Moreno, Blanco, Medina (2010)

## 五、基于曲线与运动学的规划 (Curve-Based & Kinematic Planning)
-   **多项式曲线 (Polynomial Curves)**
    -   Richter, Bry, Roy (2013)
-   **贝塞尔曲线 (Bezier Curves)**
    -   Bezier (1960s), Forrest (1972)
-   **样条曲线 (Spline Curves)**
    -   **三次样条曲线 (Cubic Spline)**
        -   Ahlberg, Nilson, Walsh (1967)
    -   **B样条曲线 (B-Spline)**
        -   de Boor (1972), Cox (1972)
-   **时间弹性带 (Timed Elastic Band - TEB)**
    -   Rösmann, Hoffmann, Bertram (2012, 2017)
-   **Dubins 曲线 (Dubins Curves)**
    -   Dubins (1957)
-   **Reeds-Shepp 曲线 (Reeds-Shepp Curves)**
    -   Reeds & Shepp (1990)
-   **车辆路径问题 (Vehicle Routing Problem - VRP)**
    -   Dantzig & Ramser (1959)

## 六、基于模型的控制与规划 (Model-Based Control & Planning)
-   **PID 控制器 (PID Controller - for path following)**
    -   Minorsky (1922), Ziegler & Nichols (1942)
-   **线性二次型调节器 (Linear Quadratic Regulator - LQR)**
    -   Kalman (1960)
-   **模型预测控制 (Model Predictive Control - MPC)**
    -   Cutler & Ramaker (1980), Garcia, Prett, Morari (1989)

## 七、多智能体路径规划 (Multi-Agent Path Finding - MAPF)
-   **基于速度障碍 (Velocity Obstacle - VO) 的方法**
    -   **速度障碍 (VO)**
        -   Fiorini & Shiller (1998)
    -   **相互速度障碍 (Reciprocal Velocity Obstacles - RVO)**
        -   van den Berg, Lin, Manocha (2008)
    -   **混合相互速度障碍 (Hybrid Reciprocal Velocity Obstacles - HRVO)**
        -   Snape, van den Berg, Guy, Manocha (2011)
    -   **最优相互碰撞避免 (Optimal Reciprocal Collision Avoidance - ORCA)**
        -   van den Berg, Guy, Lin, Manocha (2008, 2011)
    -   **行人最优相互碰撞避免 (Pedestrian ORCA - PORCA)**
        -   Luo, Cai, Bera, Hsu, Lee, Manocha (2018)
    -   **椭圆相互速度障碍 (Elliptical Reciprocal Velocity Obstacles - ERVO / EORCA)**
        -   Best, Narang, Manocha (2016)
-   **基于搜索的冲突解决 (Search-Based Conflict Resolution)**
    -   **冲突驱动搜索 (Conflict-Based Search - CBS)**
        -   Sharon, Stern, Felner, Sturtevant (2012, 2015)
    -   **分层协作 A\* (Hierarchical Cooperative A\* - HCA\*)**
        -   Silver (2005)
    -   **窗口化分层协作 A\* (Windowed HCA\* - WHCA\*)**
        -   Silver (2005)

## 八、其他规划方法 (Other Planning Methods)
-   **凸集图规划 (Graph of Convex Sets - GCS / GCS\*)**
    -   Marcucci, Tedrake (2019), Chia, Jiang, Graesdal, Kaelbling, Tedrake (2024)

