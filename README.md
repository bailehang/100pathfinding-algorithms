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
