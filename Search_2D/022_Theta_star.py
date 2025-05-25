"""
Theta* 2D
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class ThetaStar:
    """
    Theta*: Any-Angle Path Planning on Grids
    Theta* allows paths to go through any angle, not just along grid edges
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        
        # 用于可视化的视线检查记录
        self.los_checks = []
        
        # 用于动态可视化 - 创建图形对象
        self.fig = plt.figure()
        
        # 使用plotting.py中的Plotting类
        self.plot = plotting.Plotting(s_start, s_goal)
        
        # 当前搜索状态
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []

    def searching(self):
        """
        Theta* path searching.
        :return: path, visited order
        """
        # 初始化绘图
        self.plot.plot_grid("Theta*")

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            self.current_visited.append(s)

            # 更新当前路径和可视化
            if s == self.s_goal:
                self.current_path = self.extract_path(self.PARENT)
            else:
                # 显示从起点到当前节点的路径
                temp_path = self.extract_temp_path(s)
                self.current_path = temp_path
            
            # 每隔一定数量的节点更新一次可视化
            if len(self.CLOSED) % 5 == 0 or s == self.s_goal:
                self.update_plot()

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                # 可视化当前正在检查的视线
                los_result = self.line_of_sight(self.PARENT[s], s_n)
                self.los_checks.append((self.PARENT[s], s_n, los_result))
                self.current_los_checks.append((self.PARENT[s], s_n, los_result))
                
                # Path 2 - Try to use parent of current node (line-of-sight checking)
                if los_result:
                    # Line-of-sight exists, consider path from parent
                    new_cost = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf

                    if new_cost < self.g[s_n]:
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = self.PARENT[s]  # Skip a step in the path
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                else:
                    # No line-of-sight, do regular A* update (Path 1)
                    new_cost = self.g[s] + self.cost(s, s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf

                    if new_cost < self.g[s_n]:
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = s
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # 最终更新
        self.update_plot(final=True)
        plt.show()
        
        return self.extract_path(self.PARENT), self.CLOSED

    def update_plot(self, final=False):
        """
        更新绘图，显示当前搜索状态
        """
        # 清除当前图形
        plt.cla()
        
        # 使用plotting.py中的方法绘制网格
        self.plot.plot_grid("Theta*")
        
        # 绘制已访问节点
        if self.current_visited:
            for node in self.current_visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], color='gray', marker='o')
        
        # 绘制视线检查
        if self.current_los_checks:
            for start, end, result in self.current_los_checks:
                color = 'g' if result else 'r'
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.3)
        
        # 绘制当前路径
        if self.current_path:
            self.plot.plot_path(self.current_path)
        
        # 更新图形
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        
        # 最终结果时暂停更长时间
        if final:
            plt.pause(0.5)
        else:
            plt.pause(0.01)
    
    def extract_temp_path(self, current):
        """
        提取从起点到当前节点的临时路径
        """
        path = [current]
        s = current
        
        while s != self.s_start:
            s = self.PARENT[s]
            path.append(s)
        
        return list(reversed(path))

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        nei_list = []
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # Check if points are within grid boundaries
        x_range, y_range = self.Env.x_range, self.Env.y_range
        
        # Check boundary constraints
        if (s_start[0] < 0 or s_start[0] >= x_range or 
            s_start[1] < 0 or s_start[1] >= y_range or
            s_end[0] < 0 or s_end[0] >= x_range or
            s_end[1] < 0 or s_end[1] >= y_range):
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        # Basic check for diagonal segments
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        # Bresenham's line algorithm for more thorough checking
        x0, y0 = s_start
        x1, y1 = s_end
        
        # If the line is steep, transpose the grid
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        # Swap points if needed to ensure x increases
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        y = y0
        
        # Determine step direction
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # Check each point on the line
        for x in range(x0, x1 + 1):
            if steep:
                # If steep, the coordinates are transposed
                if (y, x) in self.obs:
                    return True
            else:
                if (x, y) in self.obs:
                    return True
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        
        return False

    def line_of_sight(self, s_start, s_end):
        """
        check if there is a line-of-sight between two nodes
        :param s_start: start node
        :param s_end: end node
        :return: True if line-of-sight exists
        """
        return not self.is_collision(s_start, s_end)

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    """
    Theta*: Any-Angle Path Planning on Grids
    
    Theta* is an any-angle path planning algorithm that finds paths on a grid.
    Unlike regular A* which can only move along grid edges, Theta* allows paths
    that can go through any angle, creating smoother and more realistic paths.
    
    It does this by checking if there's a line-of-sight between a node's parent
    and its neighbors. If there is, it creates a direct path, bypassing the grid
    constraints and resulting in shorter, more natural paths.
    
    Reference: Nash, A., Daniel, K., Koenig, S., & Felner, A. (2007).
    Theta*: Any-Angle Path Planning on Grids.
    """
    s_start = (5, 5)
    s_goal = (45, 25)

    theta_star = ThetaStar(s_start, s_goal, "euclidean")
    path, visited = theta_star.searching()
    # 不需要再调用animation，因为searching方法已经包含了可视化
    #plot.animation(path, visited, "Theta*")  # animation


if __name__ == '__main__':
    main()
