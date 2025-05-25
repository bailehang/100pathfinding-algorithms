"""
Lazy Theta* 2D with Visualization
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class LazyThetaStar:
    """
    Lazy Theta*: Any-Angle Path Planning on Grids
    Lazy Theta*是Theta*的优化版本，推迟视线检查以减少计算开销
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        # 使用导入的env模块中的Env类
        self.Env = env.Env()  # 环境类

        self.u_set = self.Env.motions  # 可行输入集
        self.obs = self.Env.obs  # 障碍物位置

        self.OPEN = []  # 优先队列 / OPEN集
        self.CLOSED = []  # CLOSED集 / 访问顺序
        self.PARENT = dict()  # 记录父节点
        self.g = dict()  # 到达代价
        
        # 用于可视化的视线检查记录
        self.los_checks = []
        
        # 用于动态可视化 - 创建图形对象
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        
        # 使用plotting.py中的Plotting类
        self.plot = plotting.Plotting(s_start, s_goal)
        
        # 当前搜索状态
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []

    def searching(self):
        """
        Lazy Theta*路径搜索
        :return: 路径, 访问顺序
        """
        # 初始化绘图
        self.plot.plot_grid("Lazy Theta*")
        
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            
            # 延迟视线检查
            # 仅在扩展节点时检查视线，而不是在生成节点时
            if s != self.s_start:
                # 验证父节点和当前节点之间是否确实存在视线
                parent = self.PARENT[s]
                
                # 可视化当前正在检查的视线
                self.plot_current_check(parent, s)
                
                los_result = self.line_of_sight(parent, s)
                self.los_checks.append((parent, s, los_result))
                self.current_los_checks.append((parent, s, los_result))
                
                if not los_result:
                    # 如果没有视线，更新父节点为最佳邻居(路径1)
                    self.update_parent(s)

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

            if s == self.s_goal:  # 停止条件
                break

            for s_n in self.get_neighbor(s):
                new_g = math.inf

                # 在Lazy Theta*中，我们乐观地假设存在视线
                # 我们假设路径2有效(从祖父节点存在视线)
                if s != self.s_start:
                    # 路径2(乐观地假设存在视线)
                    new_g = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)
                    
                # 路径1(通过当前节点的传统A*路径)
                new_g_traditional = self.g[s] + self.cost(s, s_n)
                
                # 使用更好的路径(更低的代价)
                if new_g_traditional < new_g:
                    new_g = new_g_traditional
                    # 设置当前节点为父节点(路径1)
                    if s_n not in self.g or new_g < self.g[s_n]:
                        self.g[s_n] = new_g
                        self.PARENT[s_n] = s
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                else:
                    # 设置祖父节点为父节点(路径2) - 乐观地假设存在视线
                    if s_n not in self.g or new_g < self.g[s_n]:
                        self.g[s_n] = new_g
                        self.PARENT[s_n] = self.PARENT[s]  # 跳过路径中的一步
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # 最终更新
        self.update_plot(final=True)
        plt.show()
        
        return self.extract_path(self.PARENT), self.CLOSED, self.los_checks
    
    def plot_current_check(self, start, end):
        """
        可视化当前正在检查的视线
        """
        # 清除之前的临时线
        for artist in self.ax.get_children():
            if hasattr(artist, '_temp_line') and artist._temp_line:
                artist.remove()
        
        # 绘制当前检查的线
        line = self.ax.plot([start[0], end[0]], [start[1], end[1]], 'y-', linewidth=2, alpha=0.8)[0]
        line._temp_line = True
        
        # 更新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(0.1)  # 短暂暂停以显示检查过程
    
    def update_plot(self, final=False):
        """
        更新绘图，显示当前搜索状态
        """
        # 清除当前图形
        plt.cla()
        
        # 使用plotting.py中的方法绘制网格
        self.plot.plot_grid("Lazy Theta*")
        
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

    def update_parent(self, s):
        """
        如果视线检查失败，使用路径1更新父节点
        :param s: 当前节点
        """
        min_g = math.inf
        best_parent = None
        
        for neighbor in self.get_neighbor(s):
            if neighbor in self.g:
                new_g = self.g[neighbor] + self.cost(neighbor, s)
                if new_g < min_g:
                    min_g = new_g
                    best_parent = neighbor
        
        if best_parent is not None:
            self.PARENT[s] = best_parent
            self.g[s] = min_g

    def get_neighbor(self, s):
        """
        查找不在障碍物中的状态s的邻居
        :param s: 状态
        :return: 邻居
        """
        nei_list = []
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # 检查边界约束
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # 过滤掉障碍物和边界违规
                nei_list.append(s_next)
                
        return nei_list

    def cost(self, s_start, s_goal):
        """
        计算此移动的代价
        :param s_start: 起始节点
        :param s_goal: 目标节点
        :return: 此移动的代价
        :note: 代价函数可能更复杂!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        检查线段(s_start, s_end)是否碰撞
        :param s_start: 起始节点
        :param s_end: 结束节点
        :return: True: 碰撞 / False: 不碰撞
        """
        # 检查点是否在网格边界内
        x_range, y_range = self.Env.x_range, self.Env.y_range
        
        # 检查边界约束
        if (s_start[0] < 0 or s_start[0] >= x_range or 
            s_start[1] < 0 or s_start[1] >= y_range or
            s_end[0] < 0 or s_end[0] >= x_range or
            s_end[1] < 0 or s_end[1] >= y_range):
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        # 对对角线线段的基本检查
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        # Bresenham线算法进行更彻底的检查
        x0, y0 = s_start
        x1, y1 = s_end
        
        # 如果线很陡，转置网格
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        # 如果需要，交换点以确保x增加
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        y = y0
        
        # 确定步长方向
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # 检查线上的每个点
        for x in range(x0, x1 + 1):
            if steep:
                # 如果陡峭，坐标被转置
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
        检查两个节点之间是否存在视线
        :param s_start: 起始节点
        :param s_end: 结束节点
        :return: 如果存在视线则为True
        """
        return not self.is_collision(s_start, s_end)

    def f_value(self, s):
        """
        f = g + h. (g: 到达代价, h: 启发式值)
        :param s: 当前状态
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        根据PARENT集提取路径
        :return: 规划路径
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(reversed(path))

    def heuristic(self, s):
        """
        计算启发式值
        :param s: 当前节点(状态)
        :return: 启发式函数值
        """

        heuristic_type = self.heuristic_type  # 启发式类型
        goal = self.s_goal  # 目标节点

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    """
    Lazy Theta*: 网格上的任意角度路径规划
    
    Lazy Theta*是Theta*的优化版本，它推迟视线检查以减少计算开销。
    它的工作原理是乐观地假设存在直接路径，只有当节点从开放列表中实际扩展时才执行视线检查。
    
    这种"延迟"方法减少了视线检查的数量，而视线检查通常是Theta*中最昂贵的操作。
    
    参考: Nash, A., Koenig, S., & Tovey, C. (2010).
    Lazy Theta*: 3D中的任意角度路径规划和路径长度分析。
    """
    s_start = (5, 5)
    s_goal = (45, 25)

    lazy_theta_star = LazyThetaStar(s_start, s_goal, "euclidean")
    path, visited, los_checks = lazy_theta_star.searching()


if __name__ == '__main__':
    main()