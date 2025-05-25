"""
Lazy Theta* 2D with Visualization
@author: clark bai
"""

import io
import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import time


class Env:
    """Environment class for 2D grid world"""
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs


class Plotting:
    """Plotting class for visualization with GIF generation capability"""

    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = (6, 4)

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name, save_gif=False):
        """Animate the search process and final path"""
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()
        if save_gif:
            self.save_animation_as_gif(name)

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
        # Create figure with fixed size
        plt.figure(figsize=self.fig_size, dpi=100, clear=True)
        
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

        # Capture the initial grid frame
        self.capture_frame()

    def plot_visited(self, visited, cl='gray'):
        """Plot visited nodes during search"""
        if self.xI in visited:
            visited.remove(self.xI)
        if self.xG in visited:
            visited.remove(self.xG)

        count = 0
        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                plt.pause(0.01)
                self.capture_frame()

        plt.pause(0.1)
        self.capture_frame()

    def plot_path(self, path, cl='r', flag=False):
        """Plot the final path"""
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.1)
        self.capture_frame()

    def capture_frame(self):
        buf = io.BytesIO()
        
        # Get the current figure
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Save the figure to a buffer with a standard DPI
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Open the image using PIL and convert to RGB
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        
        # Convert to numpy array
        image = np.array(img_rgb)
        
        # Add to frames
        self.frames.append(image)
        
        # Close the buffer
        buf.close()

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with consistent size"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames captured: {len(self.frames)}")
        
        # Verify all frames have the same dimensions
        if self.frames:
            first_frame_shape = self.frames[0].shape
            for i, frame in enumerate(self.frames):
                if frame.shape != first_frame_shape:
                    print(f"WARNING: Frame {i} has inconsistent shape: {frame.shape} vs {first_frame_shape}")
                    # Resize inconsistent frames to match the first frame
                    resized_frame = np.array(Image.fromarray(frame).resize(
                        (first_frame_shape[1], first_frame_shape[0]), 
                        Image.LANCZOS))
                    self.frames[i] = resized_frame
        
        # Check if frames list is not empty before saving
        if self.frames:
            try:
                # Convert NumPy arrays to PIL Images
                print("Converting frames to PIL Images...")
                frames_p = []
                for i, frame in enumerate(self.frames):
                    try:
                        img = Image.fromarray(frame)
                        img_p = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                        frames_p.append(img_p)
                        if i % 10 == 0:
                            print(f"Converted frame {i+1}/{len(self.frames)}")
                    except Exception as e:
                        print(f"Error converting frame {i}: {e}")
                
                print(f"Successfully converted {len(frames_p)} frames")

                # Save with proper disposal method to avoid artifacts
                print("Saving GIF file...")
                frames_p[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames_p[1:],
                    save_all=True,
                    duration=int(1000 / fps),
                    loop=0,
                    disposal=2  # Replace previous frame
                )
                print(f"GIF animation saved to {gif_path}")
                
                # Verify file was created
                if os.path.exists(gif_path):
                    print(f"File exists with size: {os.path.getsize(gif_path) / 1024:.2f} KB")
                else:
                    print("WARNING: File does not exist after saving!")
            except Exception as e:
                print(f"Error during GIF creation: {e}")
        else:
            print("No frames to save!")

        # Close the figure
        plt.close()


class LazyThetaStar:
    """
    Lazy Theta*: Any-Angle Path Planning on Grids
    Lazy Theta*是Theta*的优化版本，推迟视线检查以减少计算开销
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        # 使用内置的Env类
        self.Env = Env()  # 环境类

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
        
        # 使用内置的Plotting类
        self.plot = Plotting(s_start, s_goal)
        
        # 当前搜索状态
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []

    def searching(self):
        """
        Lazy Theta*路径搜索 with gif generation
        :return: 路径, 访问顺序
        """
        print("Starting Lazy Theta* algorithm with GIF generation...")
        
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

        # 最终更新并生成GIF
        self.update_plot(final=True)
        
        path = self.extract_path(self.PARENT)
        print(f"Path found with {len(path)} nodes, visited {len(self.CLOSED)} nodes")
        
        # 生成GIF
        print("Generating GIF animation...")
        self.plot.save_animation_as_gif("023_LazyTheta_star")
        
        return path, self.CLOSED, self.los_checks
    
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
        
        # 重新绘制网格
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title("Lazy Theta*")
        plt.axis("equal")
        
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
            path_x = [self.current_path[i][0] for i in range(len(self.current_path))]
            path_y = [self.current_path[i][1] for i in range(len(self.current_path))]
            plt.plot(path_x, path_y, linewidth='3', color='r')
        
        # 重新绘制起点和终点以确保它们在最上层
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        
        # 捕获帧
        self.plot.capture_frame()
        
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
    
    print("Lazy Theta* algorithm completed successfully!")
    print(f"Final path length: {len(path)} nodes")
    print(f"Total nodes visited: {len(visited)} nodes")
    print(f"Total line-of-sight checks: {len(los_checks)} checks")


if __name__ == '__main__':
    main()
