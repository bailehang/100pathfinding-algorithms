"""
A_star 2D
@author: huiming zhou
"""

import math
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import heapq


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

        # Add boundary obstacles
        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))
        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        # Add additional obstacles
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
    """Plotting class for visualization"""

    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []

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

    def animation_ara_star(self, path, visited, name, save_gif=False):
        """Animation for ARA* algorithm"""
        for i in range(len(path)):
            plt.cla()
            self.plot_grid(name)
            self.plot_visited(visited[i], cl='gray')
            self.plot_path(path[i])
            plt.pause(1)
        
        if save_gif:
            self.save_animation_as_gif(name)
        
        plt.show()

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
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
        """Capture the current figure as a frame with correct color handling"""
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Get the RGBA buffer from the canvas
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        
        # Convert to numpy array
        data = np.frombuffer(buf, dtype=np.uint8)
        
        # Calculate the correct dimensions based on the data size
        # Each pixel has 4 channels (ARGB), so total size = w * h * 4
        # Therefore, each color channel has w * h elements
        total_pixels = len(data) // 4
        
        # Calculate width and height that will work with the data size
        # We can use the aspect ratio from get_width_height() but ensure total pixels match
        aspect_ratio = w / h
        calculated_h = int(np.sqrt(total_pixels / aspect_ratio))
        calculated_w = int(total_pixels / calculated_h)
        
        # Extract color channels
        r = data[1::4]  # Red channel
        g = data[2::4]  # Green channel
        b = data[3::4]  # Blue channel
        
        # Reshape using calculated dimensions
        image = np.stack([r, g, b], axis=-1).reshape((calculated_h, calculated_w, 3))

        # Add to frames
        self.frames.append(image)

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with proper color handling"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames captured: {len(self.frames)}")
        
        # Check absolute path
        abs_path = os.path.abspath(gif_path)
        print(f"Absolute path: {abs_path}")

        # Check if f`rames `list is not empty before saving
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


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.env = Env()  # class Env

        self.u_set = self.env.motions  # feasible input set
        self.obs = self.env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

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

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

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
    s_start = (5, 5)
    s_goal = (45, 25)

    print("Starting A* algorithm")
    astar = AStar(s_start, s_goal, "euclidean")
    plot = Plotting(s_start, s_goal)

    print("Searching for path...")
    path, visited = astar.searching()
    print(f"Path found with {len(path)} nodes, visited {len(visited)} nodes")
    
    print("Starting animation and GIF creation...")
    plot.animation(path, visited, "005_Astar", save_gif=True)  # Save animation as gif
    print("Animation and GIF creation completed")


if __name__ == '__main__':
    main()
