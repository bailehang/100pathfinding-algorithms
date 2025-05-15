"""
Depth-first Searching_2D (DFS)
@author: clark bai
"""
import math
import os
import sys
import time
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

            plt.pause(0.02)
            self.capture_frame()

        plt.pause(0.5)
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

        # Convert to numpy array and reshape
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        r = data[1::4]  # Red channel
        g = data[2::4]  # Green channel
        b = data[3::4]  # Blue channel
        image = np.stack([r, g, b], axis=-1).reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Add to frames
        self.frames.append(image)

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with proper color handling"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")

        # Check if frames list is not empty before saving
        if self.frames:
            # Convert NumPy arrays to PIL Images, then to GIF-compatible mode (P with palette)
            frames_p = [Image.fromarray(frame).convert('P', palette=Image.ADAPTIVE, colors=256) for frame in self.frames]

            # Save with proper disposal method to avoid artifacts
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
        else:
            print("No frames to save!")

        # Close the figure
        plt.close()


class DFS:
    """Depth-First Search implementation"""
    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.env = Env()
        self.obs = self.env.obs_map()
        self.visited = []
        self.path = []

    def is_collision(self, p0, p1):
        """Bresenham line sampling to check if there's an obstacle between two points"""
        x0, y0 = p0;  x1, y1 = p1
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1, 1) if x0 < x1 else (-1, 1)
        if y0 > y1: sy = -1
        err = dx - dy
        while True:
            if (x0, y0) in self.obs:            # Hit an obstacle
                return True
            if (x0, y0) == (x1, y1):            # Reached the end point
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def cost(self, s, g):
        """Euclidean distance; returns infinity if the straight line is blocked by an obstacle"""
        if self.is_collision(s, g):
            return math.inf
        return math.hypot(g[0] - s[0], g[1] - s[1])
    # --------------------------------

    def searching(self):
        stack  = [self.s_start]
        parent = {self.s_start: None}

        while stack:
            current = stack.pop()
            self.visited.append(current)
            if current == self.s_goal:
                return self.extract_path(parent), self.visited

            # Generate neighbors
            neighbors = []
            for dx, dy in self.env.motions:
                nb = (current[0] + dx, current[1] + dy)

                # Boundary & obstacle filtering
                if (nb in self.obs or
                    not (0 <= nb[0] < self.env.x_range) or
                    not (0 <= nb[1] < self.env.y_range)):
                    continue
                if nb in self.visited or nb in stack:
                    continue
                neighbors.append(nb)

            # Sort by cost(neighbor, goal) (smaller is better)
            neighbors.sort(key=lambda n: self.cost(n, self.s_goal))

            # Put the most promising node at the top of the stack → popped first
            for nb in reversed(neighbors):
                stack.append(nb)
                parent[nb] = current

        return None, self.visited

    def extract_path(self, parent):
        path, node = [], self.s_goal
        while node:
            path.append(node)
            node = parent[node]
        return path[::-1]

        
def main():
    """Main function to run the BFS algorithm"""
    s_start = (5, 5)
    s_goal = (45, 25)

    bfs = DFS(s_start, s_goal)  # Third parameter is ignored in BFS
    plot = Plotting(s_start, s_goal)

    path, visited = bfs.searching()
    plot.animation(path, visited, "002_dfs", save_gif=True)


if __name__ == '__main__':
    main()