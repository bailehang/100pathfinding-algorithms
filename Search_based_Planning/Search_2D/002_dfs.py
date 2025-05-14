"""
Depth-first Searching_2D (DFS)
@author: clark bai
"""
import math
import os
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

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                plt.pause(0.001)
                self.capture_frame()

        plt.pause(0.001)
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
        print(f"Number of frames to save: {len(self.frames)}")

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
    """Depth-First Search implementation with shortest path tracking"""
    def __init__(self, s_start, s_goal, _):
        # Initialize parameters
        self.s_start = s_start
        self.s_goal = s_goal

        # Initialize environment
        self.Env = Env()
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles

        # Initialize sets and dictionaries
        self.OPEN = []            # stack for DFS (LIFO)
        self.OPEN_set = set()     # set for O(1) lookups in OPEN
        self.CLOSED = set()       # visited nodes as a set for O(1) lookup
        self.PARENT = dict()      # parent nodes for path reconstruction
        self.g = dict()           # cost to come (needed for shortest path)

    def searching(self):
        """DFS algorithm modified to maintain shortest path information"""
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # Cost from start to start is 0
        self.OPEN.append(self.s_start)
        self.OPEN_set.add(self.s_start)
        visited_list = []  # For animation purposes

        while self.OPEN:
            s = self.OPEN.pop()  # LIFO for DFS
            self.OPEN_set.remove(s)

            # Add to visualization
            if s not in self.CLOSED:
                visited_list.append(s)

            # Add to closed set
            self.CLOSED.add(s)

            # Check if goal reached
            if s == self.s_goal:
                break

            # Explore neighbors
            for s_n in self.get_neighbor(s):
                # Calculate new cost to this neighbor
                new_cost = self.g[s] + self.calculate_distance(s, s_n)

                # Check if new cost is better or node is unvisited
                if s_n not in self.g or new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost  # Update cost
                    self.PARENT[s_n] = s    # Update parent

                    # If we find a better path to a node in CLOSED,
                    # remove it from CLOSED to reconsider it
                    if s_n in self.CLOSED:
                        self.CLOSED.remove(s_n)

                    # Add to OPEN if not already there (using O(1) set lookup)
                    if s_n not in self.OPEN_set:
                        self.OPEN.append(s_n)
                        self.OPEN_set.add(s_n)

        return self.extract_path(self.PARENT), list(visited_list)

    def get_neighbor(self, s):
        """Get valid neighboring nodes - optimized for speed without heuristics"""
        neighbors = []

        # Generate all possible neighbors
        for u in self.u_set:
            s_n = (s[0] + u[0], s[1] + u[1])

            # Check if valid move (not collision)
            if not self.is_collision(s, s_n):
                neighbors.append(s_n)

        return neighbors

    def calculate_distance(self, s_start, s_goal):
        """Calculate Euclidean distance between two points - for path cost calculation"""
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """Check if there's a collision between two nodes - optimized"""
        # Quick check for endpoint obstacles
        if s_start in self.obs or s_end in self.obs:
            return True

        # Only check diagonal movements (more expensive)
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            # Calculate diagonal passing cells
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            # Check if either diagonal cell is an obstacle
            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def extract_path(self, PARENT):
        """Extract path from parent dictionary - optimized version"""
        # Check if path exists
        if self.s_goal not in PARENT:
            return []

        # Reconstruct path efficiently
        path = [self.s_goal]
        s = self.s_goal

        while s != self.s_start:
            s = PARENT[s]
            path.append(s)

        return path


def main():
    """Main function to run the DFS algorithm"""
    s_start = (5, 5)
    s_goal = (45, 25)

    dfs = DFS(s_start, s_goal, None)  # Third parameter is ignored in DFS
    plot = Plotting(s_start, s_goal)

    path, visited = dfs.searching()
    plot.animation(path, visited, "002_dfs", save_gif=True)


if __name__ == '__main__':
    main()
