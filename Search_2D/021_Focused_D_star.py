"""
Focused D* 2D
@author: clark bai
"""

from metrics import install_metrics
install_metrics()

import os
import sys
import math
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class FocusedDStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()
        self.Plot = plotting.Plotting(self.s_start, self.s_goal)

        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.fig = plt.figure()

        self.OPEN = []  # priority queue
        self.t = {}  # state tags: NEW, OPEN, CLOSED
        self.PARENT = {}  # parent pointers
        self.h = {}  # cost to go (from node to goal)
        self.k = {}  # key values
        self.path = []
        self.visited = set()
        self.count = 0
        self.curr_pos = s_start  # current robot position

    def init(self):
        """
        Initialize data structures for the algorithm
        """
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.t[(i, j)] = 'NEW'
                self.k[(i, j)] = float("inf")
                self.h[(i, j)] = float("inf")
                self.PARENT[(i, j)] = None

        self.h[self.s_goal] = 0.0
        self.k[self.s_goal] = self.heuristic(self.s_goal, self.s_start)
        self.OPEN.append((self.k[self.s_goal], self.s_goal))
        self.t[self.s_goal] = 'OPEN'

    def run(self):
        """
        Run the Focused D* algorithm
        """
        self.init()
        self.Plot.plot_grid("Focused D*")

        # Process states until s_start is expanded or OPEN is empty
        while self.OPEN and self.t[self.s_start] != 'CLOSED':
            self.process_state()

        self.path = self.extract_path(self.s_start, self.s_goal)
        self.plot_path(self.path)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def capture_frame(self):
        """Capture the current Matplotlib figure for GIF output."""
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        frame = np.array(image)
        buf.close()
        return frame

    def save_gif(self, frames, name, fps=2):
        """Save captured frames under Search_2D/gif."""
        if not frames:
            print("No frames captured; GIF was not saved.")
            return

        gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        palette_frames = [
            Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256)
            for frame in frames
        ]
        palette_frames[0].save(
            gif_path,
            format="GIF",
            append_images=palette_frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        print(f"GIF animation saved to {gif_path}")

    def draw_state(self, title, highlight=None):
        """Redraw the grid, visited nodes, current path, and optional markers."""
        plt.cla()
        self.Plot.plot_grid(title)
        self.plot_visited(self.visited)
        self.plot_path(self.path)

        if highlight:
            x_values = [p[0] for p in highlight]
            y_values = [p[1] for p in highlight]
            plt.plot(x_values, y_values, "ro", label="Changed obstacle")
            plt.legend()

    def run_demonstration(self):
        """Run a deterministic Focused D* demo and save it as a GIF."""
        print("Starting Focused D* demonstration...")
        frames = []
        plt.figure(figsize=(6, 4), dpi=100)

        self.init()
        step_count = 0
        while self.OPEN and self.t[self.s_start] != 'CLOSED':
            self.process_state()
            step_count += 1
            if step_count % 8 == 0:
                self.draw_state("021 Focused D* - Initial Search")
                frames.append(self.capture_frame())

        self.path = self.extract_path(self.s_start, self.s_goal)
        self.draw_state("021 Focused D* - Initial Path")
        frames.append(self.capture_frame())

        original_path = [p for p in self.path if p not in (self.s_start, self.s_goal)]
        obstacle_sequence = []
        if original_path:
            fractions = (0.35, 0.5, 0.65)
            for fraction in fractions:
                index = min(len(original_path) - 1, max(0, int(len(original_path) * fraction)))
                candidate = original_path[index]
                if candidate not in obstacle_sequence:
                    obstacle_sequence.append(candidate)

        for idx, obstacle in enumerate(obstacle_sequence, start=1):
            print(f"Adding dynamic obstacle {idx}: {obstacle}")
            self.obs.add(obstacle)
            self.Plot.update_obs(self.obs)
            self.curr_pos = self.s_start
            self.visited = set()
            self.count += 1
            self.replan()
            self.draw_state(f"021 Focused D* - Replan {idx}", [obstacle])
            frames.append(self.capture_frame())

        if obstacle_sequence:
            removed = obstacle_sequence[len(obstacle_sequence) // 2]
            print(f"Removing dynamic obstacle: {removed}")
            self.obs.remove(removed)
            self.Plot.update_obs(self.obs)
            self.curr_pos = self.s_start
            self.visited = set()
            self.count += 1
            self.replan()
            self.draw_state("021 Focused D* - Final Path", [removed])
            frames.append(self.capture_frame())

        self.save_gif(frames, "021_Focused_D_star", fps=2)
        plt.close()
        print(f"Focused D* demonstration finished. Final path length: {len(self.path)} nodes")

    def on_press(self, event):
        """
        Handle mouse click events to add and remove obstacles
        """
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            if (x, y) in self.obs:
                print("Remove obstacle at: s =", x, ",", "y =", y)
                self.obs.remove((x, y))
                self.Plot.update_obs(self.obs)
                self.visited = set()
                self.count += 1
                self.replan()
            elif (x, y) not in self.obs:
                print("Add obstacle at: s =", x, ",", "y =", y)
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
                if not self.path:
                    self.curr_pos = self.s_start
                else:
                    self.curr_pos = self.path[0]
                self.visited = set()
                self.count += 1
                path_affected = False
                for i in range(len(self.path) - 1):
                    if self.is_collision(self.path[i], self.path[i + 1]):
                        path_affected = True
                        break
                if path_affected or (x, y) in self.path:
                    print("Path affected by new obstacle. Replanning...")
                    self.replan()

            # Clear and redraw the plot
            plt.cla()
            self.Plot.plot_grid("Focused D*")
            self.plot_visited(self.visited)
            self.plot_path(self.path)

            self.fig.canvas.draw_idle()

    def replan(self):
        """
        Replan the path when obstacles change
        """
        # Reset for complete replanning
        self.OPEN = []

        # Reset all state information
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                node = (i, j)
                if node != self.s_goal:  # Keep goal information
                    self.t[node] = 'NEW'
                    self.h[node] = float("inf")
                    self.PARENT[node] = None
                if node in self.obs:
                    self.h[node] = float("inf")
                    self.k[node] = float("inf")

        # Reinitialize search from goal
        self.h[self.s_goal] = 0.0
        self.k[self.s_goal] = self.heuristic(self.s_goal, self.curr_pos)
        self.OPEN.append((self.k[self.s_goal], self.s_goal))
        self.t[self.s_goal] = 'OPEN'

        # Process all states until start position is expanded or no path exists
        while self.OPEN:
            self.process_state()
            if self.t[self.curr_pos] == 'CLOSED' or not self.OPEN:
                break

        # Extract new path from current position to goal
        self.path = self.extract_path(self.curr_pos, self.s_goal)

    def process_state(self):
        """
        Process the state with minimum k value in OPEN list
        """
        if not self.OPEN:
            return

        # Find state with minimum k value
        self.OPEN.sort(key=lambda x: x[0])
        _, s = self.OPEN.pop(0)
        self.visited.add(s)
        self.t[s] = 'CLOSED'

        # For each neighbor of state s
        for s_n in self.get_neighbor(s):
            if self.t[s_n] == 'NEW':
                self.h[s_n] = float("inf")

            # If neighbor's cost needs to be updated
            if self.h[s] + self.cost(s, s_n) < self.h[s_n]:
                self.h[s_n] = self.h[s] + self.cost(s, s_n)
                self.PARENT[s_n] = s

                # If neighbor already in OPEN list, update its key
                if self.t[s_n] == 'OPEN':
                    # Remove old entry
                    self.OPEN = [item for item in self.OPEN if item[1] != s_n]

                # Add or update neighbor in OPEN list
                self.k[s_n] = self.h[s_n] + self.heuristic(s_n, self.curr_pos)
                self.OPEN.append((self.k[s_n], s_n))
                self.t[s_n] = 'OPEN'

    def extract_path(self, s_start, s_end):
        """
        Extract path from s_start to s_end based on parent pointers
        """
        path = [s_start]
        s = s_start

        # Safety check to prevent infinite loops
        max_iterations = self.x * self.y
        iteration = 0

        while s != s_end and iteration < max_iterations:
            # Check if current node has a parent
            if self.PARENT[s] is None:
                # No path exists - try to find an alternative path
                # Find the best next step based on heuristic costs
                neighbors = self.get_neighbor(s)
                if not neighbors:
                    print("No valid path found - blocked by obstacles")
                    return path  # Return partial path

                best_neighbor = None
                min_cost = float("inf")

                for neighbor in neighbors:
                    # Calculate cost through this neighbor
                    cost = self.h[neighbor]
                    if cost < min_cost:
                        min_cost = cost
                        best_neighbor = neighbor

                if best_neighbor is None or min_cost == float("inf"):
                    print("No valid path to goal")
                    return path  # Return partial path

                s = best_neighbor
            else:
                s = self.PARENT[s]

            path.append(s)
            iteration += 1

            if s == s_end:
                break

        if iteration >= max_iterations:
            print("Path extraction reached maximum iterations")

        return path

    def get_neighbor(self, s):
        """
        Get neighbors of state s that are not obstacles
        """
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if 0 <= s_next[0] < self.x and 0 <= s_next[1] < self.y and s_next not in self.obs:
                nei_list.add(s_next)
        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate cost between two states
        """
        if self.is_collision(s_start, s_goal):
            return float("inf")
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if path between s_start and s_end collides with obstacles
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

    def heuristic(self, s, goal):
        """
        Calculate heuristic distance
        """
        if self.heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def plot_path(self, path):
        """
        Plot the path
        """
        if path:
            px = [x[0] for x in path]
            py = [x[1] for x in path]
            plt.plot(px, py, linewidth=2)
            plt.plot(self.s_start[0], self.s_start[1], "bs")
            plt.plot(self.s_goal[0], self.s_goal[1], "gs")

    def plot_visited(self, visited):
        """
        Plot visited states
        """
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    fdstar = FocusedDStar(s_start, s_goal, "euclidean")
    fdstar.run_demonstration()


if __name__ == '__main__':
    main()
