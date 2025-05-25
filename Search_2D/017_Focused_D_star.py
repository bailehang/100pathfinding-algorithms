"""
Focused D* 2D
@author: clark bai
"""

import os
import math
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image


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
    """Plotting class for visualization with GIF support"""

    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = (6, 4)

    def update_obs(self, obs):
        self.obs = obs

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs", label="Start")
        plt.plot(self.xG[0], self.xG[1], "gs", label="Goal")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name, fontsize=14)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()

    def plot_visited(self, visited, color='lightblue'):
        """Plot visited nodes during search"""
        if not visited:
            return
            
        # Remove start and goal from visited for cleaner visualization
        visited_clean = [x for x in visited if x != self.xI and x != self.xG]
        
        if visited_clean:
            for x in visited_clean:
                plt.plot(x[0], x[1], marker='o', color=color, alpha=0.7)

    def plot_path(self, path, color='red', linewidth=3):
        """Plot the final path"""
        if not path or len(path) < 2:
            return
            
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        
        plt.plot(path_x, path_y, linewidth=linewidth, color=color, label="Path")

    def capture_frame(self):
        """Capture current plot as frame for GIF"""
        buf = io.BytesIO()
        
        # Get the current figure
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Save the figure to buffer
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Open image and convert to RGB
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        
        # Convert to numpy array
        image = np.array(img_rgb)
        
        # Add to frames
        self.frames.append(image)
        
        # Close buffer
        buf.close()

    def save_animation_as_gif(self, name, fps=2):
        """Save frames as GIF animation"""
        # Create gif directory
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames captured: {len(self.frames)}")
        
        if self.frames:
            try:
                # Convert frames to PIL Images
                frames_p = []
                for i, frame in enumerate(self.frames):
                    img = Image.fromarray(frame)
                    img_p = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                    frames_p.append(img_p)
                
                # Save GIF
                frames_p[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames_p[1:],
                    save_all=True,
                    duration=int(1000 / fps),
                    loop=0,
                    disposal=2
                )
                print(f"GIF animation saved to {gif_path}")
                
                if os.path.exists(gif_path):
                    print(f"File size: {os.path.getsize(gif_path) / 1024:.2f} KB")
                
            except Exception as e:
                print(f"Error during GIF creation: {e}")
        else:
            print("No frames to save!")


class FocusedDStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()
        self.Plot = Plotting(self.s_start, self.s_goal)

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

    def run_demonstration(self):
        """
        Run complete demonstration with adding obstacles, then removing them
        """
        print("Starting Focused D* complete demonstration...")
        
        # Create figure
        plt.figure(figsize=self.Plot.fig_size, dpi=100)
        
        # Part 1: Initial planning phase
        print("PART 1: Initial Focused D* planning...")
        self.init()

        # Process states until s_start is expanded or OPEN is empty
        step_count = 0
        while self.OPEN and self.t[self.s_start] != 'CLOSED':
            self.process_state()
            step_count += 1
            
            # Capture intermediate frames during search
            if step_count % 5 == 0:  # Capture every 5 steps
                plt.cla()
                self.Plot.plot_grid("017 Focused D* - Initial Search")
                self.Plot.plot_visited(self.visited, 'lightblue')
                if self.path:
                    self.Plot.plot_path(self.path)
                self.Plot.capture_frame()

        self.path = self.extract_path(self.s_start, self.s_goal)
        
        # Plot initial state
        plt.cla()
        self.Plot.plot_grid("017 Focused D* - Initial Path")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        self.Plot.capture_frame()
        
        # Part 2: Add obstacles sequence that require focused search
        print("PART 2: Adding obstacles and showing focused replanning...")
        obstacle_sequence = [
            (18, 16), (32, 16), (35, 18), (25, 14), (28, 13), (15, 20)
        ]
        
        for step, (x, y) in enumerate(obstacle_sequence):
            print(f"  Adding obstacle {step + 1}: ({x}, {y})")
            
            if x >= 0 and x < self.x and y >= 0 and y < self.y and (x, y) not in self.obs:
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
                
                # Update current position and check if replanning is needed
                if not self.path:
                    self.curr_pos = self.s_start
                else:
                    self.curr_pos = self.path[0] if self.path else self.s_start
                
                self.visited = set()
                self.count += 1
                
                # Check if path is affected
                path_affected = False
                if self.path:
                    for i in range(len(self.path) - 1):
                        if self.is_collision(self.path[i], self.path[i + 1]):
                            path_affected = True
                            break
                    if (x, y) in self.path:
                        path_affected = True

                if path_affected:
                    print(f"    Path affected, replanning using Focused D*...")
                    self.replan()
                
                # Extract updated path
                self.path = self.extract_path(self.curr_pos, self.s_goal)
                
                # Plot current state
                plt.cla()
                self.Plot.plot_grid(f"Adding Obstacle {step + 1}")
                self.Plot.plot_visited(self.visited, 'lightblue')
                self.Plot.plot_path(self.path)
                plt.plot(x, y, "ro", label="New Obstacle")
                plt.legend()
                
                self.Plot.capture_frame()
        
        # Part 3: Add some blocking obstacles for removal demo
        print("PART 3: Adding blocking obstacles...")
        blocking_obstacles = [
            (22, 15), (22, 16), (22, 17),  # First block
            (12, 18), (13, 18), (14, 18),  # Second block
            (35, 12), (35, 13), (35, 14)   # Third block
        ]
        
        for obs in blocking_obstacles:
            if obs not in self.obs:
                self.obs.add(obs)
        self.Plot.update_obs(self.obs)
        
        # Replan with all obstacles
        self.curr_pos = self.s_start
        self.visited = set()
        self.replan()
        self.path = self.extract_path(self.curr_pos, self.s_goal)
        initial_path_length = len(self.path) if self.path else 0
        
        plt.cla()
        self.Plot.plot_grid("With All Blocking Obstacles")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        self.Plot.capture_frame()
        
        # Part 4: Remove obstacles and show path improvement
        print("PART 4: Removing obstacles and improving path...")
        removal_sequence = [
            [(22, 15), (22, 16), (22, 17)],  # Remove first block
            [(12, 18), (13, 18), (14, 18)],  # Remove second block
            [(35, 12), (35, 13), (35, 14)]   # Remove third block
        ]
        
        removal_names = ["First Block", "Second Block", "Third Block"]
        
        for group_idx, obstacle_group in enumerate(removal_sequence):
            print(f"  Removing {removal_names[group_idx]}: {obstacle_group}")
            
            # Remove obstacles
            removed_obstacles = []
            for obs in obstacle_group:
                if obs in self.obs:
                    self.obs.remove(obs)
                    removed_obstacles.append(obs)
            
            if removed_obstacles:
                self.Plot.update_obs(self.obs)
                
                # Replan with reduced obstacles
                self.curr_pos = self.s_start
                self.visited = set()
                self.count += 1
                self.replan()
                
                # Search for new path
                self.path = self.extract_path(self.curr_pos, self.s_goal)
                
                # Plot current state
                plt.cla()
                self.Plot.plot_grid(f"Removed {removal_names[group_idx]}")
                self.Plot.plot_visited(self.visited, 'lightblue')
                self.Plot.plot_path(self.path)
                
                # Highlight removed obstacles
                if removed_obstacles:
                    removed_x = [obs[0] for obs in removed_obstacles]
                    removed_y = [obs[1] for obs in removed_obstacles]
                    plt.plot(removed_x, removed_y, "go", label="Removed Obstacles")
                    plt.legend()
                
                self.Plot.capture_frame()
        
        # Final summary frame
        final_path_length = len(self.path) if self.path else 0
        improvement = 0
        if initial_path_length > 0 and final_path_length > 0:
            improvement = ((initial_path_length - final_path_length) / initial_path_length) * 100
        
        plt.cla()
        self.Plot.plot_grid(f"017 Focused D* Final Path (Improved by {improvement:.1f}%)")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        self.Plot.capture_frame()
        
        # Save the complete GIF
        self.Plot.save_animation_as_gif("017_Focused_D_star", fps=2)
        
        print(f"Focused D* complete demonstration finished!")
        print(f"Path improvement: {improvement:.1f}%")
        print(f"Initial path length: {initial_path_length} nodes")
        print(f"Final path length: {final_path_length} nodes")


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    print("Starting Focused D* Demonstration with GIF generation")
    print("This will show: Initial path -> Adding obstacles dynamically -> Focused replanning")
    
    fdstar = FocusedDStar(s_start, s_goal, "euclidean")
    fdstar.run_demonstration()


if __name__ == '__main__':
    main()
