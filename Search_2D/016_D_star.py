"""
D_star 2D 
@author: clark bai
"""

import os
import math
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


class DStar:
    def __init__(self, s_start, s_goal):
        self.s_start, self.s_goal = s_start, s_goal

        self.Env = Env()
        self.Plot = Plotting(self.s_start, self.s_goal)

        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.OPEN = set()
        self.t = dict()
        self.PARENT = dict()
        self.h = dict()
        self.k = dict()
        self.path = []
        self.visited = set()
        self.count = 0

    def init(self):
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.t[(i, j)] = 'NEW'
                self.k[(i, j)] = 0.0
                self.h[(i, j)] = float("inf")
                self.PARENT[(i, j)] = None

        self.h[self.s_goal] = 0.0

    def run_demonstration(self):
        """
        Run complete demonstration with adding obstacles, then removing them
        """
        print("Starting D* complete demonstration...")
        
        # Create figure
        plt.figure(figsize=self.Plot.fig_size, dpi=100)
        
        # Part 1: Initial planning phase
        print("PART 1: Initial D* planning...")
        self.init()
        self.insert(self.s_goal, 0)

        # Process until start node is closed
        while True:
            self.process_state()
            if self.t[self.s_start] == 'CLOSED':
                break

        self.path = self.extract_path(self.s_start, self.s_goal)
        
        # Plot initial state
        plt.cla()
        self.Plot.plot_grid("016 D* - Initial Path")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(0.5)
        self.Plot.capture_frame()
        
        # Part 2: Add obstacles sequence
        print("PART 2: Adding obstacles and repairing path...")
        obstacle_sequence = [
            (16, 16), (16, 17), (32, 15), (36, 17), (30, 14), (30, 13)
        ]
        
        for step, (x, y) in enumerate(obstacle_sequence):
            print(f"  Adding obstacle {step + 1}: ({x}, {y})")
            
            if x >= 0 and x < self.x and y >= 0 and y < self.y and (x, y) not in self.obs:
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
                
                # Update path using D* repair mechanism
                s = self.s_start
                self.visited = set()
                self.count += 1

                # Check if current path is affected and repair if necessary
                path_affected = False
                current_s = self.s_start
                while current_s != self.s_goal:
                    if self.PARENT[current_s] is None:
                        path_affected = True
                        break
                    if self.is_collision(current_s, self.PARENT[current_s]):
                        path_affected = True
                        self.modify(current_s)
                        break
                    current_s = self.PARENT[current_s]

                if path_affected:
                    print(f"    Path affected, repairing using D* mechanism...")
                
                # Extract updated path
                self.path = self.extract_path(self.s_start, self.s_goal)
                
                # Plot current state
                plt.cla()
                self.Plot.plot_grid(f"Adding Obstacle {step + 1}")
                self.Plot.plot_visited(self.visited, 'lightblue')
                self.Plot.plot_path(self.path)
                plt.plot(x, y, "ro", label="New Obstacle")
                plt.legend()
                
                plt.pause(0.3)
                self.Plot.capture_frame()
        
        # Part 3: Add blocking obstacles for removal demo
        print("PART 3: Adding blocking obstacles...")
        blocking_obstacles = [
            (25, 15), (25, 16), (25, 17),  # First block
            (15, 18), (16, 18), (17, 18),  # Second block
            (20, 10), (20, 11), (20, 12)   # Third block
        ]
        
        for obs in blocking_obstacles:
            self.obs.add(obs)
        self.Plot.update_obs(self.obs)
        
        # Reinitialize and search with all obstacles
        self.init()
        self.insert(self.s_goal, 0)
        
        while True:
            k_min = self.process_state()
            if k_min == -1 or self.t[self.s_start] == 'CLOSED':
                break
        
        self.visited = set()
        self.path = self.extract_path(self.s_start, self.s_goal)
        initial_path_length = len(self.path) if self.path else 0
        
        plt.cla()
        self.Plot.plot_grid("With All Blocking Obstacles")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(0.5)
        self.Plot.capture_frame()
        
        # Part 4: Remove obstacles and show path improvement
        print("PART 4: Removing obstacles and improving path...")
        removal_sequence = [
            [(25, 15), (25, 16), (25, 17)],  # Remove first block
            [(15, 18), (16, 18), (17, 18)],  # Remove second block
            [(20, 10), (20, 11), (20, 12)]   # Remove third block
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
                
                # Reinitialize and search with reduced obstacles
                self.init()
                self.insert(self.s_goal, 0)
                
                while True:
                    k_min = self.process_state()
                    if k_min == -1 or self.t[self.s_start] == 'CLOSED':
                        break
                
                self.visited = set()
                self.count += 1
                
                # Search for new path
                self.path = self.extract_path(self.s_start, self.s_goal)
                
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
                
                plt.pause(0.5)
                self.Plot.capture_frame()
        
        # Final summary frame
        final_path_length = len(self.path) if self.path else 0
        improvement = 0
        if initial_path_length > 0 and final_path_length > 0:
            improvement = ((initial_path_length - final_path_length) / initial_path_length) * 100
        
        plt.cla()
        self.Plot.plot_grid(f"016 D* Final Path (Improved by {improvement:.1f}%)")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(1.0)
        self.Plot.capture_frame()
        
        # Save the complete GIF
        self.Plot.save_animation_as_gif("016_D_star", fps=2)
        
        # Show final result
        plt.show()
        
        print(f"D* complete demonstration finished!")
        print(f"Path improvement: {improvement:.1f}%")
        print(f"Initial path length: {initial_path_length} nodes")
        print(f"Final path length: {final_path_length} nodes")

    def run_interactive(self):
        """
        Run interactive demonstration where user can click to add/remove obstacles
        """
        print("Starting D* interactive demonstration...")
        print("Click on the plot to add/remove obstacles and see D* path repair in action")
        
        # Initial planning phase
        self.init()
        self.insert(self.s_goal, 0)

        while True:
            self.process_state()
            if self.t[self.s_start] == 'CLOSED':
                break

        self.path = self.extract_path(self.s_start, self.s_goal)
        
        # Create figure and plot
        self.fig = plt.figure(figsize=self.Plot.fig_size, dpi=100)
        self.Plot.plot_grid("016 Dynamic A* (D*)")
        self.plot_path(self.path)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            if (x, y) in self.obs:
                print("Remove obstacle at: s =", x, ",", "y =", y)
                self.obs.remove((x, y))
                self.Plot.update_obs(self.obs)

                s = self.s_start
                self.visited = set()
                self.count += 1

                while s != self.s_goal:
                    if self.is_collision(s, self.PARENT[s]):
                        self.modify(s)
                        continue
                    s = self.PARENT[s]

                self.path = self.extract_path(self.s_start, self.s_goal)

                plt.cla()
                self.Plot.plot_grid("016 Dynamic A* (D*)")
                self.plot_visited(self.visited)
                self.plot_path(self.path)
            elif (x, y) not in self.obs:
                print("Add obstacle at: s =", x, ",", "y =", y)
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)

                s = self.s_start
                self.visited = set()
                self.count += 1

                while s != self.s_goal:
                    if self.is_collision(s, self.PARENT[s]):
                        self.modify(s)
                        continue
                    s = self.PARENT[s]

                self.path = self.extract_path(self.s_start, self.s_goal)

                plt.cla()
                self.Plot.plot_grid("016 Dynamic A* (D*)")
                self.plot_visited(self.visited)
                self.plot_path(self.path)

            self.fig.canvas.draw_idle()

    def extract_path(self, s_start, s_end):
        path = [s_start]
        s = s_start
        while True:
            if s not in self.PARENT or self.PARENT[s] is None:
                return []
            s = self.PARENT[s]
            path.append(s)
            if s == s_end:
                return path

    def process_state(self):
        s = self.min_state()  # get node in OPEN set with min k value
        if s is not None:
            self.visited.add(s)

        if s is None:
            return -1  # OPEN set is empty

        k_old = self.get_k_min()  # record the min k value of this iteration (min path cost)
        self.delete(s)  # move state s from OPEN set to CLOSED set

        # k_min < h[s] --> s: RAISE state (increased cost)
        if k_old < self.h[s]:
            for s_n in self.get_neighbor(s):
                if self.h[s_n] <= k_old and \
                        self.h[s] > self.h[s_n] + self.cost(s_n, s):

                    # update h_value and choose parent
                    self.PARENT[s] = s_n
                    self.h[s] = self.h[s_n] + self.cost(s_n, s)

        # s: k_min >= h[s] -- > s: LOWER state (cost reductions)
        if k_old == self.h[s]:
            for s_n in self.get_neighbor(s):
                if self.t[s_n] == 'NEW' or \
                        (self.PARENT[s_n] == s and self.h[s_n] != self.h[s] + self.cost(s, s_n)) or \
                        (self.PARENT[s_n] != s and self.h[s_n] > self.h[s] + self.cost(s, s_n)):

                    # Condition:
                    # 1) t[s_n] == 'NEW': not visited
                    # 2) s_n's parent: cost reduction
                    # 3) s_n find a better parent
                    self.PARENT[s_n] = s
                    self.insert(s_n, self.h[s] + self.cost(s, s_n))
        else:
            for s_n in self.get_neighbor(s):
                if self.t[s_n] == 'NEW' or \
                        (self.PARENT[s_n] == s and self.h[s_n] != self.h[s] + self.cost(s, s_n)):

                    # Condition:
                    # 1) t[s_n] == 'NEW': not visited
                    # 2) s_n's parent: cost reduction
                    self.PARENT[s_n] = s
                    self.insert(s_n, self.h[s] + self.cost(s, s_n))
                else:
                    if self.PARENT[s_n] != s and \
                            self.h[s_n] > self.h[s] + self.cost(s, s_n):

                        # Condition: LOWER happened in OPEN set (s), s should be explored again
                        self.insert(s, self.h[s])
                    else:
                        if self.PARENT[s_n] != s and \
                                self.h[s] > self.h[s_n] + self.cost(s_n, s) and \
                                self.t[s_n] == 'CLOSED' and \
                                self.h[s_n] > k_old:

                            # Condition: LOWER happened in CLOSED set (s_n), s_n should be explored again
                            self.insert(s_n, self.h[s_n])

        return self.get_k_min()

    def min_state(self):
        """
        choose the node with the minimum k value in OPEN set.
        :return: state
        """

        if not self.OPEN:
            return None

        return min(self.OPEN, key=lambda x: self.k[x])

    def get_k_min(self):
        """
        calc the min k value for nodes in OPEN set.
        :return: k value
        """

        if not self.OPEN:
            return -1

        return min([self.k[x] for x in self.OPEN])

    def insert(self, s, h_new):
        """
        insert node into OPEN set.
        :param s: node
        :param h_new: new or better cost to come value
        """

        if self.t[s] == 'NEW':
            self.k[s] = h_new
        elif self.t[s] == 'OPEN':
            self.k[s] = min(self.k[s], h_new)
        elif self.t[s] == 'CLOSED':
            self.k[s] = min(self.h[s], h_new)

        self.h[s] = h_new
        self.t[s] = 'OPEN'
        self.OPEN.add(s)

    def delete(self, s):
        """
        delete: move state s from OPEN set to CLOSED set.
        :param s: state should be deleted
        """

        if self.t[s] == 'OPEN':
            self.t[s] = 'CLOSED'

        self.OPEN.remove(s)

    def modify(self, s):
        """
        start processing from state s.
        :param s: is a node whose status is RAISE or LOWER.
        """

        self.modify_cost(s)

        while True:
            k_min = self.process_state()

            if k_min >= self.h[s]:
                break

    def modify_cost(self, s):
        # if node in CLOSED set, put it into OPEN set.
        # Since cost may be changed between s - s.parent, calc cost(s, s.p) again

        if self.t[s] == 'CLOSED':
            self.insert(s, self.h[self.PARENT[s]] + self.cost(s, self.PARENT[s]))

    def get_neighbor(self, s):
        nei_list = set()

        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if 0 <= s_next[0] < self.x and 0 <= s_next[1] < self.y and s_next not in self.obs:
                nei_list.add(s_next)

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
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
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

    def plot_path(self, path):
        if not path:
            return
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2, color='red', label='Path')
        plt.plot(self.s_start[0], self.s_start[1], "bs", label='Start')
        plt.plot(self.s_goal[0], self.s_goal[1], "gs", label='Goal')

    def plot_visited(self, visited):
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

    print("Starting D* Demonstration with GIF generation")
    print("This will show: Initial path -> Adding obstacles dynamically -> Path repair using D*")
    
    dstar = DStar(s_start, s_goal)
    dstar.run_demonstration()


if __name__ == '__main__':
    main()
