"""
Repairing_Astar 2D - Modified for GIF generation
@author: clark bai, modified for automatic demonstration
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


class RepairingAStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()
        self.Plot = Plotting(self.s_start, self.s_goal)

        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        # Cost to come and parent dictionaries
        self.g = {}
        self.parent = {}

        # Current optimal path
        self.path = []
        self.affected_nodes = set()

        # Initialize costs
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.g[(i, j)] = float("inf")

        self.g[self.s_start] = 0
        self.parent[self.s_start] = self.s_start
        self.visited = set()
        self.count = 0

    def run_demonstration(self):
        """
        Run automatic demonstration with predefined obstacle sequence
        """
        # Obstacle sequence as provided
        obstacle_sequence = [
            (16, 16), 
            (16, 17), 
            (32, 15), (36, 17), (30, 14),  (30, 13), 
           
        ]
        
        print("Starting Repairing A* demonstration...")
        
        # Create figure
        plt.figure(figsize=self.Plot.fig_size, dpi=100)
        
        # Initial search and plot
        print("Performing initial A* search...")
        self.path = self.searching()
        
        # Plot initial state
        plt.cla()
        self.Plot.plot_grid("Repairing A* - Initial Path")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(0.5)
        self.Plot.capture_frame()
        
        # Process each obstacle in sequence
        for step, (x, y) in enumerate(obstacle_sequence):
            print(f"Step {step + 1}: Adding obstacle at ({x}, {y})")
            
            # Check if coordinates are valid
            if x < 0 or x >= self.x or y < 0 or y >= self.y:
                print(f"Invalid coordinates ({x}, {y}), skipping...")
                continue
            
            # Reset visited for new search
            self.visited = set()
            self.count += 1
            
            # Add obstacle (only if not already present)
            if (x, y) not in self.obs:
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
                
                # Identify affected nodes
                self.identify_affected_nodes((x, y))
                
                # Repair path if necessary
                if self.is_path_affected():
                    print(f"  Path affected, repairing...")
                    self.repair_path()
                else:
                    print(f"  Path not affected")
                
                # Plot current state
                plt.cla()
                self.Plot.plot_grid(f"Repairing A* - Step {step + 1}")
                self.Plot.plot_visited(self.visited, 'lightblue')
                self.Plot.plot_path(self.path)
                
                # Highlight the newly added obstacle
                plt.plot(x, y, "ro", label="New Obstacle")
                plt.legend()
                
                plt.pause(0.5)
                self.Plot.capture_frame()
            else:
                print(f"  Obstacle already exists at ({x}, {y})")
        
        # Save the GIF
        self.Plot.save_animation_as_gif("012_Repairing_Astar_Modified", fps=2)
        
        # Show final result
        plt.show()
        
        print("Demonstration completed!")

    def run_obstacle_removal_demonstration(self):
        """
        Run demonstration showing the effect of removing obstacles
        """
        print("Starting Repairing A* demonstration with obstacle removal...")
        
        # First add some obstacles to create a challenging scenario
        additional_obstacles = [
            (25, 15), (25, 16), (25, 17),  # Block to be removed
            (15, 18), (16, 18), (17, 18),  # Another block
            (35, 10), (35, 11), (35, 12)   # Third block
        ]
        
        # Add additional obstacles to create a more complex scenario
        for obs in additional_obstacles:
            self.obs.add(obs)
        self.Plot.update_obs(self.obs)
        
        # Create figure
        plt.figure(figsize=self.Plot.fig_size, dpi=100)
        
        # Initial search with all obstacles
        print("Performing initial A* search with all obstacles...")
        self.visited = set()
        self.path = self.searching()
        initial_path_length = len(self.path) if self.path else 0
        
        # Plot initial state
        plt.cla()
        self.Plot.plot_grid("Repairing A* - Initial Path with Obstacles")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(0.5)
        self.Plot.capture_frame()
        
        # Define obstacles to remove (3 groups of obstacles)
        removal_sequence = [
            (25, 15), (25, 16), (25, 17),  # Block to be removed
            (15, 18), (16, 18), (17, 18),  # Another block
            (35, 10), (35, 11), (35, 12)   # Third block
        ]
        
        removal_names = ["First Block", "Second Block", "Third Block"]
        
        # Process each removal group
        for group_idx, obstacle_group in enumerate(removal_sequence):
            print(f"Step {group_idx + 1}: Removing {removal_names[group_idx]} obstacles: {obstacle_group}")
            
            # Remove obstacles from the group
            removed_obstacles = []
            for obs in obstacle_group:
                if obs in self.obs:
                    self.obs.remove(obs)
                    removed_obstacles.append(obs)
                    print(f"  Removed obstacle at {obs}")
            
            if removed_obstacles:
                self.Plot.update_obs(self.obs)
                
                # Reset visited for new search
                self.visited = set()
                self.count += 1
                
                # Store current path for comparison
                old_path = self.path.copy() if self.path else []
                old_path_length = len(old_path)
                
                # Search for potentially better path
                self.path = self.searching()
                new_path_length = len(self.path) if self.path else 0
                
                # Check if we found a better path
                if self.path and old_path:
                    path_cost_old = sum(self.cost(old_path[i], old_path[i+1]) for i in range(len(old_path)-1))
                    path_cost_new = sum(self.cost(self.path[i], self.path[i+1]) for i in range(len(self.path)-1))
                    
                    if path_cost_new < path_cost_old:
                        print(f"  Found better path! Cost improved from {path_cost_old:.2f} to {path_cost_new:.2f}")
                    else:
                        print(f"  Path updated but not necessarily better. Old cost: {path_cost_old:.2f}, New cost: {path_cost_new:.2f}")
                elif self.path and not old_path:
                    print(f"  Found path where none existed before!")
                
                # Plot current state
                plt.cla()
                self.Plot.plot_grid(f"Repairing A* - Removed {removal_names[group_idx]}")
                self.Plot.plot_visited(self.visited, 'lightblue')
                self.Plot.plot_path(self.path)
                
                # Highlight the removed obstacles
                if removed_obstacles:
                    removed_x = [obs[0] for obs in removed_obstacles]
                    removed_y = [obs[1] for obs in removed_obstacles]
                    plt.plot(removed_x, removed_y, "go", label="Removed Obstacles")
                    plt.legend()
                
                plt.pause(0.5)
                self.Plot.capture_frame()
            else:
                print(f"  No obstacles to remove in {removal_names[group_idx]}")
        
        # Show improvement summary
        final_path_length = len(self.path) if self.path else 0
        print(f"\nPath improvement summary:")
        print(f"Initial path length: {initial_path_length} nodes")
        print(f"Final path length: {final_path_length} nodes")
        if initial_path_length > 0 and final_path_length > 0:
            improvement = ((initial_path_length - final_path_length) / initial_path_length) * 100
            print(f"Improvement: {improvement:.1f}%")
        
        # Save the GIF
        self.Plot.save_animation_as_gif("012_Repairing_Astar_Obstacle_Removal", fps=2)
        
        # Show final result
        plt.show()
        
        print("Obstacle removal demonstration completed!")

    def run_complete_demonstration(self):
        """
        Run complete demonstration with adding obstacles, then removing them
        All in one continuous GIF
        """
        print("Starting complete Repairing A* demonstration...")
        
        # Create figure
        plt.figure(figsize=self.Plot.fig_size, dpi=100)
        
        # Part 1: Initial search
        print("PART 1: Initial A* search...")
        self.path = self.searching()
        
        # Plot initial state
        plt.cla()
        self.Plot.plot_grid("Repairing A* - Initial Path")
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
                self.visited = set()
                self.count += 1
                
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
                
                # Identify affected nodes and repair path
                self.identify_affected_nodes((x, y))
                if self.is_path_affected():
                    self.repair_path()
                
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
        
        # Search with all obstacles
        self.visited = set()
        self.path = self.searching()
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
            [(15, 18), (16, 18), (17, 18)],  # Second block 
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
                self.visited = set()
                self.count += 1
                
                # Search for new path
                self.path = self.searching()
                
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
        self.Plot.plot_grid(f"Final Path (Improved by {improvement:.1f}%)")
        self.Plot.plot_visited(self.visited, 'lightblue')
        self.Plot.plot_path(self.path)
        plt.pause(1.0)
        self.Plot.capture_frame()
        
        # Save the complete GIF
        self.Plot.save_animation_as_gif("012_Repairing_Astar_Complete", fps=2)
        
        # Show final result
        plt.show()
        
        print(f"Complete demonstration finished!")
        print(f"Path improvement: {improvement:.1f}%")
        print(f"Initial path length: {initial_path_length} nodes")
        print(f"Final path length: {final_path_length} nodes")

    def searching(self):
        """Standard A* search"""
        open_set = {self.s_start}
        closed_set = set()
        
        # Initialize g and parent
        self.g = {}
        self.parent = {}
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.g[(i, j)] = float("inf")
                
        self.g[self.s_start] = 0
        self.parent[self.s_start] = self.s_start
        
        while open_set:
            # Select node with minimum f_value
            s = min(open_set, key=lambda x: self.g[x] + self.h(x))
            self.visited.add(s)
            
            # Check if goal is reached
            if s == self.s_goal:
                return self.extract_path()
                
            # Remove s from open_set and add to closed_set
            open_set.remove(s)
            closed_set.add(s)
            
            # Expand neighbors
            for s_n in self.get_neighbor(s):
                if s_n in closed_set:
                    continue
                    
                new_cost = self.g[s] + self.cost(s, s_n)
                
                if s_n not in open_set or new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.parent[s_n] = s
                    if s_n not in open_set:
                        open_set.add(s_n)
                        
        return []

    def identify_affected_nodes(self, changed_node):
        """Identify nodes affected by environment changes"""
        self.affected_nodes = set()
        
        # Find nodes that have the changed node as their neighbor
        for i in range(max(0, changed_node[0] - 2), min(self.x, changed_node[0] + 3)):
            for j in range(max(0, changed_node[1] - 2), min(self.y, changed_node[1] + 3)):
                node = (i, j)
                if node in self.parent and node not in self.obs:
                    if changed_node in self.get_neighbor(node) or node == changed_node:
                        self.affected_nodes.add(node)
        
        # Check if any part of the path is affected
        for i in range(len(self.path) - 1):
            s1, s2 = self.path[i], self.path[i + 1]
            if self.is_collision(s1, s2):
                self.affected_nodes.add(s1)
                self.affected_nodes.add(s2)

    def is_path_affected(self):
        """Check if the current path is affected by environment changes"""
        if not self.path:
            return True
            
        # Check if any node in the path is in affected_nodes
        for node in self.path:
            if node in self.affected_nodes:
                return True
                
        # Check if any segment of the path is now in collision
        for i in range(len(self.path) - 1):
            if self.is_collision(self.path[i], self.path[i + 1]):
                return True
                
        return False

    def repair_path(self):
        """Repair the existing path by performing a partial search"""
        # Find the first affected node in the path
        affected_index = -1
        for i, node in enumerate(self.path):
            if node in self.affected_nodes:
                affected_index = i
                break
                
        if affected_index == -1:
            # If no specific node is affected, check path segments
            for i in range(len(self.path) - 1):
                if self.is_collision(self.path[i], self.path[i + 1]):
                    affected_index = i
                    break
        
        if affected_index != -1:
            # Keep the valid part of the path and recompute from the affected node
            valid_path = self.path[:affected_index]
            
            # If valid_path is empty, perform a complete A* search
            if not valid_path:
                self.path = self.searching()
                return
            
            # Otherwise, start search from the last valid node
            new_start = valid_path[-1]
            
            # Initialize search from new_start to goal
            open_set = {new_start}
            closed_set = set()
            
            # Update g values for the search
            temp_g = {}
            temp_parent = {}
            for i in range(self.Env.x_range):
                for j in range(self.Env.y_range):
                    temp_g[(i, j)] = float("inf")
                    
            temp_g[new_start] = self.g[new_start]
            temp_parent[new_start] = self.parent[new_start]
            
            # A* search from new_start to goal
            while open_set and self.s_goal not in closed_set:
                s = min(open_set, key=lambda x: temp_g[x] + self.h(x))
                self.visited.add(s)
                
                # Check if goal is reached
                if s == self.s_goal:
                    # Reconstruct the partial path
                    partial_path = [self.s_goal]
                    while partial_path[-1] != new_start:
                        partial_path.append(temp_parent[partial_path[-1]])
                    partial_path.reverse()
                    
                    # Update the full path
                    self.path = valid_path + partial_path[1:]
                    
                    # Update the parent and g values
                    for i in range(len(partial_path) - 1):
                        self.parent[partial_path[i + 1]] = partial_path[i]
                        self.g[partial_path[i + 1]] = temp_g[partial_path[i + 1]]
                    
                    return
                
                # Remove s from open_set and add to closed_set
                open_set.remove(s)
                closed_set.add(s)
                
                # Expand neighbors
                for s_n in self.get_neighbor(s):
                    if s_n in closed_set:
                        continue
                        
                    new_cost = temp_g[s] + self.cost(s, s_n)
                    
                    if s_n not in open_set or new_cost < temp_g[s_n]:
                        temp_g[s_n] = new_cost
                        temp_parent[s_n] = s
                        if s_n not in open_set:
                            open_set.add(s_n)
            
            # If no path found, perform complete A* search
            self.path = self.searching()
        else:
            # If no affected node found but path is affected, do a complete search
            self.path = self.searching()

    def get_neighbor(self, s):
        """Find neighbors of state s that not in obstacles"""
        s_list = set()

        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if 0 <= s_next[0] < self.x and 0 <= s_next[1] < self.y and s_next not in self.obs:
                s_list.add(s_next)

        return s_list

    def h(self, s):
        """Calculate heuristic"""
        heuristic_type = self.heuristic_type
        goal = self.s_goal

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def cost(self, s_start, s_goal):
        """Calculate Cost for this motion"""
        if self.is_collision(s_start, s_goal):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """Check if the line segment between s_start and s_end collides with obstacles"""
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

    def extract_path(self):
        """Extract the path based on the parent dictionary"""
        path = [self.s_goal]
        s = self.s_goal

        while True:
            if s not in self.parent:
                return []
                
            s = self.parent[s]
            path.append(s)
            
            if s == self.s_start:
                break

        return list(reversed(path))


def main():
    x_start = (5, 5)
    x_goal = (45, 25)

    print("Starting Repairing A* Complete Demonstration with GIF generation")
    print("This will show: Initial path -> Adding obstacles -> Removing obstacles -> Path optimization")
    
    repairing_astar = RepairingAStar(x_start, x_goal, "euclidean")
    repairing_astar.run_complete_demonstration()


if __name__ == '__main__':
    main()
