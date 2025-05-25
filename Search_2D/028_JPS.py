"""
JPS
@author: clark bai
"""

import io
import math
import os
import heapq
import time
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
        # Create figure with fixed size
        plt.figure(figsize=self.fig_size, dpi=100, clear=True)
        
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs",  label='Start')
        plt.plot(self.xG[0], self.xG[1], "gs",  label='Goal')
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name, fontsize=14)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)

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
            plt.plot(path_x, path_y, linewidth='3', color='r', label='Final Path')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.1)
        self.capture_frame()

    def capture_frame(self):
        """Capture current frame for GIF animation"""
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

    def save_animation_as_gif(self, name, fps=10):
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


class JPS:
    """Jump Point Search algorithm
    
    JPS is an optimization to A* for uniform-cost grid maps that reduces symmetries
    by selectively expanding nodes. It identifies "jump points" to minimize the
    number of nodes put into the open set while maintaining optimality.
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = set()  # CLOSED set / VISITED
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        
        # Record jump points
        self.jump_points = []

    def searching(self, plot):
        """
        Jump Point Search - Simplified
        :param plot: Plotting object for visualization
        :return: path, visited order
        """
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        visited = [self.s_start]
        
        # Add start node to open list
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        
        # Debug info
        print(f"Starting search from {self.s_start} to {self.s_goal}")
        nodes_processed = 0
        
        # Main search loop
        while self.OPEN:
            # Get node with lowest f-value
            _, current = heapq.heappop(self.OPEN)
            nodes_processed += 1
            
            # Skip if already visited
            if current in self.CLOSED:
                continue
                
            # Add to visited nodes
            self.CLOSED.add(current)
            visited.append(current)
            
            # Check if goal reached
            if current == self.s_goal:
                print(f"Goal reached after processing {nodes_processed} nodes!")
                break
            
            # Debug - print current position periodically
            if nodes_processed % 50 == 0:
                print(f"Processing node {nodes_processed}: {current}")
            
            # Find all successors
            neighbors = self.get_neighbors(current)
            
            # Dynamic plotting - plot current node and visited nodes
            plt.plot(current[0], current[1], 'ro') # Current node
            
            # Plot visited nodes
            for node_v in visited[-50:]: # Only plot recent visited nodes for performance
                if node_v != self.s_start and node_v != self.s_goal and node_v != current:
                    plt.plot(node_v[0], node_v[1], 'gray', marker='.')
            
            # Plot currently considered neighbors
            for neighbor_n in neighbors:
                if not self.is_obstacle(neighbor_n):
                    plt.plot(neighbor_n[0], neighbor_n[1], 'yo', alpha=0.5) # Neighbor nodes
            
            # Update display and capture frame
            plt.pause(0.01)
            plot.capture_frame()
            
            for neighbor in neighbors:
                # Check if neighbor is valid
                if self.is_obstacle(neighbor):
                    continue
                
                # Try to jump from current to neighbor
                jp = self.find_jump_point(current, neighbor)
                
                if jp:
                    # Record found jump point
                    self.jump_points.append((current, jp))
                    
                    # Plot jump point and connection line
                    plt.plot(jp[0], jp[1], 'bo') # Jump point
                    plt.plot([current[0], jp[0]], [current[1], jp[1]], 'g-', linewidth=1.5, alpha=0.7) # Connection to jump point
                    plt.pause(0.05)  # Pause longer when a jump point is found
                    plot.capture_frame()
                    
                    # Calculate cost to this jump point
                    new_cost = self.g[current] + self.cost(current, jp)
                    
                    # Update if better path found
                    if jp not in self.g or new_cost < self.g[jp]:
                        self.g[jp] = new_cost
                        self.PARENT[jp] = current
                        heapq.heappush(self.OPEN, (self.f_value(jp), jp))
        
        # Extract path
        path = self.extract_path(self.PARENT)
        
        # Report results
        if path:
            print(f"Path found with {len(path)} nodes")
            print(f"Found {len(self.jump_points)} jump points")
            
            # Final plot
            plt.cla() # Clear current axes
            plot.plot_grid("Jump Point Search (JPS) - Final Result")
            
            # Plot visited nodes
            for node_v in visited:
                if node_v != self.s_start and node_v != self.s_goal:
                    plt.plot(node_v[0], node_v[1], 'gray', marker='.')
            
            # Plot jump points
            jump_points_only = list(set([jp_node for _, jp_node in self.jump_points]))
            for jp_node_item in jump_points_only:
                if jp_node_item != self.s_start and jp_node_item != self.s_goal:
                    plt.plot(jp_node_item[0], jp_node_item[1], 'bo')
            
            # Plot jump point connection lines
            for start_node, end_node in self.jump_points:
                plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'g-', linewidth=1.5, alpha=0.7)
            
            # Plot the final path
            plot.plot_path(path)
            
            # Add legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='b', label='Start Point', linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='g', label='Goal Point', linestyle='None'),
                plt.Line2D([0], [0], marker='.', color='gray', label='Visited Node', linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='b', label='Jump Point', linestyle='None'),
                plt.Line2D([0], [0], color='g', label='Jump Connection', linewidth=1.5, alpha=0.7),
                plt.Line2D([0], [0], color='r', label='Final Path', linewidth=2)
            ]
            plt.legend(handles=handles, loc='upper right')
  
            plt.pause(1) # Pause for 1 second to view final result
            plot.capture_frame()
        else:
            print(f"No path found after processing {nodes_processed} nodes")
            
        return path, visited

    def extract_temp_path(self, current_node):
        """
        Extract temporary path from start to current node
        :param current_node: Current node
        :return: Temporary path
        """
        path = [current_node]
        s_node = current_node
        
        while s_node != self.s_start:
            if s_node not in self.PARENT:
                return [] # Path does not exist
            s_node = self.PARENT[s_node]
            path.append(s_node)
        
        return list(reversed(path))

    def get_neighbors(self, s_node):
        """
        Find neighbors of state s_node that are not in obstacles
        :param s_node: State
        :return: Neighbors
        """
        nei_list = []
        for u_motion in self.u_set:
            s_next = (s_node[0] + u_motion[0], s_node[1] + u_motion[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list
    
    def find_jump_point(self, current_node, neighbor_node):
        """
        Detect jump point - Iterative implementation
        :param current_node: Current node
        :param neighbor_node: Neighbor node
        :return: Jump point or None
        """
        # Direction from current_node to neighbor_node
        dx = neighbor_node[0] - current_node[0]
        dy = neighbor_node[1] - current_node[1]
        
        # Normalize the direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        
        # Check if the initial neighbor_node is valid
        if self.is_obstacle(neighbor_node):
            return None
        
        # If the neighbor_node is the goal, return it immediately
        if neighbor_node == self.s_goal:
            return neighbor_node
            
        # Start iterative check
        node_to_check = neighbor_node
        steps = 0
        max_steps = 1000  # Increase the maximum number of steps to prevent early return
        
        while steps < max_steps:
            steps += 1
            x_coord, y_coord = node_to_check
            
            # Diagonal movement
            if dx != 0 and dy != 0:
                # Check forced neighbors
                if ((self.is_obstacle((x_coord - dx, y_coord)) and not self.is_obstacle((x_coord - dx, y_coord + dy))) or
                    (self.is_obstacle((x_coord, y_coord - dy)) and not self.is_obstacle((x_coord + dx, y_coord - dy)))):
                    return node_to_check
                    
                # Recursively check horizontal and vertical directions
                h_jp = self.find_jump_point(node_to_check, (x_coord + dx, y_coord))
                if h_jp:
                    return node_to_check # If a jump point is found horizontally, current node_to_check is a jump point
                    
                v_jp = self.find_jump_point(node_to_check, (x_coord, y_coord + dy))
                if v_jp:
                    return node_to_check # If a jump point is found vertically, current node_to_check is a jump point
                
            # Horizontal movement
            elif dx != 0: # Straight horizontal movement
                # Check forced neighbors
                if ((self.is_obstacle((x_coord, y_coord + 1)) and not self.is_obstacle((x_coord + dx, y_coord + 1))) or
                    (self.is_obstacle((x_coord, y_coord - 1)) and not self.is_obstacle((x_coord + dx, y_coord - 1)))):
                    return node_to_check
                    
            # Vertical movement
            elif dy != 0: # Straight vertical movement
                # Check forced neighbors
                if ((self.is_obstacle((x_coord + 1, y_coord)) and not self.is_obstacle((x_coord + 1, y_coord + dy))) or
                    (self.is_obstacle((x_coord - 1, y_coord)) and not self.is_obstacle((x_coord - 1, y_coord + dy)))):
                    return node_to_check
            
            # Calculate the next position in the current direction
            next_x, next_y = x_coord + dx, y_coord + dy
            next_pos = (next_x, next_y)
            
            # If the next position is invalid (obstacle or out of bounds), stop
            if self.is_obstacle(next_pos):
                return None # No jump point found in this direction
                
            # If the next position is the goal, it's a jump point
            if next_pos == self.s_goal:
                return self.s_goal
                
            # Move to the next position
            node_to_check = next_pos
            
        # If max_steps reached or no jump point condition met along the straight line
        return node_to_check
    
    def is_forced_neighbor(self, node_to_check, direction_vec):
        """
        Check if the node_to_check has a forced neighbor
        :param node_to_check: Current node
        :param direction_vec: Movement direction (dx, dy)
        :return: Boolean, True if there is a forced neighbor
        """
        x_coord, y_coord = node_to_check
        dx_dir, dy_dir = direction_vec
        
        # Horizontal movement
        if dy_dir == 0: # Moving horizontally
            # Check for obstacles above/below that force a turn
            if self.is_obstacle((x_coord, y_coord + 1)) and not self.is_obstacle((x_coord + dx_dir, y_coord + 1)):
                return True
            if self.is_obstacle((x_coord, y_coord - 1)) and not self.is_obstacle((x_coord + dx_dir, y_coord - 1)):
                return True
                
        # Vertical movement
        elif dx_dir == 0: # Moving vertically
            # Check for obstacles left/right that force a turn
            if self.is_obstacle((x_coord + 1, y_coord)) and not self.is_obstacle((x_coord + 1, y_coord + dy_dir)):
                return True
            if self.is_obstacle((x_coord - 1, y_coord)) and not self.is_obstacle((x_coord - 1, y_coord + dy_dir)):
                return True
                
        # No forced neighbor found for straight moves based on these rules
        return False
    
    def is_obstacle(self, node_to_check):
        """
        Check if a node_to_check is an obstacle or out of bounds
        :param node_to_check: Node to check
        :return: True if obstacle or out of bounds, False otherwise
        """
        x_coord, y_coord = node_to_check
        
        # Check if out of bounds
        if not (0 <= x_coord < self.Env.x_range and 0 <= y_coord < self.Env.y_range):
            return True
            
        # Check if in obstacle set
        if node_to_check in self.obs:
            return True
            
        return False

    def cost(self, s_start_node, s_goal_node):
        """
        Calculate cost between two nodes (Euclidean distance)
        :param s_start_node: starting node
        :param s_goal_node: end node
        :return: Cost for this motion (distance)
        """
        # This check might be redundant if is_obstacle is checked before calling cost
        if self.is_obstacle(s_start_node) or self.is_obstacle(s_goal_node):
            return math.inf # Infinite cost if one of the nodes is an obstacle

        return math.hypot(s_goal_node[0] - s_start_node[0], s_goal_node[1] - s_start_node[1])

    def f_value(self, s_node):
        """
        Calculate f value (f = g + h)
        :param s_node: current state/node
        :return: f value
        """
        return self.g[s_node] + self.heuristic(s_node)

    def extract_path(self, parent_map):
        """
        Extract the path based on the parent_map set
        :param parent_map: Dictionary storing parent of each node
        :return: The planning path from start to goal
        """
        # Check if a path to the goal was found
        if self.s_goal not in parent_map:
            return [] # No path found
            
        # Reconstruct path from goal to start
        path = [self.s_goal]
        current_s = self.s_goal

        while current_s != self.s_start:
            current_s = parent_map[current_s]
            path.append(current_s)

            if current_s == self.s_start: # Should be caught by while condition, but good for clarity
                break
        
        path.reverse() # Reverse the path to be from start to goal
        return path

    def heuristic(self, s_node):
        """
        Calculate heuristic (estimated cost from s_node to goal)
        :param s_node: current node
        :return: heuristic value
        """
        goal_node = self.s_goal
        
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - s_node[0]) + abs(goal_node[1] - s_node[1])
        else:  # Default to euclidean
            return math.hypot(goal_node[0] - s_node[0], goal_node[1] - s_node[1])


def run_jps(s_start_coord, s_goal_coord, run_title="", save_gif=False):
    """
    Run Jump Point Search (JPS)
    :param s_start_coord: Start point coordinates
    :param s_goal_coord: Goal point coordinates
    :param run_title: Title for the JPS run
    :param save_gif: Whether to save GIF animation
    """
    if run_title:
        print(f"\n===== {run_title} =====")
    
    # Create JPS and plotting objects
    jps_solver = JPS(s_start_coord, s_goal_coord, "euclidean")
    plot = Plotting(s_start_coord, s_goal_coord)
    
    # Display environment info
    print(f"Grid size: {jps_solver.Env.x_range} × {jps_solver.Env.y_range}")
    print(f"Start: {s_start_coord}, Goal: {s_goal_coord}")
    print(f"Number of obstacles: {len(jps_solver.Env.obs)}")
    
    # Plot initial grid
    plot.plot_grid("Jump Point Search (JPS) - Live Demo")
    
    # Run JPS
    print("\nRunning Jump Point Search (JPS)...")
    start_time = time.time()
    jps_path, jps_visited_nodes = jps_solver.searching(plot)
    end_time = time.time()
    jps_run_time = end_time - start_time
    
    print(f"JPS Runtime: {jps_run_time:.4f} seconds")
    print(f"JPS Nodes explored: {len(jps_visited_nodes)}")
    
    if jps_path:
        print(f"JPS found a path with {len(jps_path)} nodes")
    else:
        print("JPS could not find a path.")
    
    # Show the plot
    plt.show()
    
    # Save GIF if requested
    if save_gif:
        plot.save_animation_as_gif("028_JPS", fps=10)


def main():
    """
    Testing JPS implementation
    """
    print("Jump Point Search (JPS) Implementation")
    print("--------------------------------------")

    s_start_main = (5, 5)
    s_goal_main = (45, 25)
    run_jps(s_start_main, s_goal_main, "Test Case JPS", save_gif=True)


if __name__ == '__main__':
    main()
