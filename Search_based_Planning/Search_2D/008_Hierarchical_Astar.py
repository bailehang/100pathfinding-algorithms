"""
Hierarchical A* 2D
Self-contained implementation with GIF generation capability
@author: clark bai (original algorithm)
Modified to be self-contained with GIF support
"""

import io
import os
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
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
    """Plotting class for visualization"""

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

    def animation_hierarchical_astar(self, fine_path, coarse_path_fine, visited_fine, grid_lines, name, save_gif=False):
        """
        Animate the hierarchical A* process
        :param fine_path: Final fine path
        :param coarse_path_fine: Coarse path in fine coordinates
        :param visited_fine: Visited nodes in the fine grid
        :param grid_lines: Coarse grid lines for visualization
        :param name: Title of the animation
        :param save_gif: Whether to save as GIF
        """
        self.plot_grid(name)
        
        # Plot coarse grid
        self.plot_grid_lines(grid_lines)
        
        # Plot visited nodes
        self.plot_visited(visited_fine)
        
        # Plot the coarse path
        if coarse_path_fine:
            self.plot_path(coarse_path_fine, cl='blue', flag=True)
        
        # Plot the fine path
        if fine_path:
            self.plot_path(fine_path)
        
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

    def plot_grid_lines(self, grid_lines):
        """Plot grid lines for hierarchical grid visualization"""
        for line in grid_lines:
            start, end = line
            plt.plot([start[0], end[0]], [start[1], end[1]], 'lightgray', linestyle='--', alpha=0.7)
        
        # Capture frame after drawing grid lines
        self.capture_frame()

    def plot_visited(self, visited, cl='gray'):
        """Plot visited nodes during search"""
        if not visited:
            return
            
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
        if not path:
            return
            
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


class HierarchicalAStar:
    """
    Hierarchical A* algorithm implementation
    Uses a two-level approach:
    1. High-level planning with a sparse grid (cells without obstacle clusters)
    2. Low-level refinement connecting path segments
    """
    def __init__(self, s_start, s_goal, heuristic_type, coarse_size=6):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.coarse_size = coarse_size  # Size of each coarse grid cell (6x6)

        self.Env = Env()
        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        # Initialize coarse grid parameters
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        self.coarse_x_range = self.x_range // self.coarse_size + 1
        self.coarse_y_range = self.y_range // self.coarse_size + 1
        
        # Build coarse grid abstraction
        self.coarse_grid = self.build_coarse_grid()
        
        # Translate start and goal to coarse coordinates
        self.c_start = self.to_coarse_coords(s_start)
        self.c_goal = self.to_coarse_coords(s_goal)

    def build_coarse_grid(self):
        """
        Build the coarse grid representation
        :return: dictionary of coarse grid cells and their properties
        """
        coarse_grid = {}
        
        # Initialize coarse grid cells
        for i in range(self.coarse_x_range):
            for j in range(self.coarse_y_range):
                coarse_grid[(i, j)] = {
                    'traversable': True,
                    'sparse': True,  # Initially assume all cells are sparse
                    'fine_coords': [],
                    'connections': {},  # Store connectivity info to neighboring cells
                    'obstacle_ratio': 0.0  # Track how much of the cell is blocked
                }
                
                # Count total possible cells and obstacle cells
                total_cells = 0
                obstacle_cells = 0
                
                # Add fine coordinates that belong to this coarse cell
                for x in range(i * self.coarse_size, (i + 1) * self.coarse_size):
                    for y in range(j * self.coarse_size, (j + 1) * self.coarse_size):
                        if x < self.x_range and y < self.y_range:
                            total_cells += 1
                            if (x, y) not in self.obs:
                                coarse_grid[(i, j)]['fine_coords'].append((x, y))
                            else:
                                obstacle_cells += 1
                
                # Calculate obstacle ratio
                if total_cells > 0:
                    coarse_grid[(i, j)]['obstacle_ratio'] = obstacle_cells / total_cells
                
                # Mark as non-traversable if empty or highly obstructed (>70% obstacles)
                if not coarse_grid[(i, j)]['fine_coords'] or coarse_grid[(i, j)]['obstacle_ratio'] > 0.7:
                    coarse_grid[(i, j)]['traversable'] = False
                
                # Mark as non-sparse if obstacles are clustered (>30% obstacles)
                if coarse_grid[(i, j)]['obstacle_ratio'] > 0.3:
                    coarse_grid[(i, j)]['sparse'] = False
        
        # Check connectivity between adjacent coarse cells
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # 8-connectivity
        
        for i in range(self.coarse_x_range):
            for j in range(self.coarse_y_range):
                if not coarse_grid[(i, j)]['traversable']:
                    continue
                    
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    
                    # Skip if neighbor is out of bounds
                    if not (0 <= ni < self.coarse_x_range and 0 <= nj < self.coarse_y_range):
                        continue
                        
                    # Skip if neighbor is not traversable
                    if not coarse_grid[(ni, nj)]['traversable']:
                        continue
                    
                    # For diagonal moves, check both adjacent cardinal cells are traversable
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (not 0 <= i+dx < self.coarse_x_range or
                            not 0 <= j < self.coarse_y_range or
                            not coarse_grid[(i+dx, j)]['traversable']):
                            continue
                        if (not 0 <= i < self.coarse_x_range or
                            not 0 <= j+dy < self.coarse_y_range or
                            not coarse_grid[(i, j+dy)]['traversable']):
                            continue
                    
                    # Check if there's a valid path between these two coarse cells
                    path_exists, connected_points = self.verify_connectivity(
                        coarse_grid[(i, j)]['fine_coords'], 
                        coarse_grid[(ni, nj)]['fine_coords']
                    )
                    
                    if path_exists:
                        # Store connection points for later use in path refinement
                        coarse_grid[(i, j)]['connections'][(ni, nj)] = connected_points
        
        return coarse_grid

    def verify_connectivity(self, points1, points2):
        """
        Verify if there's a valid path between two sets of points
        :param points1: list of points from first coarse cell
        :param points2: list of points from second coarse cell
        :return: (path_exists, connected_points)
        """
        # If either set is empty, there's no connectivity
        if not points1 or not points2:
            return False, []
            
        # Sample points to test (for efficiency)
        points1_sample = self.sample_points(points1, 5)  # Try up to 5 start points
        points2_sample = self.sample_points(points2, 5)  # Try up to 5 goal points
        
        # Try to find a path between any pair of points
        for p1 in points1_sample:
            for p2 in points2_sample:
                # For adjacent coarse cells, if points are close enough
                # and there's no obstacle between them, add as a connection
                manhattan_dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                if manhattan_dist <= 2 and not self.is_collision(p1, p2):
                    return True, [(p1, p2)]
                    
                # For points further apart, run a small A* search
                path, _ = self.fine_grid_astar(p1, p2, max_iter=100)
                if path:  # If a path exists
                    return True, [(p1, p2)]
                    
        # No valid path found between any sampled points
        return False, []
        
    def sample_points(self, points, max_samples):
        """
        Sample a representative set of points from a larger set
        :param points: list of points to sample from
        :param max_samples: maximum number of samples to take
        :return: sampled points
        """
        if len(points) <= max_samples:
            return points
            
        # Take points distributed throughout the list 
        step = len(points) // max_samples
        return [points[i] for i in range(0, len(points), step)][:max_samples]

    def to_coarse_coords(self, fine_coords):
        """
        Convert fine grid coordinates to coarse grid coordinates
        :param fine_coords: (x, y) in fine grid
        :return: (x, y) in coarse grid
        """
        return (fine_coords[0] // self.coarse_size, 
                fine_coords[1] // self.coarse_size)

    def to_fine_coords(self, coarse_coords):
        """
        Convert coarse grid coordinates to fine grid coordinates (center point)
        :param coarse_coords: (x, y) in coarse grid
        :return: (x, y) in fine grid
        """
        x_center = coarse_coords[0] * self.coarse_size + self.coarse_size // 2
        y_center = coarse_coords[1] * self.coarse_size + self.coarse_size // 2
        
        return (x_center, y_center)

    def get_coarse_neighbors(self, coarse_node):
        """
        Get connected neighboring coarse grid cells
        :param coarse_node: (x, y) in coarse grid
        :return: list of connected neighboring coarse cells
        """
        neighbors = []
        
        # Only return neighbors that have verified connectivity
        if coarse_node in self.coarse_grid:
            for neighbor in self.coarse_grid[coarse_node]['connections'].keys():
                neighbors.append(neighbor)
                
        return neighbors

    def get_fine_neighbors(self, fine_node):
        """
        Get traversable neighboring fine grid cells
        :param fine_node: (x, y) in fine grid
        :return: list of traversable neighboring fine cells
        """
        return [(fine_node[0] + u[0], fine_node[1] + u[1]) for u in self.u_set 
                if (fine_node[0] + u[0], fine_node[1] + u[1]) not in self.obs and
                0 <= fine_node[0] + u[0] < self.x_range and
                0 <= fine_node[1] + u[1] < self.y_range]

    def coarse_heuristic(self, coarse_node):
        """
        Calculate heuristic value for coarse grid
        :param coarse_node: coarse grid coordinates
        :return: heuristic value
        """
        if self.heuristic_type == "manhattan":
            return abs(self.c_goal[0] - coarse_node[0]) + abs(self.c_goal[1] - coarse_node[1])
        else:  # euclidean
            return math.hypot(self.c_goal[0] - coarse_node[0], self.c_goal[1] - coarse_node[1])

    def fine_heuristic(self, fine_node, goal_node):
        """
        Calculate heuristic value for fine grid
        :param fine_node: fine grid coordinates
        :param goal_node: goal coordinates in fine grid
        :return: heuristic value
        """
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - fine_node[0]) + abs(goal_node[1] - fine_node[1])
        else:  # euclidean
            return math.hypot(goal_node[0] - fine_node[0], goal_node[1] - fine_node[1])

    def high_level_search(self):
        """
        Perform A* search on the coarse grid, preferring sparse cells
        :return: coarse path, visited coarse cells
        """
        open_set = []
        closed_set = []
        parent = {self.c_start: self.c_start}
        g = {self.c_start: 0, self.c_goal: math.inf}
        
        heapq.heappush(open_set, (self.coarse_heuristic(self.c_start), self.c_start))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            closed_set.append(current)
            
            if current == self.c_goal:
                break
                
            for neighbor in self.get_coarse_neighbors(current):
                # Apply higher cost for non-sparse cells to prefer sparse cell paths
                cell_cost = 1.0
                if not self.coarse_grid[neighbor]['sparse']:
                    cell_cost = 2.0  # Higher cost for cells with obstacle clusters
                
                new_cost = g[current] + cell_cost
                
                if neighbor not in g:
                    g[neighbor] = math.inf
                    
                if new_cost < g[neighbor]:
                    g[neighbor] = new_cost
                    parent[neighbor] = current
                    f = new_cost + self.coarse_heuristic(neighbor)
                    heapq.heappush(open_set, (f, neighbor))
        
        # Extract coarse path
        coarse_path = self.extract_path(parent, self.c_start, self.c_goal)
        
        return coarse_path, closed_set

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) collides with obstacles.
        :param s_start: start node
        :param s_end: end node
        :return: True: collision / False: no collision
        """
        if s_start in self.obs or s_end in self.obs:
            return True

        # For diagonal moves, check the square formed by the two points
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True
                
        # Additional collision checking for longer segments using Bresenham's line algorithm
        if abs(s_end[0] - s_start[0]) > 1 or abs(s_end[1] - s_start[1]) > 1:
            line_points = self.bresenham_line(s_start[0], s_start[1], s_end[0], s_end[1])
            for point in line_points:
                if point in self.obs:
                    return True

        return False
        
    def bresenham_line(self, x0, y0, x1, y1):
        """
        Implementation of Bresenham's line algorithm to get all points on a line
        :param x0, y0: starting point
        :param x1, y1: ending point
        :return: list of points on the line
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            points.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        points.append((x1, y1))
        return points
        
    def low_level_search(self, start_cell, end_cell):
        """
        Find path between two adjacent coarse cells using A*
        :param start_cell: starting coarse cell coordinates
        :param end_cell: ending coarse cell coordinates
        :return: fine path between entry and exit points of the coarse cells, visited cells
        """
        # Check if we have pre-computed connection points
        if end_cell in self.coarse_grid[start_cell]['connections']:
            # Use the precomputed connection points
            connected_points = self.coarse_grid[start_cell]['connections'][end_cell]
            if connected_points:
                start_point, end_point = connected_points[0]  # Use the first connected pair
                path, visited = self.fine_grid_astar(start_point, end_point)
                return path, visited
        
        # Fallback to finding best connection points
        start_points = self.coarse_grid[start_cell]['fine_coords']
        end_points = self.coarse_grid[end_cell]['fine_coords']
        
        if not start_points or not end_points:
            return [], []
        
        # Try multiple start and end points until we find a valid path
        for start_point in self.sample_points(start_points, 5):
            for end_point in self.sample_points(end_points, 5):
                path, visited = self.fine_grid_astar(start_point, end_point)
                if path:  # Found a valid path
                    return path, visited
        
        # If we couldn't find a path, try best entry/exit points
        start_point = self.find_best_entry_point(start_points, end_points)
        end_point = self.find_best_entry_point(end_points, start_points)
        
        path, visited = self.fine_grid_astar(start_point, end_point)
        return path, visited

    def find_best_entry_point(self, points_from, points_to):
        """
        Find the best entry/exit point between two coarse cells
        :param points_from: candidate points in the from cell
        :param points_to: candidate points in the to cell
        :return: best point with minimum distance to points in the other cell
        """
        best_point = points_from[0]
        min_dist = math.inf
        
        # Find point with minimum average distance to all points in the other cell
        for p_from in points_from:
            # For efficiency, sample a few points from the to_cell
            sample_size = min(10, len(points_to))
            sample_points = points_to[:sample_size]
            
            avg_dist = sum(math.hypot(p_from[0] - p_to[0], p_from[1] - p_to[1]) 
                          for p_to in sample_points) / len(sample_points)
            
            if avg_dist < min_dist:
                min_dist = avg_dist
                best_point = p_from
                
        return best_point

    def fine_grid_astar(self, start, goal, max_iter=float('inf')):
        """
        A* search on fine grid between two points
        :param start: start coordinates
        :param goal: goal coordinates
        :param max_iter: maximum number of iterations (nodes to expand)
        :return: fine path, visited cells
        """
        open_set = []
        closed_set = []
        parent = {start: start}
        g = {start: 0, goal: math.inf}
        
        heapq.heappush(open_set, (self.fine_heuristic(start, goal), start))
        
        iterations = 0
        
        while open_set and iterations < max_iter:
            iterations += 1
            
            _, current = heapq.heappop(open_set)
            closed_set.append(current)
            
            if current == goal:
                break
                
            for neighbor in self.get_fine_neighbors(current):
                # Calculate actual cost between cells
                new_cost = g[current] + math.hypot(neighbor[0] - current[0], 
                                                  neighbor[1] - current[1])
                
                if neighbor not in g:
                    g[neighbor] = math.inf
                    
                if new_cost < g[neighbor]:
                    g[neighbor] = new_cost
                    parent[neighbor] = current
                    f = new_cost + self.fine_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
        
        # Extract fine path
        fine_path = self.extract_path(parent, start, goal)
        
        return fine_path, closed_set

    def extract_path(self, parent, start, goal):
        """
        Extract path from parent dictionary
        :param parent: parent dictionary
        :param start: start node
        :param goal: goal node
        :return: path from start to goal
        """
        path = [goal]
        current = goal
        
        while current != start:
            if current not in parent:
                return []  # No path found
            current = parent[current]
            path.append(current)
            
        return list(reversed(path))

    def searching(self):
        """
        Main hierarchical A* search function
        :return: complete fine path, coarse path, visited coarse cells, visited fine cells
        """
        # Perform high-level search on coarse grid (preferring sparse cells)
        coarse_path, visited_coarse = self.high_level_search()
        
        if not coarse_path:
            # If no coarse path found, fallback to standard A*
            print("No valid coarse path found, falling back to standard A*")
            standard_path, visited = self.fine_grid_astar(self.s_start, self.s_goal)
            return standard_path, [], [], visited
        
        # Refine path through low-level search
        fine_path = []
        visited_fine = []
        
        # Connect start point to first coarse cell
        # Find a good entry point to the first coarse cell
        first_coarse_cell = coarse_path[0]
        if first_coarse_cell != self.c_start or not fine_path:
            start_points = self.coarse_grid[first_coarse_cell]['fine_coords']
            if start_points:
                best_path = None
                best_visited = None
                
                for point in self.sample_points(start_points, 5):
                    path, visited = self.fine_grid_astar(self.s_start, point)
                    if path and (best_path is None or len(path) < len(best_path)):
                        best_path = path
                        best_visited = visited
                
                if best_path:
                    fine_path = best_path
                    visited_fine = best_visited
        
        # Connect through all coarse cells
        for i in range(len(coarse_path) - 1):
            current_cell = coarse_path[i]
            next_cell = coarse_path[i + 1]
            
            # Find path between adjacent coarse cells
            segment_path, segment_visited = self.low_level_search(current_cell, next_cell)
            
            # If couldn't find a path, try harder
            if not segment_path:
                start_points = self.coarse_grid[current_cell]['fine_coords']
                end_points = self.coarse_grid[next_cell]['fine_coords']
                
                for p1 in self.sample_points(start_points, 5):
                    for p2 in self.sample_points(end_points, 5):
                        segment_path, segment_visited = self.fine_grid_astar(p1, p2)
                        if segment_path:
                            break
                    if segment_path:
                        break
            
            # Connect segments
            if fine_path and segment_path:
                # Connect if endpoints don't match
                if fine_path[-1] != segment_path[0]:
                    connector, connector_visited = self.fine_grid_astar(fine_path[-1], segment_path[0])
                    if connector:
                        fine_path.extend(connector[1:])  # Avoid duplicate points
                        visited_fine.extend(connector_visited)
                
                # Add current segment (avoid duplicates)
                if fine_path and fine_path[-1] == segment_path[0]:
                    fine_path.extend(segment_path[1:])
                else:
                    fine_path.extend(segment_path)
                    
                visited_fine.extend(segment_visited)
            elif segment_path:
                fine_path.extend(segment_path)
                visited_fine.extend(segment_visited)
        
        # Connect to goal point
        if fine_path and fine_path[-1] != self.s_goal:
            goal_path, goal_visited = self.fine_grid_astar(fine_path[-1], self.s_goal)
            
            if goal_path:
                # Add path to goal (avoid duplicates)
                if fine_path[-1] == goal_path[0]:
                    fine_path.extend(goal_path[1:])
                else:
                    fine_path.extend(goal_path)
                visited_fine.extend(goal_visited)
        
        # Ensure valid path from start to goal
        if not fine_path or fine_path[0] != self.s_start or fine_path[-1] != self.s_goal:
            # Fallback to direct A* but keep coarse path for visualization
            print("Invalid hierarchical path - using direct A*")
            direct_path, direct_visited = self.fine_grid_astar(self.s_start, self.s_goal)
            if direct_path:
                return direct_path, coarse_path, visited_coarse, direct_visited
        
        # Post-process path for smoothness
        fine_path = self.post_process_path(fine_path)
        
        return fine_path, coarse_path, visited_coarse, visited_fine
        
    def post_process_path(self, path):
        """
        Post-process the path to ensure it avoids obstacles and is reasonably smooth
        :param path: input path to process
        :return: processed path
        """
        if not path or len(path) < 3:
            return path
            
        # Remove redundant waypoints (path smoothing)
        i = 0
        processed_path = [path[0]]
        
        while i < len(path) - 1:
            current = path[i]
            found_jump = False
            
            # Look ahead as far as possible for direct connections
            for j in range(len(path) - 1, i, -1):
                if not self.is_collision(current, path[j]):
                    processed_path.append(path[j])
                    i = j
                    found_jump = True
                    break
            
            # If no valid jump found, take the next point
            if not found_jump:
                i += 1
                if i < len(path):
                    processed_path.append(path[i])
                    
        # Ensure the last point is included
        if processed_path[-1] != path[-1]:
            processed_path.append(path[-1])
            
        return processed_path

    def visualize_coarse_grid(self):
        """
        Create visualization data for the coarse grid
        :return: list of coarse grid cell boundaries for visualization
        """
        grid_lines = []
        
        # Create horizontal lines
        for i in range(self.coarse_y_range + 1):
            y = i * self.coarse_size
            grid_lines.append([(0, y), (self.x_range, y)])
            
        # Create vertical lines
        for i in range(self.coarse_x_range + 1):
            x = i * self.coarse_size
            grid_lines.append([(x, 0), (x, self.y_range)])
            
        return grid_lines

    def get_sparse_cells(self):
        """
        Get list of sparse cells (cells without obstacle clusters)
        :return: list of sparse cell coordinates
        """
        return [cell for cell, props in self.coarse_grid.items() 
                if props['sparse'] and props['traversable']]


def main():
    """Main function to run the Hierarchical A* algorithm"""
    s_start = (5, 5)
    s_goal = (45, 25)

    hierarchical_astar = HierarchicalAStar(s_start, s_goal, "euclidean")
    plot = Plotting(s_start, s_goal)

    fine_path, coarse_path, visited_coarse, visited_fine = hierarchical_astar.searching()
    
    # Convert coarse path to fine coordinates for visualization
    coarse_path_fine = [hierarchical_astar.to_fine_coords(cell) for cell in coarse_path]
    
    # Create grid lines for visualization
    grid_lines = hierarchical_astar.visualize_coarse_grid()
    
    # Animate and save as GIF
    plot.animation_hierarchical_astar(fine_path, coarse_path_fine, visited_fine, grid_lines, 
                                     "008_Hierarchical_Astar", save_gif=True)


if __name__ == '__main__':
    main()
