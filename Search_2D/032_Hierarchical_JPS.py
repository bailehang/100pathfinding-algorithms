"""
Hierarchical JPS (Jump Point Search)
@author: Trae AI (based on JPS+ and Hierarchical A* implementations)
"""

import os
import sys
import math
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class HierarchicalJPS:
    """
    Hierarchical Jump Point Search Algorithm.
    Combines hierarchical grid abstraction with JPS efficiency.
    Uses a two-level approach:
    1. High-level planning with a coarse grid containing jump points
    2. Low-level refinement using JPS between coarse cells
    """
    def __init__(self, s_start, s_goal, heuristic_type, coarse_size=6):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.coarse_size = coarse_size  # Size of each coarse grid cell (6x6)

        self.Env = env.Env()
        self.u_set = self.Env.motions  # feasible input set (8 directions)
        self.obs = self.Env.obs  # position of obstacles

        # Initialize coarse grid parameters
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        self.coarse_x_range = self.x_range // self.coarse_size + 1
        self.coarse_y_range = self.y_range // self.coarse_size + 1
        
        # For online search
        self.OPEN = []
        self.CLOSED = set()
        self.PARENT = dict()
        self.g = dict()

        # Build coarse grid abstraction
        self.coarse_grid = {} # Will store coarse cells and their properties
        
        # Translate start and goal to coarse coordinates
        self.c_start = self.to_coarse_coords(s_start)
        self.c_goal = self.to_coarse_coords(s_goal)

        # For visualization
        self.plot_util = plotting.Plotting(self.s_start, self.s_goal)
        self.fig, self.ax = plt.subplots()

    def _is_valid(self, node):
        """Check if a node is within bounds."""
        return 0 <= node[0] < self.x_range and 0 <= node[1] < self.y_range

    def _is_obstacle(self, node):
        """Check if a node is an obstacle or out of bounds."""
        if not self._is_valid(node):
            return True
        return node in self.obs

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
        :return: (x, y) in fine grid (center of coarse cell)
        """
        x_center = coarse_coords[0] * self.coarse_size + self.coarse_size // 2
        y_center = coarse_coords[1] * self.coarse_size + self.coarse_size // 2
        
        return (x_center, y_center)

    def _jps_find_jump_point(self, parent, current, max_steps=100):
        """
        Core JPS logic to find the next jump point from 'parent' towards 'current'.
        Returns the jump point if found, otherwise None.
        :param parent: parent node
        :param current: current node to explore from
        :param max_steps: maximum number of steps to prevent infinite loops
        :return: jump point if found, otherwise None
        """
        if not self._is_valid(current) or self._is_obstacle(current):
            return None

        if current == self.s_goal:
            return current

        dx = current[0] - parent[0]
        dy = current[1] - parent[1]

        # Normalize direction
        norm_dx = dx // max(abs(dx), 1) if dx != 0 else 0
        norm_dy = dy // max(abs(dy), 1) if dy != 0 else 0

        # Check for forced neighbors
        # Diagonal movement
        if norm_dx != 0 and norm_dy != 0:
            # Check horizontally and vertically for forced neighbors
            # Forced neighbor along x-axis?
            if self._is_obstacle((current[0] - norm_dx, current[1])) and \
               not self._is_obstacle((current[0] - norm_dx, current[1] + norm_dy)):
                return current
            # Forced neighbor along y-axis?
            if self._is_obstacle((current[0], current[1] - norm_dy)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] - norm_dy)):
                return current

            # Recursive calls for diagonal movement
            if self._jps_find_jump_point(current, (current[0] + norm_dx, current[1]), max_steps-1): # Horizontal search
                return current
            if self._jps_find_jump_point(current, (current[0], current[1] + norm_dy), max_steps-1): # Vertical search
                return current
        # Straight movement (Horizontal)
        elif norm_dx != 0: # dx != 0, dy == 0
            # Forced neighbor above?
            if self._is_obstacle((current[0], current[1] + 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] + 1)):
                return current
            # Forced neighbor below?
            if self._is_obstacle((current[0], current[1] - 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] - 1)):
                return current
        # Straight movement (Vertical)
        else: # dx == 0, dy != 0
            # Forced neighbor to the right?
            if self._is_obstacle((current[0] + 1, current[1])) and \
               not self._is_obstacle((current[0] + 1, current[1] + norm_dy)):
                return current
            # Forced neighbor to the left?
            if self._is_obstacle((current[0] - 1, current[1])) and \
               not self._is_obstacle((current[0] - 1, current[1] + norm_dy)):
                return current
        
        # Continue jumping in the same direction if within step limit
        if max_steps <= 0:
            return None
            
        next_node = (current[0] + norm_dx, current[1] + norm_dy)
        return self._jps_find_jump_point(current, next_node, max_steps-1)

    def build_coarse_grid(self):
        """
        Build the coarse grid representation and identify jump points within each cell
        :return: dictionary of coarse grid cells and their properties
        """
        print("Building coarse grid...")
        self.plot_util.plot_grid("Hierarchical JPS - Building Coarse Grid")
        plt.title("Hierarchical JPS - Building Coarse Grid")
        
        # Initialize coarse grid cells
        for i in range(self.coarse_x_range):
            for j in range(self.coarse_y_range):
                self.coarse_grid[(i, j)] = {
                    'traversable': True,
                    'sparse': True,  # Initially assume all cells are sparse
                    'fine_coords': [],
                    'connections': {},  # Store connectivity info to neighboring cells
                    'jump_points': set(),  # Store jump points in this cell
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
                                self.coarse_grid[(i, j)]['fine_coords'].append((x, y))
                            else:
                                obstacle_cells += 1
                
                # Calculate obstacle ratio
                if total_cells > 0:
                    self.coarse_grid[(i, j)]['obstacle_ratio'] = obstacle_cells / total_cells
                
                # Mark as non-traversable if empty or highly obstructed (>70% obstacles)
                if not self.coarse_grid[(i, j)]['fine_coords'] or self.coarse_grid[(i, j)]['obstacle_ratio'] > 0.7:
                    self.coarse_grid[(i, j)]['traversable'] = False
                
                # Mark as non-sparse if obstacles are clustered (>30% obstacles)
                if self.coarse_grid[(i, j)]['obstacle_ratio'] > 0.3:
                    self.coarse_grid[(i, j)]['sparse'] = False
                    
                # Visualize cell sparsity
                cell_center = self.to_fine_coords((i, j))
                if self.coarse_grid[(i, j)]['traversable']:
                    if self.coarse_grid[(i, j)]['sparse']:
                        # Light blue for sparse cells
                        plt.plot(cell_center[0], cell_center[1], 'o', color='skyblue', markersize=5, alpha=0.5)
                    else:
                        # Yellow for dense but traversable cells
                        plt.plot(cell_center[0], cell_center[1], 'o', color='gold', markersize=5, alpha=0.5)
        
        plt.pause(0.01)
        print("Coarse grid built. Identifying jump points...")
        
        return self.coarse_grid

    def identify_cell_jump_points(self):
        """
        Identify jump points within each traversable coarse cell
        """
        print("Identifying jump points within coarse cells...")
        
        # Define a function to determine if a point is a potential jump point
        def is_potential_jp(point, cell_coords):
            # Add the start and goal point if they're in this cell
            if point == self.s_start or point == self.s_goal:
                return True
                
            # Check all 8 directions from this point
            for dx, dy in self.u_set:
                next_point = (point[0] + dx, point[1] + dy)
                # Try to find a jump point in this direction
                jp = self._jps_find_jump_point(point, next_point, max_steps=self.coarse_size*2)
                if jp and self.to_coarse_coords(jp) != cell_coords:
                    # If we found a jump point in another cell, this is a boundary jump point
                    return True
            
            return False
            
        # For each traversable cell
        for cell_coords, cell_data in self.coarse_grid.items():
            if not cell_data['traversable']:
                continue
                
            # Add start and goal as jump points if they're in this cell
            if self.to_coarse_coords(self.s_start) == cell_coords:
                cell_data['jump_points'].add(self.s_start)
            if self.to_coarse_coords(self.s_goal) == cell_coords:
                cell_data['jump_points'].add(self.s_goal)
                
            # Sample points within the cell to check for jump points
            # We check points near the boundary of the cell as they're more likely to be jump points
            boundary_points = []
            i, j = cell_coords
            
            # Add boundary points of the cell to check
            for x in range(i * self.coarse_size, (i + 1) * self.coarse_size):
                for y in range(j * self.coarse_size, (j + 1) * self.coarse_size):
                    # If point is valid and not an obstacle
                    if (x < self.x_range and y < self.y_range and (x, y) not in self.obs):
                        # Check if point is near a boundary of the cell
                        if (x == i * self.coarse_size or x == (i + 1) * self.coarse_size - 1 or
                            y == j * self.coarse_size or y == (j + 1) * self.coarse_size - 1):
                            boundary_points.append((x, y))
            
            # Check each boundary point to see if it's a jump point
            cell_jps = set()
            for point in boundary_points:
                if is_potential_jp(point, cell_coords):
                    cell_jps.add(point)
                    
            # Visualize the identified jump points in this cell
            for jp in cell_jps:
                plt.plot(jp[0], jp[1], 'mo', markersize=4)
                
            # Store the jump points in the cell data
            cell_data['jump_points'].update(cell_jps)
            
            # Progress indicator
            if len(cell_jps) > 0:
                print(f"Cell {cell_coords}: Found {len(cell_jps)} jump points")
                plt.pause(0.01)
        
        plt.pause(0.5)
        print("Jump point identification complete.")

    def compute_cell_connections(self):
        """
        Compute connections between coarse cells using jump points
        """
        print("Computing connections between coarse cells...")
        
        # Iterate through each pair of adjacent cells
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # 8-connectivity
        
        for i in range(self.coarse_x_range):
            for j in range(self.coarse_y_range):
                if not self.coarse_grid.get((i, j), {}).get('traversable', False):
                    continue
                    
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    
                    # Skip if neighbor is out of bounds
                    if not (0 <= ni < self.coarse_x_range and 0 <= nj < self.coarse_y_range):
                        continue
                        
                    # Skip if neighbor is not traversable
                    if not self.coarse_grid.get((ni, nj), {}).get('traversable', False):
                        continue
                    
                    # For diagonal moves, check both adjacent cardinal cells are traversable
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (not 0 <= i+dx < self.coarse_x_range or
                            not 0 <= j < self.coarse_y_range or
                            not self.coarse_grid.get((i+dx, j), {}).get('traversable', False)):
                            continue
                        if (not 0 <= i < self.coarse_x_range or
                            not 0 <= j+dy < self.coarse_y_range or
                            not self.coarse_grid.get((i, j+dy), {}).get('traversable', False)):
                            continue
                    
                    # Get jump points from both cells
                    cell1_jps = self.coarse_grid[(i, j)]['jump_points']
                    cell2_jps = self.coarse_grid[(ni, nj)]['jump_points']
                    
                    if not cell1_jps or not cell2_jps:
                        continue
                    
                    # Find connections between jump points in the two cells
                    connection_found = False
                    for jp1 in cell1_jps:
                        for jp2 in cell2_jps:
                            # Use direct connection if there's a direct path
                            if not self.is_collision(jp1, jp2):
                                if (ni, nj) not in self.coarse_grid[(i, j)]['connections']:
                                    self.coarse_grid[(i, j)]['connections'][(ni, nj)] = []
                                self.coarse_grid[(i, j)]['connections'][(ni, nj)].append((jp1, jp2))
                                
                                # Visualize connection
                                plt.plot([jp1[0], jp2[0]], [jp1[1], jp2[1]], 'c-', linewidth=0.5, alpha=0.5)
                                connection_found = True
                    
                    if connection_found:
                        print(f"Found connection: Cell {(i, j)} to Cell {(ni, nj)}")
                        plt.pause(0.01)
        
        plt.pause(0.5)
        print("Cell connections computed.")

    def precompute_hierarchical_graph(self):
        """
        Precompute the hierarchical graph for JPS
        """
        print("Starting hierarchical graph precomputation...")
        
        # Build the coarse grid
        self.build_coarse_grid()
        
        # Draw grid lines for visualization
        grid_lines = self.visualize_coarse_grid()
        for line in grid_lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                    color='gray', linestyle='--', alpha=0.3)
        
        # Identify jump points within cells
        self.identify_cell_jump_points()
        
        # Compute connections between cells
        self.compute_cell_connections()
        
        # Mark start and goal on the map
        plt.plot(self.s_start[0], self.s_start[1], 'bs', markersize=8)
        plt.plot(self.s_goal[0], self.s_goal[1], 'gs', markersize=8)
        
        plt.title("Hierarchical JPS - Precomputation Complete")
        plt.pause(1)
        
        print("Hierarchical graph precomputation complete.")

    def high_level_search(self):
        """
        Perform high-level A* search on the coarse grid, preferring sparse cells
        :return: coarse path, visited coarse cells
        """
        print("Starting high-level search...")
        plt.title("Hierarchical JPS - High-Level Search")
        
        open_set = []
        closed_set = []
        parent = {self.c_start: self.c_start}
        g = {self.c_start: 0, self.c_goal: math.inf}
        
        heapq.heappush(open_set, (self.coarse_heuristic(self.c_start), self.c_start))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            closed_set.append(current)
            
            # Visualize the explored coarse cell
            cell_center = self.to_fine_coords(current)
            plt.plot(cell_center[0], cell_center[1], 'yo', markersize=6)
            plt.pause(0.05)
            
            if current == self.c_goal:
                print("Goal found in high-level search!")
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
                    
                    # Visualize consideration of neighbor
                    neighbor_center = self.to_fine_coords(neighbor)
                    plt.plot([cell_center[0], neighbor_center[0]], 
                             [cell_center[1], neighbor_center[1]], 'b-', alpha=0.3)
                    plt.pause(0.01)
        
        # Extract coarse path
        coarse_path = self.extract_path(parent, self.c_start, self.c_goal)
        
        if coarse_path:
            print(f"High-level path found with {len(coarse_path)} coarse cells")
            
            # Visualize the coarse path
            coarse_centers = [self.to_fine_coords(cell) for cell in coarse_path]
            x_coords = [center[0] for center in coarse_centers]
            y_coords = [center[1] for center in coarse_centers]
            plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
            
            # Highlight the path cells
            for center in coarse_centers:
                plt.plot(center[0], center[1], 'bo', markersize=7)
                
            plt.pause(0.5)
        else:
            print("No high-level path found.")
        
        return coarse_path, closed_set

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

    def jps_local_search(self, start_point, goal_point):
        """
        Use JPS to find a path between two points
        :param start_point: start coordinates
        :param goal_point: goal coordinates
        :return: path, visited nodes
        """
        open_set = []
        closed_set = set()
        parent = {start_point: start_point}
        g = {start_point: 0}
        visited = []
        
        heapq.heappush(open_set, (self.jps_heuristic(start_point, goal_point), start_point))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            visited.append(current)
            
            if current == goal_point:
                break
                
            # Get successors using JPS principle
            successors = self.jps_get_successors(current, parent.get(current, current), goal_point)
            
            for successor in successors:
                if successor in closed_set:
                    continue
                    
                # Calculate the cost from start to successor
                new_cost = g[current] + self.cost(current, successor)
                
                if successor not in g or new_cost < g[successor]:
                    g[successor] = new_cost
                    parent[successor] = current
                    f = new_cost + self.jps_heuristic(successor, goal_point)
                    heapq.heappush(open_set, (f, successor))
        
        # Extract path
        path = self.extract_path(parent, start_point, goal_point)
        
        return path, visited

    def jps_get_successors(self, node, parent, goal):
        """
        Get successor nodes using JPS principles
        :param node: current node
        :param parent: parent of the current node
        :param goal: goal node
        :return: list of successor nodes
        """
        successors = []
        
        # If goal is adjacent, add it directly
        if abs(node[0] - goal[0]) <= 1 and abs(node[1] - goal[1]) <= 1 and not self.is_collision(node, goal):
            successors.append(goal)
            return successors
            
        # Calculate the direction from parent to current
        dx = node[0] - parent[0]
        dy = node[1] - parent[1]
        
        # Normalized direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        # For the first node, try all 8 directions
        if node == parent:
            for direction in self.u_set:
                nx, ny = node[0] + direction[0], node[1] + direction[1]
                next_node = (nx, ny)
                
                if not self._is_valid(next_node) or self._is_obstacle(next_node):
                    continue
                    
                jp = self._jps_find_jump_point(node, next_node)
                if jp:
                    successors.append(jp)
            return successors
            
        # For straight movement, only consider straight and forced neighbors
        if dx == 0 or dy == 0:
            # Continue in the same direction
            next_node = (node[0] + dx, node[1] + dy)
            jp = self._jps_find_jump_point(node, next_node)
            if jp:
                successors.append(jp)
                
            # Check for forced neighbors
            if dx != 0:  # Horizontal movement
                if self._is_obstacle((node[0], node[1] + 1)):
                    # Check for forced neighbor below
                    forced = (node[0] + dx, node[1] + 1)
                    if not self._is_obstacle(forced):
                        jp = self._jps_find_jump_point(node, forced)
                        if jp:
                            successors.append(jp)
                            
                if self._is_obstacle((node[0], node[1] - 1)):
                    # Check for forced neighbor above
                    forced = (node[0] + dx, node[1] - 1)
                    if not self._is_obstacle(forced):
                        jp = self._jps_find_jump_point(node, forced)
                        if jp:
                            successors.append(jp)
            else:  # Vertical movement
                if self._is_obstacle((node[0] + 1, node[1])):
                    # Check for forced neighbor to the right
                    forced = (node[0] + 1, node[1] + dy)
                    if not self._is_obstacle(forced):
                        jp = self._jps_find_jump_point(node, forced)
                        if jp:
                            successors.append(jp)
                            
                if self._is_obstacle((node[0] - 1, node[1])):
                    # Check for forced neighbor to the left
                    forced = (node[0] - 1, node[1] + dy)
                    if not self._is_obstacle(forced):
                        jp = self._jps_find_jump_point(node, forced)
                        if jp:
                            successors.append(jp)
        # For diagonal movement
        else:
            # Continue in the same diagonal direction
            next_node = (node[0] + dx, node[1] + dy)
            jp = self._jps_find_jump_point(node, next_node)
            if jp:
                successors.append(jp)
                
            # Check horizontal and vertical neighbors
            horizontal = (node[0] + dx, node[1])
            jp = self._jps_find_jump_point(node, horizontal)
            if jp:
                successors.append(jp)
                
            vertical = (node[0], node[1] + dy)
            jp = self._jps_find_jump_point(node, vertical)
            if jp:
                successors.append(jp)
                
            # Check for forced neighbors (when moving diagonally)
            if self._is_obstacle((node[0] - dx, node[1])):
                forced = (node[0], node[1] + dy)
                if not self._is_obstacle(forced):
                    jp = self._jps_find_jump_point(node, forced)
                    if jp:
                        successors.append(jp)
                        
            if self._is_obstacle((node[0], node[1] - dy)):
                forced = (node[0] + dx, node[1])
                if not self._is_obstacle(forced):
                    jp = self._jps_find_jump_point(node, forced)
                    if jp:
                        successors.append(jp)
                        
        return successors

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) collides with obstacles.
        :param s_start: start node
        :param s_end: end node
        :return: True: collision / False: no collision
        """
        if s_start in self.obs or s_end in self.obs:
            return True

        # Check for any obstacles in the path using Bresenham's line algorithm
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

    def jps_heuristic(self, s_node, goal_node):
        """
        Calculate heuristic value for JPS local search
        :param s_node: current node
        :param goal_node: goal node
        :return: heuristic value
        """
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - s_node[0]) + abs(goal_node[1] - s_node[1])
        else:  # euclidean
            return math.hypot(goal_node[0] - s_node[0], goal_node[1] - s_node[1])

    def cost(self, s_start, s_end):
        """
        Calculate the cost between two nodes
        :param s_start: start node
        :param s_end: end node
        :return: cost value
        """
        return math.hypot(s_end[0] - s_start[0], s_end[1] - s_start[1])

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

    def low_level_refine(self, coarse_path):
        """
        Refine coarse path using JPS local search
        :param coarse_path: path of coarse cells
        :return: fine path, visited nodes
        """
        if not coarse_path:
            return [], []
            
        print("Refining path with JPS local search...")
        plt.title("Hierarchical JPS - Low-Level Refinement")
        
        fine_path = [self.s_start]
        all_visited = []
        
        # For each pair of adjacent coarse cells in the path
        for i in range(len(coarse_path) - 1):
            current_cell = coarse_path[i]
            next_cell = coarse_path[i + 1]
            
            # Get connection points between these cells
            if next_cell in self.coarse_grid[current_cell]['connections']:
                connections = self.coarse_grid[current_cell]['connections'][next_cell]
                if connections:
                    # Use the first connection (could be improved by selecting best one)
                    start_jp, end_jp = connections[0]
                    
                    # If we're not at the start cell, connect to the start jump point
                    if i > 0 and fine_path[-1] != start_jp:
                        connect_path, connect_visited = self.jps_local_search(fine_path[-1], start_jp)
                        if connect_path:
                            # Avoid duplicating the start point
                            fine_path.extend(connect_path[1:])
                            all_visited.extend(connect_visited)
                            
                            # Visualize this connection
                            self.plot_util.plot_path(connect_path, 'orange')
                            plt.pause(0.1)
                    
                    # Connect between cells
                    segment_path, segment_visited = self.jps_local_search(start_jp, end_jp)
                    if segment_path:
                        # Avoid duplicating points
                        if fine_path[-1] == segment_path[0]:
                            fine_path.extend(segment_path[1:])
                        else:
                            fine_path.extend(segment_path)
                            
                        all_visited.extend(segment_visited)
                        
                        # Visualize this segment
                        self.plot_util.plot_path(segment_path, 'orange')
                        plt.pause(0.1)
            else:
                # No direct connection found, try to connect any points from current to next cell
                print(f"No direct connection between {current_cell} and {next_cell}, trying to find a path...")
                
                # Sample points from both cells
                current_fine_points = self.coarse_grid[current_cell]['fine_coords']
                next_fine_points = self.coarse_grid[next_cell]['fine_coords']
                
                # Try to find a path between sample points
                best_path = None
                best_visited = None
                
                # Use current position as the start point
                start_point = fine_path[-1]
                
                # Sample end points from next cell's jump points or use any traversable point
                end_points = list(self.coarse_grid[next_cell]['jump_points'])
                if not end_points and next_fine_points:
                    # If no jump points, use up to 5 sample points
                    sample_size = min(5, len(next_fine_points))
                    end_points = next_fine_points[:sample_size]
                
                # Try to find the best path
                for end_point in end_points:
                    path, visited = self.jps_local_search(start_point, end_point)
                    if path and (best_path is None or len(path) < len(best_path)):
                        best_path = path
                        best_visited = visited
                
                if best_path:
                    # Avoid duplicating the start point
                    if fine_path[-1] == best_path[0]:
                        fine_path.extend(best_path[1:])
                    else:
                        fine_path.extend(best_path)
                        
                    all_visited.extend(best_visited)
                    
                    # Visualize this path
                    self.plot_util.plot_path(best_path, 'orange')
                    plt.pause(0.1)
                else:
                    print(f"Failed to find path between {current_cell} and {next_cell}")
        
        # Connect to goal if needed
        if fine_path[-1] != self.s_goal:
            goal_path, goal_visited = self.jps_local_search(fine_path[-1], self.s_goal)
            if goal_path:
                # Avoid duplicating points
                if fine_path[-1] == goal_path[0]:
                    fine_path.extend(goal_path[1:])
                else:
                    fine_path.extend(goal_path)
                    
                all_visited.extend(goal_visited)
                
                # Visualize path to goal
                self.plot_util.plot_path(goal_path, 'orange')
                plt.pause(0.1)
        
        # Post-process for a smoother path
        fine_path = self.post_process_path(fine_path)
        
        return fine_path, all_visited

    def post_process_path(self, path):
        """
        Post-process the path to remove redundant waypoints and make it smoother
        :param path: input path
        :return: smoothed path
        """
        if not path or len(path) < 3:
            return path
            
        # Path smoothing by removing redundant waypoints
        i = 0
        processed_path = [path[0]]
        
        while i < len(path) - 1:
            current = path[i]
            found_jump = False
            
            # Look ahead to find direct connections
            for j in range(len(path) - 1, i, -1):
                if not self.is_collision(current, path[j]):
                    processed_path.append(path[j])
                    i = j
                    found_jump = True
                    break
            
            # If no jump found, take next point
            if not found_jump:
                i += 1
                if i < len(path):
                    processed_path.append(path[i])
                    
        # Ensure goal is included
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

    def searching(self):
        """
        Main hierarchical JPS search function
        :return: path, visited cells
        """
        # Phase 1: Precompute the hierarchical graph
        start_time_precompute = time.time()
        self.precompute_hierarchical_graph()
        end_time_precompute = time.time()
        print(f"Precomputation time: {end_time_precompute - start_time_precompute:.4f} seconds")
        
        # Phase 2: High-level search on coarse grid
        start_time_high = time.time()
        coarse_path, visited_coarse = self.high_level_search()
        end_time_high = time.time()
        print(f"High-level search time: {end_time_high - start_time_high:.4f} seconds")
        
        if not coarse_path:
            print("No coarse path found, searching failed.")
            return [], []
        
        # Phase 3: Low-level refinement using JPS
        start_time_low = time.time()
        fine_path, visited_fine = self.low_level_refine(coarse_path)
        end_time_low = time.time()
        print(f"Low-level refinement time: {end_time_low - start_time_low:.4f} seconds")
        
        # Final visualization
        if fine_path:
            plt.title("Hierarchical JPS - Final Path")
            self.plot_util.plot_path(fine_path)
            
            # Create legend
            start_marker = plt.Line2D([], [], marker='s', color='b', label='Start Point', markerfacecolor='b', markersize=8)
            goal_marker = plt.Line2D([], [], marker='s', color='g', label='Goal Point', markerfacecolor='g', markersize=8)
            jp_marker = plt.Line2D([], [], marker='o', color='m', label='Jump Point', markerfacecolor='m', markersize=5)
            coarse_marker = plt.Line2D([], [], marker='o', color='b', label='Coarse Path', markerfacecolor='b', markersize=7)
            visited_marker = plt.Line2D([], [], marker='o', color='y', label='Visited Cell', markerfacecolor='y', markersize=6)
            path_line = plt.Line2D([], [], color='r', label='Final Path', linewidth=2)
            
            plt.legend(handles=[start_marker, goal_marker, jp_marker, coarse_marker, visited_marker, path_line], loc='best')
            
            total_time = end_time_precompute - start_time_precompute + end_time_high - start_time_high + end_time_low - start_time_low
            print(f"Total search time: {total_time:.4f} seconds")
            print(f"Path found with {len(fine_path)} nodes.")
            
        else:
            print("Failed to find a complete path.")
        
        plt.pause(0.1)
        return fine_path, visited_fine


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    heuristic_type = "euclidean"

    hierarchical_jps = HierarchicalJPS(s_start, s_goal, heuristic_type)
    path, visited = hierarchical_jps.searching()
    
    if path:
        print("Path found!")
        
    plt.show()


if __name__ == '__main__':
    main()
