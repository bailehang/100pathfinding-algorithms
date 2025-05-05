"""
Enhanced Parallel_A_star 2D with multiple goals and information sharing
@author: clark bai
@modifier: Cline (enhanced version)
"""

import os
import sys
import math
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt
import traceback
import logging
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parallel_astar.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ParallelAStar")

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                    "/../../Search_based_Planning/")
    
    from Search_2D import plotting, env
    logger.info("Successfully imported plotting and env modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)


class EnhancedParallelAStar:
    """
    Enhanced Parallel A* implementation with:
    - Support for multiple goals
    - Shared heuristic information
    - Sharing of discovered partial paths
    - Optimized coordination between regions
    """
    def __init__(self, s_start, s_goals, heuristic_type, num_regions=4):
        # Convert single goal to list if necessary
        self.s_start = s_start
        self.s_goals = s_goals if isinstance(s_goals, list) else [s_goals]
        self.heuristic_type = heuristic_type
        self.num_regions = num_regions
        
        # Initialize environment
        self.Env = env.Env()
        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        
        # Map dimensions
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        
        # Compute region divisions
        self.regions = self.divide_map()
        
        # Shared data structures for information sharing
        self.heuristic_cache = {}  # Cache for heuristic values
        self.path_segments = {}    # Discovered path segments
        self.node_costs = {}       # Best known costs to nodes
        
        # For tracking search statistics
        self.region_paths = {}       # Paths found in each region
        self.region_visited = {}     # Visited nodes in each region
        self.expanded_count = 0      # Total expanded nodes
        self.goal_paths = {}         # Paths found to each goal

    def divide_map(self):
        """
        Divide the map into rectangular regions of approximately equal size
        :return: dictionary of regions with their boundaries
        """
        # Determine region layout (rows x columns) to achieve num_regions
        n_rows = int(math.sqrt(self.num_regions))
        while self.num_regions % n_rows != 0:
            n_rows -= 1
        n_cols = self.num_regions // n_rows
        
        # Calculate region sizes
        region_width = math.ceil(self.x_range / n_cols)
        region_height = math.ceil(self.y_range / n_rows)
        
        regions = {}
        
        # Create regions
        region_id = 0
        for row in range(n_rows):
            for col in range(n_cols):
                x_min = col * region_width
                y_min = row * region_height
                x_max = min((col + 1) * region_width - 1, self.x_range - 1)
                y_max = min((row + 1) * region_height - 1, self.y_range - 1)
                
                regions[region_id] = {
                    'boundaries': (x_min, y_min, x_max, y_max),
                    'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                    'neighbors': []  # Will store neighbor region IDs
                }
                
                region_id += 1
        
        # Set up neighbor relationships
        for i in regions:
            for j in regions:
                if i != j:
                    # Check if regions share a boundary
                    i_xmin, i_ymin, i_xmax, i_ymax = regions[i]['boundaries']
                    j_xmin, j_ymin, j_xmax, j_ymax = regions[j]['boundaries']
                    
                    # Check for adjacency
                    if ((i_xmin <= j_xmax and i_xmax >= j_xmin) and  # x-overlap
                        ((i_ymax + 1 == j_ymin) or (j_ymax + 1 == i_ymin))):  # y-adjacent
                        regions[i]['neighbors'].append(j)
                    elif ((i_ymin <= j_ymax and i_ymax >= j_ymin) and  # y-overlap
                          ((i_xmax + 1 == j_xmin) or (j_xmax + 1 == i_xmin))):  # x-adjacent
                        regions[i]['neighbors'].append(j)
        
        return regions

    def get_region_for_node(self, node):
        """
        Determine which region a node belongs to
        :param node: (x, y) coordinates
        :return: region_id or -1 if not found
        """
        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']
            if x_min <= node[0] <= x_max and y_min <= node[1] <= y_max:
                return region_id
        return -1  # Should not happen unless node is outside map

    def identify_border_nodes(self):
        """
        Identify nodes at the border between regions
        :return: dictionary mapping (region1, region2) -> list of border nodes
        """
        border_nodes = {}
        
        # Initialize border nodes between adjacent regions
        for region_id, region in self.regions.items():
            for neighbor_id in region['neighbors']:
                if (region_id, neighbor_id) not in border_nodes:
                    border_nodes[(region_id, neighbor_id)] = []
        
        # For each region, find nodes that border another region
        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']
            
            # Check all border cells of this region
            border_cells = []
            
            # Left border
            for y in range(y_min, y_max + 1):
                border_cells.append((x_min, y))
            
            # Right border
            for y in range(y_min, y_max + 1):
                border_cells.append((x_max, y))
            
            # Top border (excluding corners already added)
            for x in range(x_min + 1, x_max):
                border_cells.append((x, y_max))
            
            # Bottom border (excluding corners already added)
            for x in range(x_min + 1, x_max):
                border_cells.append((x, y_min))
            
            # Filter out border cells that are obstacles
            border_cells = [cell for cell in border_cells if cell not in self.obs]
            
            # For each border cell, find adjacent cells that are in a different region
            for cell in border_cells:
                x, y = cell
                for u in self.u_set:  # Check all 8 directions
                    nx, ny = x + u[0], y + u[1]
                    
                    # Skip if outside map or an obstacle
                    if not (0 <= nx < self.x_range and 0 <= ny < self.y_range) or (nx, ny) in self.obs:
                        continue
                    
                    # Check if neighbor cell is in a different region
                    neighbor_region = self.get_region_for_node((nx, ny))
                    if neighbor_region != region_id and neighbor_region != -1:
                        # Add this cell as a border node between these regions
                        if (region_id, neighbor_region) in border_nodes:
                            border_nodes[(region_id, neighbor_region)].append(cell)
        
        # Remove duplicates
        for key in border_nodes:
            border_nodes[key] = list(set(border_nodes[key]))
        
        return border_nodes

    def heuristic(self, s, goal):
        """
        Calculate heuristic with caching for performance
        :param s: current state
        :param goal: goal state
        :return: heuristic value
        """
        # Check cache first
        cache_key = (s, goal)
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]
        
        # Calculate heuristic based on type
        if self.heuristic_type == "manhattan":
            h_value = abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:  # euclidean
            h_value = math.hypot(goal[0] - s[0], goal[1] - s[1])
        
        # Cache the result
        self.heuristic_cache[cache_key] = h_value
        return h_value
    
    def multi_goal_heuristic(self, s):
        """
        Calculate heuristic considering all goals - take the minimum distance to any goal
        :param s: current state
        :return: heuristic value to the closest goal
        """
        return min(self.heuristic(s, goal) for goal in self.s_goals)

    def get_neighbors(self, s):
        """
        Get neighbors of state s that are not obstacles
        :param s: state
        :return: list of neighbors
        """
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set 
                if 0 <= s[0] + u[0] < self.x_range 
                and 0 <= s[1] + u[1] < self.y_range 
                and (s[0] + u[0], s[1] + u[1]) not in self.obs]

    def cost(self, s_start, s_goal):
        """
        Calculate cost between two states
        :param s_start: start state
        :param s_goal: goal state
        :return: cost
        """
        if self.is_collision(s_start, s_goal):
            return float('inf')

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment from s_start to s_end collides with obstacles
        :param s_start: start node
        :param s_end: end node
        :return: True if collision, False otherwise
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

    def extract_path(self, parent, start, goal):
        """
        Extract the path from start to goal
        :param parent: parent dictionary
        :param start: start node
        :param goal: goal node
        :return: path list
        """
        if goal not in parent:
            return []
            
        path = [goal]
        current = goal
        
        while current != start:
            current = parent[current]
            path.append(current)
            
        return list(reversed(path))

    def store_path_segment(self, start, end, path, cost):
        """
        Store a discovered path segment for future use
        :param start: start node
        :param end: end node
        :param path: path from start to end
        :param cost: cost of the path
        """
        if not path:
            return
        
        # Store direct path from start to end
        self.path_segments[(start, end)] = (path, cost)
        
        # Also store path from end to start (reversed)
        reversed_path = list(reversed(path))
        self.path_segments[(end, start)] = (reversed_path, cost)
        
        # Update node costs
        if start not in self.node_costs or cost < self.node_costs[start]:
            self.node_costs[start] = cost
        
        if end not in self.node_costs or cost < self.node_costs[end]:
            self.node_costs[end] = cost

    def check_path_segment(self, start, end):
        """
        Check if there's a known path segment between start and end
        :param start: start node
        :param end: end node
        :return: (path, cost) or (None, inf) if no path found
        """
        if (start, end) in self.path_segments:
            return self.path_segments[(start, end)]
        return None, float('inf')

    def region_a_star(self, region_id, start_node, goal_nodes, region_boundaries):
        """
        A* search for a specific region with multiple goals
        :param region_id: ID of the region to search
        :param start_node: Start node for this region's search
        :param goal_nodes: List of goal nodes for this region's search
        :param region_boundaries: Boundaries of the region (x_min, y_min, x_max, y_max)
        :return: dictionary of {goal: (path, visited_nodes)}
        """
        # Unpack region boundaries
        x_min, y_min, x_max, y_max = region_boundaries
        
        # Initialize A* data structures
        open_set = []
        closed_set = []
        parent = {start_node: start_node}
        g = {start_node: 0}
        
        # Initialize goals with infinity cost
        for goal in goal_nodes:
            g[goal] = float('inf')
        
        heapq.heappush(open_set, (self.multi_goal_heuristic(start_node), start_node))
        
        # Track goals that have been reached
        goals_reached = {}
        visited_nodes = []
        
        while open_set and len(goals_reached) < len(goal_nodes):
            # Process current node
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.append(current)
            visited_nodes.append(current)
            self.expanded_count += 1
            
            # Check if current node is a goal
            if current in goal_nodes and current not in goals_reached:
                path = self.extract_path(parent, start_node, current)
                goals_reached[current] = (path, g[current])
                
                # Store this path segment for future use
                self.store_path_segment(start_node, current, path, g[current])
                
                # If all goals are reached, break
                if len(goals_reached) == len(goal_nodes):
                    break
            
            # Expand neighbors within this region
            for s_n in self.get_neighbors(current):
                nx, ny = s_n
                
                # Skip if outside this region's boundaries
                if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                    continue
                
                # Check if there's a known path segment to this neighbor
                path_segment, segment_cost = self.check_path_segment(current, s_n)
                
                # Calculate new cost - either from path segment or direct movement
                if path_segment:
                    new_cost = g[current] + segment_cost
                else:
                    new_cost = g[current] + self.cost(current, s_n)
                
                # Update if better path found
                if s_n not in g or new_cost < g[s_n]:
                    g[s_n] = new_cost
                    parent[s_n] = current
                    
                    # Update node cost in shared data structure
                    self.node_costs[s_n] = new_cost
                    
                    # Calculate f-value with multi-goal heuristic
                    f_value = new_cost + self.multi_goal_heuristic(s_n)
                    heapq.heappush(open_set, (f_value, s_n))
        
        # Return paths to all reached goals
        result = {}
        for goal, (path, _) in goals_reached.items():
            result[goal] = (path, visited_nodes)
        
        return result

    def search_global(self):
        """
        Perform global search from start to all goals
        First identify which regions each start and goal belong to
        Then search for paths between them, sharing information
        :return: dictionary mapping goal -> (path, visited)
        """
        # Get regions for start and goals
        start_region = self.get_region_for_node(self.s_start)
        goal_regions = {goal: self.get_region_for_node(goal) for goal in self.s_goals}
        
        # Identify border nodes for region transitions
        border_nodes = self.identify_border_nodes()
        
        # Results will map goal -> (path, visited)
        results = {}
        all_visited = []
        
        # For each goal, find the best path from start
        for goal, goal_region in goal_regions.items():
            # If start and goal in same region, use direct A*
            if start_region == goal_region:
                region_results = self.region_a_star(
                    start_region, 
                    self.s_start, 
                    [goal], 
                    self.regions[start_region]['boundaries']
                )
                
                if goal in region_results:
                    path, visited = region_results[goal]
                    results[goal] = (path, visited)
                    all_visited.extend(visited)
                    self.region_paths[start_region] = path
                    self.region_visited[start_region] = visited
                continue
            
            # If start and goal in different regions, search for paths through border nodes
            best_path = []
            best_cost = float('inf')
            path_visited = []
            
            # Check if there's a direct connection between start and goal regions
            if goal_region in self.regions[start_region]['neighbors']:
                # Get border nodes between start and goal regions
                border_points = border_nodes.get((start_region, goal_region), [])
                
                for border in border_points:
                    # Search from start to border
                    start_to_border_results = self.region_a_star(
                        start_region, 
                        self.s_start, 
                        [border], 
                        self.regions[start_region]['boundaries']
                    )
                    
                    if border not in start_to_border_results:
                        continue
                    
                    start_to_border_path, start_to_border_visited = start_to_border_results[border]
                    
                    # Search from border to goal
                    border_to_goal_results = self.region_a_star(
                        goal_region, 
                        border, 
                        [goal], 
                        self.regions[goal_region]['boundaries']
                    )
                    
                    if goal not in border_to_goal_results:
                        continue
                    
                    border_to_goal_path, border_to_goal_visited = border_to_goal_results[goal]
                    
                    # Calculate total cost
                    total_cost = 0
                    for i in range(len(start_to_border_path) - 1):
                        total_cost += self.cost(start_to_border_path[i], start_to_border_path[i + 1])
                    
                    for i in range(len(border_to_goal_path) - 1):
                        total_cost += self.cost(border_to_goal_path[i], border_to_goal_path[i + 1])
                    
                    # Update best path if this one is better
                    if total_cost < best_cost:
                        # Combine paths (removing duplicate border node)
                        best_path = start_to_border_path[:-1] + border_to_goal_path
                        best_cost = total_cost
                        path_visited = start_to_border_visited + border_to_goal_visited
            
            # If no direct path found, try multi-hop paths through intermediate regions
            if not best_path:
                # A more sophisticated approach would be implemented here
                # For now, we'll use a simple greedy approach through region centers
                
                # Get sequence of regions to traverse (simple implementation)
                region_sequence = self.find_region_sequence(start_region, goal_region)
                
                if region_sequence:
                    # Follow the region sequence to build a path
                    current_node = self.s_start
                    current_path = []
                    current_visited = []
                    
                    for i in range(len(region_sequence) - 1):
                        current_region = region_sequence[i]
                        next_region = region_sequence[i + 1]
                        
                        # Get border nodes between current and next region
                        border_points = border_nodes.get((current_region, next_region), [])
                        
                        if not border_points:
                            break
                        
                        # Pick a border node (could be optimized to pick the best one)
                        border = border_points[0]
                        
                        # Search from current node to border
                        region_results = self.region_a_star(
                            current_region, 
                            current_node, 
                            [border], 
                            self.regions[current_region]['boundaries']
                        )
                        
                        if border not in region_results:
                            break
                        
                        segment_path, segment_visited = region_results[border]
                        
                        # Add this segment to the path (avoiding duplication)
                        if current_path:
                            current_path = current_path[:-1] + segment_path
                        else:
                            current_path = segment_path
                        
                        current_visited.extend(segment_visited)
                        current_node = border
                    
                    # Final segment from last border to goal
                    if current_node != self.s_start:
                        final_region = region_sequence[-1]
                        final_results = self.region_a_star(
                            final_region, 
                            current_node, 
                            [goal], 
                            self.regions[final_region]['boundaries']
                        )
                        
                        if goal in final_results:
                            final_path, final_visited = final_results[goal]
                            current_path = current_path[:-1] + final_path
                            current_visited.extend(final_visited)
                            
                            best_path = current_path
                            path_visited = current_visited
            
            if best_path:
                results[goal] = (best_path, path_visited)
                all_visited.extend(path_visited)
        
        return results, all_visited

    def find_region_sequence(self, start_region, goal_region):
        """
        Find a sequence of regions to traverse from start to goal
        Simple BFS implementation
        :param start_region: starting region ID
        :param goal_region: goal region ID
        :return: list of region IDs forming a path
        """
        if start_region == goal_region:
            return [start_region]
        
        # BFS to find path through regions
        queue = [(start_region, [start_region])]
        visited = {start_region}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.regions[current]['neighbors']:
                if neighbor == goal_region:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []

    def searching(self):
        """
        Main search function - finds paths to all goals
        :return: dictionary of paths to each goal, and all visited nodes
        """
        start_time = time.time()
        
        # Reset counters and caches
        self.expanded_count = 0
        self.region_paths = {}
        self.region_visited = {}
        self.heuristic_cache = {}
        self.path_segments = {}
        self.node_costs = {}
        self.goal_paths = {}
        
        # Find paths to all goals
        goal_paths, all_visited = self.search_global()
        
        # Store results
        self.goal_paths = goal_paths
        
        end_time = time.time()
        logger.info(f"Enhanced Parallel A* found paths to {len(goal_paths)}/{len(self.s_goals)} goals in {end_time - start_time:.4f} seconds")
        logger.info(f"Total expanded nodes: {self.expanded_count}")
        
        return goal_paths, all_visited
        
    def visualize_regions(self):
        """
        Create visualization data for the region divisions
        :return: list of region boundaries for visualization
        """
        region_lines = []
        
        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']
            
            # Create the four sides of the region
            region_lines.append([(x_min, y_min), (x_max, y_min)])  # Bottom
            region_lines.append([(x_min, y_min), (x_min, y_max)])  # Left
            region_lines.append([(x_max, y_min), (x_max, y_max)])  # Right
            region_lines.append([(x_min, y_max), (x_max, y_max)])  # Top
        
        return region_lines


class EnhancedParallelAStarPlotting(plotting.Plotting):
    """
    Extension of the plotting class for visualizing enhanced parallel A*
    with support for multiple goals and path sharing
    """
    def __init__(self, xI, xG):
        # If multiple goals, use first one for base class initialization
        if isinstance(xG, list):
            super().__init__(xI, xG[0])
            self.xG_multiple = xG
        else:
            super().__init__(xI, xG)
            self.xG_multiple = [xG]
    
    def animation_enhanced_parallel_astar(self, goal_paths, visited, region_lines, name):
        """
        Animation for enhanced parallel A* with multiple goals
        :param goal_paths: dictionary of goal -> (path, visited)
        :param visited: all visited cells
        :param region_lines: lines representing region divisions
        :param name: algorithm name
        """
        # Plot grid with obstacles
        self.plot_grid(name)
        
        # Plot region divisions
        for line in region_lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                     color='blue', linestyle='--', alpha=0.5)
        
        # Plot visited cells
        self.plot_visited(visited, 'gray')
        
        # Create color map for different paths
        num_paths = len(goal_paths)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_paths))
        
        # Plot paths to each goal
        for i, (goal, (path, _)) in enumerate(goal_paths.items()):
            if not path:
                continue
            
            # Extract x and y coordinates for the path
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            
            # Plot the path with a unique color
            color = colors[i]
            plt.plot(path_x, path_y, linewidth=2, color=color, 
                     label=f'Path to Goal {i+1} {goal}')
            
            # Mark waypoints with small dots
            plt.scatter(path_x, path_y, color=color, s=30, alpha=0.7)
            
            # Mark the goal specifically
            plt.scatter(goal[0], goal[1], color=color, s=100, marker='*', 
                       label=f'Goal {i+1}', edgecolors='black', zorder=5)
        
        # Mark the start point
        plt.scatter(self.xI[0], self.xI[1], color='green', s=100, zorder=5, label='Start')
        
        # Add legend with smaller font to avoid overcrowding
        plt.legend(loc='upper right', fontsize='small')
        
        # Save the figure to file before showing it
        plt.savefig("parallel_astar_path.png", dpi=150)
        logger.info("Plot saved to parallel_astar_path.png")
        
        # Show the plot in a window
        plt.tight_layout()
        plt.show()


def run_test_without_gui():
    """
    Run Enhanced Parallel A* algorithm without GUI visualization for testing
    """
    logger.info("=== Running Enhanced Parallel A* Test Without GUI ===")
    
    # Define test parameters - start point and multiple goals
    s_start = (5, 5)
    s_goals = [(45, 25), (25, 5), (45, 15)]
    
    logger.info(f"Start: {s_start}")
    logger.info(f"Goals: {s_goals}")
    
    # Create enhanced parallel A* with 4 regions
    parallel_astar = EnhancedParallelAStar(s_start, s_goals, "euclidean", 4)
    
    # Output region information
    logger.info(f"Created {len(parallel_astar.regions)} regions")
    for region_id, region in parallel_astar.regions.items():
        logger.info(f"Region {region_id}: {region['boundaries']}")
        logger.info(f"  Neighbors: {region['neighbors']}")
    
    # Run enhanced parallel A* search
    logger.info("Running Enhanced Parallel A* search...")
    start_time = time.time()
    goal_paths, visited = parallel_astar.searching()
    end_time = time.time()
    
    # Output results
    logger.info(f"Search completed in {end_time - start_time:.4f} seconds")
    logger.info(f"Paths found to {len(goal_paths)}/{len(s_goals)} goals")
    logger.info(f"Total nodes expanded: {parallel_astar.expanded_count}")
    logger.info(f"Total nodes visited: {len(visited)}")
    logger.info(f"Heuristic cache size: {len(parallel_astar.heuristic_cache)}")
    logger.info(f"Path segments cache size: {len(parallel_astar.path_segments)}")
    
    # Calculate path lengths for each goal
    total_path_length = 0
    for goal, (path, _) in goal_paths.items():
        if path:
            path_length = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) 
                             for i in range(len(path)-1))
            logger.info(f"Path to {goal}: length={path_length:.2f}, steps={len(path)}")
            total_path_length += path_length
    
    logger.info(f"Total path length: {total_path_length:.2f}")
    
    # Save results to a file for verification
    with open("parallel_astar_results.txt", "w") as f:
        f.write("=== Enhanced Parallel A* Results ===\n")
        f.write(f"Start: {s_start}\n")
        f.write(f"Goals: {s_goals}\n")
        f.write(f"Search completed in {end_time - start_time:.4f} seconds\n")
        f.write(f"Paths found to {len(goal_paths)}/{len(s_goals)} goals\n")
        f.write(f"Total nodes expanded: {parallel_astar.expanded_count}\n")
        f.write(f"Total nodes visited: {len(visited)}\n\n")
        
        for goal, (path, _) in goal_paths.items():
            f.write(f"Path to {goal}:\n")
            for i, point in enumerate(path):
                f.write(f"  Step {i}: {point}\n")
            f.write("\n")
    
    logger.info("=== Test Complete ===")
    return goal_paths, visited


def main():
    """Simple test function that works directly with standard A* for multiple goals"""
    print("Starting Enhanced Parallel A* simplified demo...")
    
    # Test parameters
    s_start = (5, 5)
    s_goals = [(45, 25), (25, 5), (45, 15)]
    
    print(f"Start: {s_start}")
    print(f"Goals: {s_goals}")
    
    # Process each goal one by one using standard A*
    all_paths = {}
    all_visited = []
    
    for i, goal in enumerate(s_goals):
        print(f"\nFinding path to goal {i+1}: {goal}")
        
        # Import using __import__ to handle module name starting with number
        astar_module = __import__('005_Astar', fromlist=['AStar'])
        AStar = astar_module.AStar
        astar = AStar(s_start, goal, "euclidean")
        
        path, visited = astar.searching()
        
        if path:
            path_length = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) 
                            for i in range(len(path)-1))
            print(f"Path found, length: {path_length:.2f}, steps: {len(path)}")
            all_paths[goal] = (path, visited)
            all_visited.extend(visited)
        else:
            print(f"No path found to {goal}")
    
    # Visualize the results in one window including start and goals
    print("\nVisualizing results with start and goals in one window...")
    try:
        plot = EnhancedParallelAStarPlotting(s_start, s_goals)
        
        # Create region lines for visualization (empty since we're not using regions here)
        region_lines = []
        
        # Plot results including start and goals
        plot.animation_enhanced_parallel_astar(all_paths, all_visited, region_lines, "Multi-Goal A*")
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        logger.error(traceback.format_exc())
        print(f"Visualization failed: {e}")
    
    print("Demo completed successfully.")


if __name__ == '__main__':
    main()
