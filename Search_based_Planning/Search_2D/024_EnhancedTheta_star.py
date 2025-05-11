"""
Enhanced Theta* 2D: Improved any-angle path planning algorithm
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class EnhancedThetaStar:
    """
    Enhanced Theta*: Improved any-angle path planning
    This algorithm extends Theta* with hierarchical pathfinding,
    improved line-of-sight, and adaptive path optimization.
    """
    def __init__(self, s_start, s_goal, heuristic_type, tie_breaking=True, grid_resolution=1):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.tie_breaking = tie_breaking  # Whether to use tie-breaking in f-value
        self.grid_resolution = grid_resolution  # Controls the hierarchical grid resolution

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        # Multi-level pathfinding 
        self.abstract_grid_size = 4  # Size of abstract grid cells (higher level)
        self.abstract_obs = self.create_abstract_grid()  # Create abstract grid for high-level planning
        
        # Standard A* structures
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        
        # Enhanced features
        self.visited_count = dict()  # Count how many times a node was considered (for tie-breaking)
        self.los_cache = dict()  # Cache line-of-sight calculations for efficiency
        
        # For visualization
        self.los_checks = []
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.plot = plotting.Plotting(s_start, s_goal)
        
        # Current search state
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []
        self.high_level_path = []  # Path from abstract grid (high-level planning)
        self.corridors = []  # Corridors for focused search

    def create_abstract_grid(self):
        """
        Create an abstract grid for hierarchical planning
        """
        abstract_obs = set()
        
        # Create abstract grid cells by grouping original grid cells
        for i in range(0, self.Env.x_range, self.abstract_grid_size):
            for j in range(0, self.Env.y_range, self.abstract_grid_size):
                # If any cell in this abstract cell is an obstacle, 
                # mark the abstract cell as an obstacle
                is_obstacle = False
                for di in range(self.abstract_grid_size):
                    for dj in range(self.abstract_grid_size):
                        if (i + di, j + dj) in self.obs:
                            is_obstacle = True
                            break
                    if is_obstacle:
                        break
                
                if is_obstacle:
                    abstract_obs.add((i // self.abstract_grid_size, j // self.abstract_grid_size))
        
        return abstract_obs

    def searching(self):
        """
        Enhanced Theta* path searching with hierarchical planning.
        :return: path, visited order
        """
        # Initialize plot
        self.plot.plot_grid("Enhanced Theta*")
        
        # Step 1: High-level planning on abstract grid
        self.high_level_planning()
        
        # Step 2: Detailed planning on original grid using corridors
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        
        # Initialize visited count for tie-breaking
        self.visited_count[self.s_start] = 1
        
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            self.current_visited.append(s)

            # Update current path and visualization
            if s == self.s_goal:
                self.current_path = self.extract_path(self.PARENT)
            else:
                # Show path from start to current node
                temp_path = self.extract_temp_path(s)
                self.current_path = temp_path
            
            # Update visualization periodically
            if len(self.CLOSED) % 5 == 0 or s == self.s_goal:
                self.update_plot()

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                # Check if this node is within the corridor (if high-level planning done)
                if self.high_level_path and not self.is_in_corridor(s_n):
                    continue  # Skip nodes outside the search corridor

                # Enhanced feature: Cache and reuse line-of-sight calculations
                cache_key = (self.PARENT[s], s_n)
                if cache_key in self.los_cache:
                    los_result = self.los_cache[cache_key]
                else:
                    los_result = self.enhanced_line_of_sight(self.PARENT[s], s_n)
                    self.los_cache[cache_key] = los_result
                
                self.los_checks.append((self.PARENT[s], s_n, los_result))
                self.current_los_checks.append((self.PARENT[s], s_n, los_result))
                
                # Path 2 - Try to use parent of current node (line-of-sight checking)
                if los_result:
                    # Line-of-sight exists, consider path from parent
                    new_cost = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf
                        self.visited_count[s_n] = 0

                    if new_cost < self.g[s_n]:
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = self.PARENT[s]  # Skip a step in the path
                        # Ensure visited_count is initialized for this node
                        if s_n not in self.visited_count:
                            self.visited_count[s_n] = 0
                        self.visited_count[s_n] += 1  # Increment visit count for tie-breaking
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                else:
                    # No line-of-sight, do regular A* update (Path 1)
                    new_cost = self.g[s] + self.cost(s, s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf
                        self.visited_count[s_n] = 0

                    if new_cost < self.g[s_n]:
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = self.PARENT[s] if los_result else s
                        # Ensure visited_count is initialized for this node
                        if s_n not in self.visited_count:
                            self.visited_count[s_n] = 0
                        self.visited_count[s_n] += 1  # Increment visit count for tie-breaking
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # Final update
        self.update_plot(final=True)
        plt.show()
        
        final_path = self.extract_path(self.PARENT)
        # Apply post-processing path optimization
        optimized_path = self.optimize_path(final_path)
        
        return optimized_path, self.CLOSED

    def high_level_planning(self):
        """
        Perform high-level planning on the abstract grid
        """
        # Convert start and goal to abstract grid coordinates
        abstract_start = (self.s_start[0] // self.abstract_grid_size, 
                          self.s_start[1] // self.abstract_grid_size)
        abstract_goal = (self.s_goal[0] // self.abstract_grid_size, 
                         self.s_goal[1] // self.abstract_grid_size)
        
        # A* on abstract grid
        open_set = []
        closed_set = set()
        abstract_g = {abstract_start: 0}
        abstract_parent = {abstract_start: abstract_start}
        
        heapq.heappush(open_set, (self.heuristic_abstract(abstract_start, abstract_goal), abstract_start))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == abstract_goal:
                break
                
            closed_set.add(current)
            
            # Get neighbors in abstract grid
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if (0 <= neighbor[0] < self.Env.x_range // self.abstract_grid_size and
                        0 <= neighbor[1] < self.Env.y_range // self.abstract_grid_size and
                        neighbor not in self.abstract_obs and
                        neighbor not in closed_set):
                        
                        # Cost in abstract grid - using Euclidean distance
                        cost = abstract_g[current] + math.hypot(dx, dy)
                        
                        if neighbor not in abstract_g or cost < abstract_g[neighbor]:
                            abstract_g[neighbor] = cost
                            abstract_parent[neighbor] = current
                            f_value = cost + self.heuristic_abstract(neighbor, abstract_goal)
                            heapq.heappush(open_set, (f_value, neighbor))
        
        # Extract high-level path
        if abstract_goal in abstract_parent:
            high_level_path = []
            current = abstract_goal
            
            while current != abstract_start:
                # Convert back to original grid coordinates (use center of abstract cell)
                original_coords = (current[0] * self.abstract_grid_size + self.abstract_grid_size // 2,
                                  current[1] * self.abstract_grid_size + self.abstract_grid_size // 2)
                high_level_path.append(original_coords)
                current = abstract_parent[current]
                
            # Add start position
            original_start = (abstract_start[0] * self.abstract_grid_size + self.abstract_grid_size // 2,
                             abstract_start[1] * self.abstract_grid_size + self.abstract_grid_size // 2)
            high_level_path.append(original_start)
            
            # Reverse path to go from start to goal
            high_level_path.reverse()
            
            # Save high level path for visualization and corridor creation
            self.high_level_path = high_level_path
            
            # Create corridors around high level path
            self.create_corridors()

    def create_corridors(self):
        """
        Create corridors around the high-level path for focused search
        """
        corridor_width = self.abstract_grid_size  # Width of corridor on each side
        
        self.corridors = []
        
        if not self.high_level_path:
            return
            
        # For each segment in high level path
        for i in range(len(self.high_level_path) - 1):
            start = self.high_level_path[i]
            end = self.high_level_path[i+1]
            
            # Create a rectangular corridor between start and end
            min_x = min(start[0], end[0]) - corridor_width
            max_x = max(start[0], end[0]) + corridor_width
            min_y = min(start[1], end[1]) - corridor_width
            max_y = max(start[1], end[1]) + corridor_width
            
            # Ensure corridor is within grid bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(self.Env.x_range - 1, max_x)
            max_y = min(self.Env.y_range - 1, max_y)
            
            self.corridors.append((min_x, min_y, max_x, max_y))

    def is_in_corridor(self, node):
        """
        Check if a node is within any corridor
        """
        if not self.corridors:
            return True  # If no corridors defined, all nodes are valid
            
        x, y = node
        
        for min_x, min_y, max_x, max_y in self.corridors:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
                
        return False

    def heuristic_abstract(self, node, goal):
        """
        Heuristic function for abstract grid
        """
        return math.hypot(goal[0] - node[0], goal[1] - node[1])

    def optimize_path(self, path):
        """
        Post-processing path optimization
        1. Remove redundant nodes where three consecutive nodes are collinear
        2. Try to connect non-adjacent nodes if line-of-sight exists
        """
        if len(path) <= 2:
            return path
            
        # Step 1: Remove redundant collinear nodes
        optimized_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Check if current node is collinear with previous and next
            p1 = optimized_path[-1]
            p2 = path[i]
            p3 = path[i + 1]
            
            # Cross product near zero means collinear
            cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            # Keep node if not collinear or if bend is significant
            if abs(cross_product) > 0.001:
                optimized_path.append(p2)
        
        # Add final node
        optimized_path.append(path[-1])
        
        # Step 2: Path smoothing - connect non-adjacent nodes if possible
        i = 0
        final_path = [optimized_path[0]]
        
        while i < len(optimized_path) - 1:
            current = optimized_path[i]
            
            # Try to connect to furthest possible node
            furthest = i
            for j in range(len(optimized_path) - 1, i, -1):
                if self.enhanced_line_of_sight(current, optimized_path[j]):
                    furthest = j
                    break
            
            # Add the furthest node with line-of-sight
            final_path.append(optimized_path[furthest])
            i = furthest
        
        return final_path

    def enhanced_line_of_sight(self, s_start, s_end):
        """
        Enhanced line-of-sight check with better precision
        :param s_start: start node
        :param s_end: end node
        :return: True if line-of-sight exists
        """
        # First do a quick check
        if self.is_collision(s_start, s_end):
            return False
            
        # Enhanced: Check additional points along the line for more precision
        dx = s_end[0] - s_start[0]
        dy = s_end[1] - s_start[1]
        distance = math.hypot(dx, dy)
        
        # For short distances, basic check is enough
        if distance < 3:
            return True
            
        # For longer distances, check additional points
        steps = min(10, int(distance))  # Adjust precision based on distance
        
        for i in range(1, steps):
            t = i / steps
            check_x = int(s_start[0] + dx * t)
            check_y = int(s_start[1] + dy * t)
            
            if (check_x, check_y) in self.obs:
                return False
        
        return True

    def update_plot(self, final=False):
        """
        Update the plot to show current search state
        """
        # Clear current figure
        plt.cla()
        
        # Draw grid using plotting.py methods
        self.plot.plot_grid("Enhanced Theta*")
        
        # Draw corridors if available
        if self.corridors:
            for min_x, min_y, max_x, max_y in self.corridors:
                width = max_x - min_x
                height = max_y - min_y
                rect = Rectangle((min_x, min_y), width, height, 
                                 fill=False, edgecolor='blue', linestyle='--', alpha=0.3)
                plt.gca().add_patch(rect)
        
        # Draw high-level path if available
        if self.high_level_path:
            xs = [node[0] for node in self.high_level_path]
            ys = [node[1] for node in self.high_level_path]
            plt.plot(xs, ys, 'b--', linewidth=2, alpha=0.5)
        
        # Draw visited nodes
        if self.current_visited:
            for node in self.current_visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], color='gray', marker='o')
        
        # Draw line-of-sight checks
        if self.current_los_checks:
            for start, end, result in self.current_los_checks:
                color = 'g' if result else 'r'
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.3)
        
        # Draw current path
        if self.current_path:
            self.plot.plot_path(self.current_path)
        
        # Update figure
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        
        # Pause longer for final result
        if final:
            plt.pause(0.5)
        else:
            plt.pause(0.01)
    
    def extract_temp_path(self, current):
        """
        Extract temporary path from start to current node
        """
        path = [current]
        s = current
        
        while s != self.s_start:
            s = self.PARENT[s]
            path.append(s)
        
        return list(reversed(path))

    def get_neighbor(self, s):
        """
        Find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        nei_list = []
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        """
        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # Check if points are within grid boundaries
        x_range, y_range = self.Env.x_range, self.Env.y_range
        
        # Check boundary constraints
        if (s_start[0] < 0 or s_start[0] >= x_range or 
            s_start[1] < 0 or s_start[1] >= y_range or
            s_end[0] < 0 or s_end[0] >= x_range or
            s_end[1] < 0 or s_end[1] >= y_range):
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        # Basic check for diagonal segments
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        # Bresenham's line algorithm for more thorough checking
        x0, y0 = s_start
        x1, y1 = s_end
        
        # If the line is steep, transpose the grid
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        # Swap points if needed to ensure x increases
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        y = y0
        
        # Determine step direction
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # Check each point on the line
        for x in range(x0, x1 + 1):
            if steep:
                # If steep, the coordinates are transposed
                if (y, x) in self.obs:
                    return True
            else:
                if (x, y) in self.obs:
                    return True
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        
        return False

    def line_of_sight(self, s_start, s_end):
        """
        Standard line-of-sight check
        :param s_start: start node
        :param s_end: end node
        :return: True if line-of-sight exists
        """
        return not self.is_collision(s_start, s_end)

    def f_value(self, s):
        """
        f = g + h with optional tie-breaking. 
        :param s: current state
        :return: f
        """
        f = self.g[s] + self.heuristic(s)
        
        # Enhanced tie-breaking by using path length and visit count
        if self.tie_breaking:
            # Small increment to prefer shorter paths when f-values are equal
            # Also consider visit count to break ties further 
            # (older nodes get higher priority, promoting depth-first behavior)
            tie_breaker = 0.001 * (1.0 / (self.visited_count[s] + 1))
            f = f + tie_breaker
            
        return f

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(reversed(path))

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """
        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    """
    Enhanced Theta*: Improved Any-Angle Path Planning
    
    Enhanced Theta* improves upon the original Theta* algorithm with several key features:
    
    1. Hierarchical Planning:
       - Uses an abstract grid for high-level planning
       - Creates search corridors for focused exploration
    
    2. Improved Line-of-Sight:
       - More precise line-of-sight calculations
       - Caching of line-of-sight results for efficiency
    
    3. Advanced Tie-Breaking:
       - Better handling of equal f-values
       - Promoting deeper exploration where beneficial
    
    4. Path Optimization:
       - Post-processing to remove redundant nodes
       - Path smoothing for more natural routes
    
    These enhancements result in faster pathfinding, smoother paths, and better
    adaptability to different environments.
    """
    s_start = (5, 5)
    s_goal = (45, 25)
    
    # Create Enhanced Theta* instance with euclidean heuristic
    enhanced_theta_star = EnhancedThetaStar(s_start, s_goal, "euclidean", tie_breaking=True)
    path, visited = enhanced_theta_star.searching()


if __name__ == '__main__':
    main()
