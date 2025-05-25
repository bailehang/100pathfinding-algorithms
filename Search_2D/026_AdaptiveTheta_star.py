"""
Adaptive Theta* 2D: Self-adapting any-angle path planning algorithm
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class AdaptiveThetaStar:
    """
    Adaptive Theta*: Self-adapting any-angle path planning algorithm
    
    This algorithm extends Theta* by dynamically adapting its behavior based on:
    1. Environment complexity (obstacle density)
    2. Computational resources (time constraints)
    3. Path quality requirements (optimality vs. speed tradeoff)
    """
    def __init__(self, s_start, s_goal, heuristic_type="euclidean", 
                 time_limit=5.0, adaptive_modes=True):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.time_limit = time_limit  # Time limit in seconds
        self.adaptive_modes = adaptive_modes  # Whether to use adaptive features

        # Environment setup
        self.Env = env.Env()
        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        # Standard A* structures
        self.OPEN = []
        self.CLOSED = []
        self.PARENT = dict()
        self.g = dict()
        self.f = dict()  # Store f-values for analysis
        
        # Adaptive components
        self.heuristic_weight = 1.0  # Initial heuristic weight (changes adaptively)
        self.los_precision = dict()  # Line-of-sight precision per region
        self.complexity_map = self.analyze_environment()  # Complexity metrics per region
        self.los_cache = dict()  # Cache line-of-sight results
        self.los_checks = 0  # Count line-of-sight checks
        self.los_computation_time = 0.0  # Time spent on LOS checks
        
        # Dynamic timeout handling
        self.start_time = None
        self.node_expansion_rate = []  # Track node expansion rate for ETA
        self.last_expansion_check = 0
        self.predicted_expansions = 0
        
        # Visualization settings
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111)
        self.plot = plotting.Plotting(s_start, s_goal)
        
        # Current search state
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []
        self.complexity_regions = []
        
        # Performance metrics
        self.mode_switches = 0
        self.expansions_per_region = dict()
        
        # Tracking current region
        self.current_region = self.get_region(s_start)
        
        # Subgrid structure (for multi-resolution search)
        self.grid_resolution = dict()  # Resolution per region
        self.initialize_grid_resolution()

    def analyze_environment(self):
        """
        Analyze environment to create complexity map
        """
        # Divide environment into regions
        region_size = 10  # Size of each region
        complexity_map = {}
        
        # Calculate obstacle density for each region
        for i in range(0, self.Env.x_range, region_size):
            for j in range(0, self.Env.y_range, region_size):
                region = (i // region_size, j // region_size)
                
                # Count obstacles in this region
                obstacle_count = 0
                total_cells = 0
                
                for x in range(i, min(i + region_size, self.Env.x_range)):
                    for y in range(j, min(j + region_size, self.Env.y_range)):
                        total_cells += 1
                        if (x, y) in self.obs:
                            obstacle_count += 1
                
                # Calculate density and complexity score
                if total_cells > 0:
                    density = obstacle_count / total_cells
                    
                    # Complexity score: based on density and pattern
                    # Higher density = higher complexity
                    # Medium density with obstacles in patterns = highest complexity
                    complexity = 0.0
                    
                    if density == 0:  # Open space
                        complexity = 0.1
                    elif density < 0.1:  # Sparse obstacles
                        complexity = 0.3
                    elif density < 0.3:  # Medium density
                        complexity = 0.7
                    else:  # High density
                        complexity = 0.9
                    
                    # Add edge complexity if region borders obstacles
                    edge_complexity = 0.0
                    if self.has_region_edge_obstacles(i, j, region_size):
                        edge_complexity = 0.2
                        
                    complexity = min(1.0, complexity + edge_complexity)
                    
                    complexity_map[region] = {
                        'density': density,
                        'complexity': complexity,
                        'obstacle_count': obstacle_count,
                        'bounds': (i, j, i + region_size, j + region_size)
                    }
                    
                    # Higher complexity regions get higher LOS precision
                    self.los_precision[region] = max(0.3, min(1.0, complexity * 1.5))
                
        return complexity_map

    def has_region_edge_obstacles(self, i, j, size):
        """
        Check if region has obstacles on its edges (complex boundary)
        """
        # Check top and bottom edges
        for x in range(i, min(i + size, self.Env.x_range)):
            if (x, j) in self.obs or (x, min(j + size - 1, self.Env.y_range - 1)) in self.obs:
                return True
                
        # Check left and right edges
        for y in range(j, min(j + size, self.Env.y_range)):
            if (i, y) in self.obs or (min(i + size - 1, self.Env.x_range - 1), y) in self.obs:
                return True
                
        return False

    def initialize_grid_resolution(self):
        """
        Initialize grid resolution based on complexity map
        """
        for region, data in self.complexity_map.items():
            complexity = data['complexity']
            
            # Set grid resolution based on complexity
            if complexity < 0.3:
                # Low complexity - coarse grid
                self.grid_resolution[region] = 2
            elif complexity < 0.7:
                # Medium complexity - normal grid
                self.grid_resolution[region] = 1
            else:
                # High complexity - fine grid
                self.grid_resolution[region] = 0.5

    def get_region(self, pos):
        """
        Get the region containing a position
        """
        region_size = 10
        return (pos[0] // region_size, pos[1] // region_size)

    def adjust_heuristic_weight(self, current, remaining_time):
        """
        Adaptively adjust heuristic weight based on:
        1. Current region complexity
        2. Remaining search time
        3. Distance to goal
        """
        region = self.get_region(current)
        
        # Get complexity of current region
        complexity = 0.5  # Default medium complexity
        if region in self.complexity_map:
            complexity = self.complexity_map[region]['complexity']
        
        # Distance to goal factor (normalized)
        dist_to_goal = self.heuristic(current, self.s_goal)
        max_dist = self.heuristic(self.s_start, self.s_goal)
        dist_factor = dist_to_goal / max_dist if max_dist > 0 else 0
        
        # Time pressure factor (0 to 1)
        time_factor = 1.0 - (remaining_time / self.time_limit)
        time_factor = max(0, min(1, time_factor))  # Clamp to [0,1]
        
        # Calculate adaptive weight:
        # - Higher when far from goal (more greedy)
        # - Higher when time is running out (more greedy)
        # - Lower in complex regions (more careful exploration)
        # - Base weight is 1.0 (standard A*)
        
        # Initialize weight based on region complexity
        weight = 1.0 - (complexity * 0.5)  # Range: 0.5 - 1.0
        
        # Increase weight as we get closer to goal (more greedy)
        weight += (1.0 - dist_factor) * 0.3  # Add up to 0.3
        
        # Increase weight as time runs out (more greedy)
        weight += time_factor * 0.7  # Add up to 0.7
        
        # Clamp to reasonable range: 0.8 - 2.0
        weight = max(0.8, min(2.0, weight))
        
        # Don't change weight too abruptly
        delta = weight - self.heuristic_weight
        self.heuristic_weight += delta * 0.2  # Smooth transition
        
        return self.heuristic_weight

    def adaptive_line_of_sight(self, s_start, s_end):
        """
        Adaptively check line-of-sight with precision based on region complexity
        """
        # Check cache first
        cache_key = (s_start, s_end)
        if cache_key in self.los_cache:
            return self.los_cache[cache_key]
            
        start_time = time.time()
        
        # Basic collision check first
        if self.is_collision(s_start, s_end):
            self.los_cache[cache_key] = False
            self.los_computation_time += time.time() - start_time
            return False
            
        # Determine precision based on region complexities
        start_region = self.get_region(s_start)
        end_region = self.get_region(s_end)
        
        # Use the higher precision of the two regions
        precision = 0.5  # Default medium precision
        if start_region in self.los_precision:
            precision = max(precision, self.los_precision[start_region])
        if end_region in self.los_precision:
            precision = max(precision, self.los_precision[end_region])
            
        # Calculate distance and determine number of checks
        distance = math.hypot(s_end[0] - s_start[0], s_end[1] - s_start[1])
        
        # Adjust checks based on distance and precision
        checks = max(1, int(distance * precision))
        
        # For very short distances or low precision, skip detailed checking
        if distance <= 2 or precision < 0.3:
            self.los_cache[cache_key] = True
            self.los_computation_time += time.time() - start_time
            return True
            
        # Check intermediate points
        dx = s_end[0] - s_start[0]
        dy = s_end[1] - s_start[1]
        
        for i in range(1, checks):
            t = i / checks
            x = int(round(s_start[0] + t * dx))
            y = int(round(s_start[1] + t * dy))
            
            if (x, y) in self.obs:
                self.los_cache[cache_key] = False
                self.los_computation_time += time.time() - start_time
                return False
                
        self.los_checks += 1
        self.los_cache[cache_key] = True
        self.los_computation_time += time.time() - start_time
        return True

    def searching(self):
        """
        Adaptive Theta* pathfinding
        """
        # Initialize visualization and timing
        self.plot.plot_grid("Adaptive Theta*")
        self.start_time = time.time()
        
        # Initialize data structures
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.f[self.s_start] = self.heuristic(self.s_start, self.s_goal)
        
        heapq.heappush(self.OPEN, (self.f[self.s_start], self.s_start))
        
        # For visualization, store regions by complexity
        for region, data in self.complexity_map.items():
            x1, y1, x2, y2 = data['bounds']
            complexity = data['complexity']
            self.complexity_regions.append((x1, y1, x2, y2, complexity))
        
        # Main search loop
        expansions = 0
        mode = "standard"  # Start in standard mode
        
        while self.OPEN:
            # Check remaining time and adjust strategy if needed
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0.1, self.time_limit - elapsed_time)
            
            # Track expansion rate for estimating completion time
            if expansions - self.last_expansion_check >= 100:
                duration = time.time() - self.start_time
                rate = expansions / duration if duration > 0 else 0
                self.node_expansion_rate.append(rate)
                self.last_expansion_check = expansions
                
                # Predict required expansions and adjust strategy if needed
                if self.adaptive_modes:
                    estimated_expansions = self.estimate_required_expansions()
                    
                    # If we're going to run out of time, switch to faster mode
                    if rate > 0 and estimated_expansions / rate > remaining_time:
                        if mode == "standard":
                            mode = "greedy"
                            self.heuristic_weight = 1.5  # More greedy
                            self.mode_switches += 1
                        elif mode == "greedy" and remaining_time < self.time_limit * 0.2:
                            mode = "very_greedy"
                            self.heuristic_weight = 2.0  # Very greedy
                            self.mode_switches += 1
                    elif mode != "standard" and estimated_expansions / rate < remaining_time * 0.5:
                        # Switch back to standard if we have plenty of time
                        mode = "standard"
                        self.heuristic_weight = 1.0
                        self.mode_switches += 1
            
            # Get next node from open set
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            self.current_visited.append(s)
            expansions += 1
            
            # Track expansions by region
            region = self.get_region(s)
            if region not in self.expansions_per_region:
                self.expansions_per_region[region] = 0
            self.expansions_per_region[region] += 1
            
            # Check if we changed regions
            if region != self.current_region:
                self.current_region = region
            
            # Update current path and visualization
            if s == self.s_goal:
                self.current_path = self.extract_path(self.PARENT)
            else:
                # Show path from start to current node
                temp_path = self.extract_temp_path(s)
                self.current_path = temp_path
            
            # Update visualization periodically
            if len(self.CLOSED) % 5 == 0 or s == self.s_goal:
                self.update_plot(mode=mode, elapsed_time=elapsed_time)
            
            # Check if goal reached
            if s == self.s_goal:
                break
                
            # Check for timeout
            if elapsed_time >= self.time_limit:
                print("Search timeout reached!")
                break
            
            # Adjust heuristic weight adaptively
            if self.adaptive_modes:
                self.heuristic_weight = self.adjust_heuristic_weight(s, remaining_time)
            
            # Process neighbors
            for s_n in self.get_adaptive_neighbors(s):
                # Adaptive LOS check
                los_result = False
                if s != self.s_start:
                    los_result = self.adaptive_line_of_sight(self.PARENT[s], s_n)
                    self.current_los_checks.append((self.PARENT[s], s_n, los_result))
                
                # Path 2 - Through parent's parent (Theta* any-angle path)
                if los_result:
                    # Direct path from parent
                    new_g = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)
                    
                    if s_n not in self.g or new_g < self.g[s_n]:
                        self.g[s_n] = new_g
                        self.PARENT[s_n] = self.PARENT[s]
                        f_value = new_g + self.heuristic(s_n, self.s_goal) * self.heuristic_weight
                        self.f[s_n] = f_value
                        heapq.heappush(self.OPEN, (f_value, s_n))
                else:
                    # Path 1 - Traditional A* path
                    new_g = self.g[s] + self.cost(s, s_n)
                    
                    if s_n not in self.g or new_g < self.g[s_n]:
                        self.g[s_n] = new_g
                        self.PARENT[s_n] = s
                        f_value = new_g + self.heuristic(s_n, self.s_goal) * self.heuristic_weight
                        self.f[s_n] = f_value
                        heapq.heappush(self.OPEN, (f_value, s_n))
        
        # Final update
        self.update_plot(mode=mode, elapsed_time=time.time() - self.start_time, final=True)
        plt.show()
        
        # If goal not reached, find best partial path
        if not self.current_path or self.current_path[-1] != self.s_goal:
            print("Goal not reached! Finding best partial path...")
            self.current_path = self.find_best_partial_path()
        
        return self.current_path, self.CLOSED

    def get_adaptive_neighbors(self, s):
        """
        Get neighbors with adaptive grid resolution based on region complexity
        """
        region = self.get_region(s)
        resolution = 1  # Default standard grid resolution
        
        if region in self.grid_resolution:
            resolution = self.grid_resolution[region]
        
        neighbors = []
        
        # Standard neighbors for all resolutions
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            
            # Check if valid
            if self.is_valid_position(s_next):
                neighbors.append(s_next)
        
        # Add extra neighbors for fine grid resolution
        if resolution < 1:
            # Add diagonal movements with finer steps
            diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in diagonals:
                # Half-step diagonals
                s_next = (s[0] + dx * resolution, s[1] + dy * resolution)
                
                # Convert to integer coordinates
                s_next = (int(round(s_next[0])), int(round(s_next[1])))
                
                # Check if valid and not already added
                if self.is_valid_position(s_next) and s_next not in neighbors:
                    neighbors.append(s_next)
        
        # For coarse grid in simple regions, can skip some neighbors
        elif resolution > 1 and region in self.complexity_map:
            complexity = self.complexity_map[region]['complexity']
            
            # In very simple regions, use larger steps
            if complexity < 0.2:
                # Filter to keep only cardinal directions and main diagonals
                filtered = []
                for n in neighbors:
                    dx, dy = n[0] - s[0], n[1] - s[1]
                    if abs(dx) <= 1 and abs(dy) <= 1:  # Keep immediate neighbors
                        filtered.append(n)
                neighbors = filtered
        
        return neighbors

    def is_valid_position(self, pos):
        """
        Check if position is valid (within grid and not in obstacles)
        """
        x, y = pos
        
        # Check grid boundaries
        if not (0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range):
            return False
            
        # Check if position is obstacle
        if (int(round(x)), int(round(y))) in self.obs:
            return False
            
        return True

    def estimate_required_expansions(self):
        """
        Estimate how many node expansions will be needed to reach goal
        """
        if not self.current_path:
            # If no path yet, use heuristic distance as rough estimate
            dist = self.heuristic(self.s_start, self.s_goal)
            return dist * 10  # Very rough estimate
            
        # Get the path so far
        path_end = self.current_path[-1]
        
        if path_end == self.s_goal:
            return len(self.CLOSED)  # We're done
            
        # Estimate based on current progress
        dist_traveled = self.g[path_end]
        dist_remaining = self.heuristic(path_end, self.s_goal)
        
        if dist_traveled > 0:
            # Estimate expansions based on current expansion/distance ratio
            expansions_per_dist = len(self.CLOSED) / dist_traveled
            estimated_remaining = dist_remaining * expansions_per_dist
            return len(self.CLOSED) + estimated_remaining
        else:
            return len(self.CLOSED) * 2  # Simple fallback estimate
    
    def find_best_partial_path(self):
        """
        Find the best partial path when goal cannot be reached
        """
        # If no nodes expanded, return just the start position
        if not self.CLOSED:
            return [self.s_start]
            
        # Find the expanded node closest to the goal
        best_node = self.s_start
        best_f = float('inf')
        
        for node in self.CLOSED:
            if node in self.f:
                if self.f[node] < best_f:
                    best_f = self.f[node]
                    best_node = node
        
        # Extract path to this best node
        return self.extract_temp_path(best_node)

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

    def is_collision(self, s_start, s_end):
        """
        Check if line segment collides with obstacles
        """
        x0, y0 = s_start
        x1, y1 = s_end
        
        # Bresenham's line algorithm
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
            
        y = y0
        for x in range(x0, x1 + 1):
            if steep:
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

    def update_plot(self, mode="standard", elapsed_time=0, final=False):
        """
        Update visualization
        """
        # Clear current figure
        plt.cla()
        
        # Draw the grid with obstacles
        self.plot.plot_grid("Adaptive Theta*")
        
        # Draw complexity regions with color coding
        self.draw_complexity_regions()
        
        # Draw visited nodes
        if self.current_visited:
            for node in self.current_visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], color='gray', marker='.', markersize=3)
        
        # Draw line-of-sight checks
        if self.current_los_checks:
            for start, end, result in self.current_los_checks[-20:]:  # Show only recent checks
                color = 'g' if result else 'r'
                alpha = 0.3 if not final else 0.1  # Less visible in final view
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=alpha, linewidth=0.5)
        
        # Draw current path
        if self.current_path:
            self.plot.plot_path(self.current_path)
        
        # Draw status information
        status_info = [
            f"Mode: {mode.upper()}",
            f"Time: {elapsed_time:.2f}s / {self.time_limit:.1f}s",
            f"Nodes: {len(self.CLOSED)}",
            f"H-weight: {self.heuristic_weight:.2f}",
            f"LOS checks: {self.los_checks}"
        ]
        
        # Place text in top-left corner
        y_offset = 0.95
        for info in status_info:
            plt.text(0.02, y_offset, info, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.7))
            y_offset -= 0.05
        
        # Update figure
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        
        if final:
            plt.pause(0.5)
            # Show region statistics on final view
            self.show_final_statistics()
        else:
            plt.pause(0.01)

    def draw_complexity_regions(self):
        """
        Draw color-coded complexity regions
        """
        # Custom colormap from green (simple) to red (complex)
        cmap = LinearSegmentedColormap.from_list("complexity", 
                                                ["green", "yellow", "orange", "red"])
        
        for x1, y1, x2, y2, complexity in self.complexity_regions:
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=1, edgecolor='gray', 
                                   facecolor=cmap(complexity), 
                                   alpha=0.1)
            plt.gca().add_patch(rect)
            
            # Add complexity value text for significant regions
            if complexity > 0.4:
                plt.text(x1 + width/2, y1 + height/2, f"{complexity:.1f}", 
                       horizontalalignment='center', verticalalignment='center',
                       color='black', fontsize=8, alpha=0.7)

    def show_final_statistics(self):
        """
        Display final statistics about the adaptive search
        """
        # Create a statistical summary figure
        stats_fig = plt.figure(figsize=(10, 6))
        
        # Plot 1: Node expansions per region vs complexity
        plt.subplot(1, 2, 1)
        complexities = []
        expansions = []
        
        for region, count in self.expansions_per_region.items():
            if region in self.complexity_map:
                complexities.append(self.complexity_map[region]['complexity'])
                expansions.append(count)
        
        plt.scatter(complexities, expansions, alpha=0.7)
        plt.xlabel('Region Complexity')
        plt.ylabel('Node Expansions')
        plt.title('Expansions vs. Complexity')
        
        # Plot 2: Path quality analysis
        plt.subplot(1, 2, 2)
        
        # Calculate path metrics
        path_length = len(self.current_path)
        if path_length >= 2:
            path_cost = sum(self.cost(self.current_path[i], self.current_path[i+1]) 
                          for i in range(path_length-1))
            direct_cost = self.cost(self.s_start, self.s_goal)
            ratio = path_cost / direct_cost if direct_cost > 0 else 0
            
            # Display path metrics
            plt.text(0.5, 0.8, f"Path Length: {path_length} nodes", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.text(0.5, 0.7, f"Path Cost: {path_cost:.2f}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.text(0.5, 0.6, f"Path/Direct Ratio: {ratio:.2f}x", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.text(0.5, 0.5, f"Mode Switches: {self.mode_switches}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.text(0.5, 0.4, f"LOS computation: {self.los_computation_time:.2f}s", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            
            # Print detailed stats to console
            print(f"\nAdaptive Theta* Statistics:")
            print(f"Path length: {path_length} nodes")
            print(f"Path cost: {path_cost:.2f}")
            print(f"Path/Direct Ratio: {ratio:.2f}x")
            print(f"Nodes expanded: {len(self.CLOSED)}")
            print(f"Mode switches: {self.mode_switches}")
            print(f"LOS checks: {self.los_checks}")
            print(f"LOS computation time: {self.los_computation_time:.2f}s")
            
        plt.axis('off')
        plt.title('Path Statistics')
        plt.tight_layout()
        plt.show()

    def cost(self, a, b):
        """
        Calculate cost between two positions
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def heuristic(self, a, b):
        """
        Calculate heuristic distance
        """
        if self.heuristic_type == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:  # Default: Euclidean
            return math.hypot(a[0] - b[0], a[1] - b[1])

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set
        """
        path = [self.s_goal]
        s = self.s_goal

        while s != self.s_start:
            s = PARENT[s]
            path.append(s)

        return list(reversed(path))


def main():
    """
    Adaptive Theta*: Self-adapting any-angle path planning algorithm
    
    This algorithm extends Theta* by dynamically adapting its behavior based on:
    1. Environment complexity - Uses different strategies in different regions
    2. Computational resources - Adapts to time constraints
    3. Path quality requirements - Dynamically balances optimality and speed
    
    Features:
    - Region-based complexity analysis
    - Adaptive heuristic weighting
    - Multi-resolution grid
    - Variable line-of-sight precision
    - Dynamic mode switching
    - Visual complexity mapping
    """
    s_start = (5, 5)
    s_goal = (45, 25)
    
    # Create adaptive Theta* planner
    adaptive_theta = AdaptiveThetaStar(
        s_start=s_start,
        s_goal=s_goal,
        heuristic_type="euclidean",
        time_limit=10.0,  # 10 second time limit
        adaptive_modes=True  # Enable adaptive features
    )
    
    # Run planning
    path, closed = adaptive_theta.searching()
    
    # Path quality check
    if path[-1] == s_goal:
        print("Goal reached successfully!")
    else:
        print("Goal not reached. Returning best partial path.")


if __name__ == '__main__':
    main()
