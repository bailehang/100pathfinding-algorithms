"""
Euclidean JPS (Jump Point Search)
@author: clark bai
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


class EuclideanJPS:
    """Euclidean Jump Point Search algorithm
    
    A variant of JPS that emphasizes Euclidean distance metrics and optimizes
    the search process using Euclidean properties. It still maintains the core
    JPS concept of identifying "jump points" to reduce node expansion.
    """
    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = set()  # CLOSED set / VISITED
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        
        # Record jump points
        self.jump_points = []
        
        # For visualization
        self.fig, self.ax = plt.subplots()
        self.plot = plotting.Plotting(self.s_start, self.s_goal)

    def searching(self):
        """
        Euclidean Jump Point Search
        :return: path, visited order
        """
        # Initialize visualization
        self.plot.plot_grid("Euclidean Jump Point Search (E-JPS) - Live Demo")
        
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        visited = [self.s_start]
        
        # Add start node to open list
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        
        # Debug info
        print(f"Starting Euclidean JPS search from {self.s_start} to {self.s_goal}")
        nodes_processed = 0
        jump_points_found = 0
        
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
                print(f"Goal reached after processing {nodes_processed} nodes and finding {jump_points_found} jump points!")
                # Plot final path
                path = self.extract_path(self.PARENT)
                self.plot.plot_path(path)
                plt.pause(0.2)
                break
            
            # Debug - print current position periodically
            if nodes_processed % 50 == 0:
                print(f"Processing node {nodes_processed}: {current}")
            
            # Find all successors
            neighbors = self.get_neighbors(current)
            
            # Dynamic plotting - plot current node and visited nodes
            plt.plot(current[0], current[1], 'ro', markersize=6) # Current node
            
            # Plot visited nodes (avoiding replotting already plotted nodes)
            for node in visited[-10:]:  # Only plot recent visits to reduce plotting overhead
                if node != self.s_start and node != self.s_goal and node != current:
                    plt.plot(node[0], node[1], 'gray', marker='.', markersize=2)
            
            # Plot currently considered neighbors
            for neighbor in neighbors:
                if not self.is_obstacle(neighbor):
                    plt.plot(neighbor[0], neighbor[1], 'yo', markersize=4, alpha=0.5) # Neighbor nodes
            
            # Update display
            plt.pause(0.05)
            
            for neighbor in neighbors:
                # Check if neighbor is valid
                if self.is_obstacle(neighbor):
                    continue
                
                # Try to jump from current to neighbor
                jp = self.find_jump_point(current, neighbor)
                
                if jp:
                    jump_points_found += 1
                    # Record found jump point
                    self.jump_points.append((current, jp))
                    
                    # Plot jump point and connection line
                    plt.plot(jp[0], jp[1], 'bo', markersize=7) # Jump point
                    plt.plot([current[0], jp[0]], [current[1], jp[1]], 'g-', linewidth=1.5, alpha=0.7) # Connection to jump point
                    plt.pause(0.05)  # Pause longer when a jump point is found
                    
                    # Calculate cost to this jump point using Euclidean distance
                    new_cost = self.g[current] + self.euclidean_cost(current, jp)
                    
                    # Update if better path found
                    if jp not in self.g or new_cost < self.g[jp]:
                        self.g[jp] = new_cost
                        self.PARENT[jp] = current
                        heapq.heappush(self.OPEN, (self.f_value(jp), jp))
                        
                        # Plot temporary path
                        temp_path = self.extract_temp_path(jp)
                        # Clear previous path lines
                        lines_to_remove = [line for line in self.ax.get_lines() if line.get_color() == 'blue' and line.get_linestyle() == '-']
                        for line in lines_to_remove:
                            line.remove()
                        # Plot new temporary path
                        if temp_path:
                            xs = [x for x, y in temp_path]
                            ys = [y for x, y in temp_path]
                            plt.plot(xs, ys, 'b-', linewidth=2) # Temporary path
                            plt.pause(0.05)
        
        # Extract path
        path = self.extract_path(self.PARENT)
        
        # Report results
        if path:
            print(f"Path found with {len(path)} nodes")
            print(f"Found {len(self.jump_points)} jump points")
            
            # Calculate total Euclidean path length
            path_length = self.calculate_path_length(path)
            print(f"Total Euclidean path length: {path_length:.2f}")
            
            # Final plot
            plt.cla() # Clear current axes
            self.plot.plot_grid("Euclidean Jump Point Search (E-JPS) - Final Result")
            
            # Plot visited nodes
            for node in visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], 'gray', marker='.', markersize=2)
            
            # Plot jump points
            jump_points_only = list(set([jp for _, jp in self.jump_points]))
            for jp in jump_points_only:
                if jp != self.s_start and jp != self.s_goal:
                    plt.plot(jp[0], jp[1], 'bo', markersize=7)
            
            # Plot jump point connection lines
            for start, end in self.jump_points:
                plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=1.5, alpha=0.7)
            
            # Plot the final path
            self.plot.plot_path(path)
            
            # Create legend with correct colors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Start Point'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Goal Point'),
                Line2D([0], [0], marker='.', color='gray', markersize=6, label='Visited Node'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Jump Point'),
                Line2D([0], [0], color='g', lw=1.5, label='Jump Connection'),
                Line2D([0], [0], color='r', lw=2, label='Final Path')
            ]
            # Add legend with explicit colors
            plt.legend(handles=legend_elements, loc='best')
            
            plt.title(f"Euclidean JPS - Path Length: {path_length:.2f}")
            plt.pause(1) # Pause for 1 second to view final result
        else:
            print(f"No path found after processing {nodes_processed} nodes")
            
        plt.show()
        return path, visited

    def extract_temp_path(self, current):
        """
        Extract temporary path from start to current node
        :param current: Current node
        :return: Temporary path
        """
        path = [current]
        s = current
        
        while s != self.s_start:
            if s not in self.PARENT:
                return [] # Path does not exist
            s = self.PARENT[s]
            path.append(s)
        
        return list(reversed(path))

    def get_neighbors(self, s):
        """
        Find neighbors of state s that are not in obstacles
        :param s: State
        :return: Neighbors
        """
        nei_list = []
        
        # For Euclidean JPS, we consider both cardinal and diagonal moves
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list
    
    def find_jump_point(self, current, neighbor):
        """
        Detect jump point for Euclidean JPS - Enhanced with Euclidean properties
        :param current: Current node
        :param neighbor: Neighbor node
        :return: Jump point or None
        """
        # Direction from current to neighbor
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        
        # Normalize the direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        
        # Check if the initial neighbor is valid
        if self.is_obstacle(neighbor):
            return None
        
        # If the neighbor is the goal, return it immediately
        if neighbor == self.s_goal:
            return neighbor
            
        # Start iterative check
        node = neighbor
        steps = 0
        max_steps = 1000  # Maximum steps to prevent infinite loops
        
        while steps < max_steps:
            steps += 1
            x, y = node
            
            # Diagonal movement
            if dx != 0 and dy != 0:
                # Check for forced neighbors using Euclidean properties
                # In Euclidean JPS, we check in 8 directions for potential forced neighbors
                for check_dx, check_dy in [(dx, 0), (0, dy), (-dx, 0), (0, -dy), (dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy)]:
                    if check_dx == dx and check_dy == dy:  # Skip the current direction
                        continue
                    
                    check_node = (x + check_dx, y + check_dy)
                    if not self.is_obstacle(check_node):
                        # Check if there's a forced neighbor condition
                        # This is a simplified check for the example
                        blocker = (x - check_dx, y - check_dy)
                        if self.is_obstacle(blocker):
                            return node  # This is a jump point
                
                # Recursively check horizontal and vertical directions
                h_jp = self.find_jump_point(node, (x + dx, y))
                if h_jp:
                    return node # Current node is a jump point
                    
                v_jp = self.find_jump_point(node, (x, y + dy))
                if v_jp:
                    return node # Current node is a jump point
                
            # Horizontal movement
            elif dx != 0:
                # Check for forced neighbors in perpendicular directions
                if self.is_obstacle((x, y + 1)) and not self.is_obstacle((x + dx, y + 1)):
                    return node
                if self.is_obstacle((x, y - 1)) and not self.is_obstacle((x + dx, y - 1)):
                    return node
                    
            # Vertical movement
            elif dy != 0:
                # Check for forced neighbors in perpendicular directions
                if self.is_obstacle((x + 1, y)) and not self.is_obstacle((x + 1, y + dy)):
                    return node
                if self.is_obstacle((x - 1, y)) and not self.is_obstacle((x - 1, y + dy)):
                    return node
            
            # Calculate the next position in the current direction
            next_x, next_y = x + dx, y + dy
            next_pos = (next_x, next_y)
            
            # If the next position is invalid (obstacle or out of bounds), stop
            if self.is_obstacle(next_pos):
                return None # No jump point found in this direction
                
            # If the next position is the goal, it's a jump point
            if self.euclidean_distance(next_pos, self.s_goal) < 1.5:  # Close enough to the goal
                return self.s_goal
                
            # Move to the next position
            node = next_pos
            
        # If we reach this point, we've exceeded max_steps
        # For Euclidean JPS, we return the furthest node we reached
        return node
    
    def is_obstacle(self, node):
        """
        Check if a node is an obstacle or out of bounds
        :param node: Node to check
        :return: True if obstacle or out of bounds, False otherwise
        """
        x, y = node
        
        # Check if out of bounds
        if not (0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range):
            return True
            
        # Check if in obstacle set
        if node in self.obs:
            return True
            
        return False

    def euclidean_cost(self, s_start, s_goal):
        """
        Calculate Euclidean cost between two nodes
        :param s_start: starting node
        :param s_goal: end node
        :return: Euclidean distance
        """
        if self.is_obstacle(s_start) or self.is_obstacle(s_goal):
            return math.inf  # Infinite cost if one of the nodes is an obstacle

        return self.euclidean_distance(s_start, s_goal)
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        :param point1: First point (x, y)
        :param point2: Second point (x, y)
        :return: Euclidean distance
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def f_value(self, s):
        """
        Calculate f value (f = g + h)
        :param s: current state
        :return: f value
        """
        return self.g[s] + self.euclidean_heuristic(s)
    
    def euclidean_heuristic(self, s):
        """
        Calculate Euclidean heuristic (estimated cost from s to goal)
        :param s: current node
        :return: Euclidean distance to goal
        """
        return self.euclidean_distance(s, self.s_goal)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set
        :param PARENT: Dictionary storing parent of each node
        :return: The planning path from start to goal
        """
        # Check if a path to the goal was found
        if self.s_goal not in PARENT:
            return [] # No path found
            
        # Reconstruct path from goal to start
        path = [self.s_goal]
        s = self.s_goal

        while s != self.s_start:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break
        
        path.reverse() # Reverse the path to be from start to goal
        return path
    
    def calculate_path_length(self, path):
        """
        Calculate the total Euclidean length of a path
        :param path: List of points [(x1,y1), (x2,y2), ...]
        :return: Total Euclidean path length
        """
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(len(path) - 1):
            length += self.euclidean_distance(path[i], path[i+1])
            
        return length


def run_euclidean_jps(s_start, s_goal, title=""):
    """
    Run Euclidean Jump Point Search (E-JPS)
    :param s_start: Start point coordinates
    :param s_goal: Goal point coordinates
    :param title: Title for the E-JPS run
    """
    if title:
        print(f"\n===== {title} =====")
    
    # Create Euclidean JPS object
    ejps = EuclideanJPS(s_start, s_goal)
    
    # Display environment info
    print(f"Grid size: {ejps.Env.x_range} × {ejps.Env.y_range}")
    print(f"Start: {s_start}, Goal: {s_goal}")
    print(f"Number of obstacles: {len(ejps.Env.obs)}")
    
    # Run Euclidean JPS
    print("\nRunning Euclidean Jump Point Search (E-JPS)...")
    start_time = time.time()
    ejps_path, ejps_visited = ejps.searching()
    end_time = time.time()
    ejps_time = end_time - start_time
    
    print(f"E-JPS Runtime: {ejps_time:.4f} seconds")
    print(f"E-JPS Nodes explored: {len(ejps_visited)}")
    
    if ejps_path:
        print(f"E-JPS found a path with {len(ejps_path)} nodes")
        path_length = ejps.calculate_path_length(ejps_path)
        print(f"E-JPS path length: {path_length:.2f}")
    else:
        print("E-JPS could not find a path.")


def main():
    """
    Testing Euclidean JPS implementation
    """
    print("Euclidean Jump Point Search (E-JPS) Implementation")
    print("-------------------------------------------------")

    s_start = (5, 5)
    s_goal = (45, 25)
    run_euclidean_jps(s_start, s_goal, "Test Case Euclidean JPS")



if __name__ == '__main__':
    main()
