"""
JPS
@author: clark bai
"""

import os
import sys
import math
import heapq
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


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

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = set()  # CLOSED set / VISITED
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        Jump Point Search - Simplified
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
            
            # Find all successors with pruning and jumping
            neighbors = []
            
            if current == self.s_start:
                # For start node, consider all valid neighbors
                for dx, dy in self.u_set:
                    nx, ny = current[0] + dx, current[1] + dy
                    if not self.is_obstacle((nx, ny)):
                        neighbors.append((nx, ny))
            else:
                # For other nodes, get pruned neighbors based on parent direction
                parent = self.PARENT[current]
                dx = current[0] - parent[0]
                dy = current[1] - parent[1]
                
                # Normalize direction
                if dx != 0:
                    dx = dx // abs(dx)
                if dy != 0:
                    dy = dy // abs(dy)
                
                # Get pruned neighbors
                neighbors = self.get_pruned_neighbors(current, (dx, dy))
                
                # Transform neighbors to actual coordinates
                neighbors = [(current[0] + nx, current[1] + ny) for nx, ny in neighbors]
            
            for neighbor in neighbors:
                # Check if neighbor is valid
                if self.is_obstacle(neighbor):
                    continue
                
                # Try to jump from current to neighbor
                jp = self.find_jump_point(current, neighbor)
                
                if jp:
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
        else:
            print(f"No path found after processing {nodes_processed} nodes")
            
        return path, visited
    
    def get_pruned_neighbors(self, node, direction):
        """
        Get pruned neighbors based on direction of travel
        :param node: Current node coordinates
        :param direction: Direction of travel (dx, dy)
        :return: List of relative neighbor coordinates to explore
        """
        x, y = node
        dx, dy = direction
        neighbors = []
        
        # Diagonal movement
        if dx != 0 and dy != 0:
            # Always include the diagonal direction
            neighbors.append((dx, dy))
            
            # Include horizontal and vertical directions if clear
            if not self.is_obstacle((x + dx, y)):
                neighbors.append((dx, 0))
            if not self.is_obstacle((x, y + dy)):
                neighbors.append((0, dy))
                
            # Check for forced neighbors (diagonal case)
            if self.is_obstacle((x - dx, y)) and not self.is_obstacle((x, y + dy)):
                neighbors.append((-dx, dy))
            if self.is_obstacle((x, y - dy)):
                neighbors.append((dx, -dy))
                
        # Cardinal movement - horizontal
        elif dx != 0:
            # Always include the horizontal direction
            neighbors.append((dx, 0))
            
            # Check for forced neighbors (horizontal case)
            if self.is_obstacle((x, y + 1)):
                neighbors.append((dx, 1))
            if self.is_obstacle((x, y - 1)):
                neighbors.append((dx, -1))
                
        # Cardinal movement - vertical
        elif dy != 0:
            # Always include the vertical direction
            neighbors.append((0, dy))
            
            # Check for forced neighbors (vertical case)
            if self.is_obstacle((x + 1, y)):
                neighbors.append((1, dy))
            if self.is_obstacle((x - 1, y)):
                neighbors.append((-1, dy))
                
        return neighbors
    
    def find_jump_point(self, current, neighbor):
        """
        Simplified jump point detection - returns the neighbor or goal directly
        :param current: Current node
        :param neighbor: Neighbor node
        :return: Jump point (neighbor or goal)
        """
        # Direction from current to neighbor
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        
        # Normalize direction (will be used for forced neighbor detection)
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        # Check if neighbor is a valid position
        if self.is_obstacle(neighbor):
            return None
            
        # If neighbor is the goal, return it immediately
        if neighbor == self.s_goal:
            return neighbor
            
        # Check if neighbor is a forced neighbor
        # For diagonal movement
        if dx != 0 and dy != 0:
            # Check for obstacles forcing a jump point
            nx, ny = neighbor
            if ((self.is_obstacle((nx - dx, ny)) and not self.is_obstacle((nx - dx, ny + dy))) or
                (self.is_obstacle((nx, ny - dy)) and not self.is_obstacle((nx + dx, ny - dy)))):
                return neighbor
                
        # For horizontal movement
        elif dx != 0:
            nx, ny = neighbor
            if ((self.is_obstacle((nx, ny + 1)) and not self.is_obstacle((nx + dx, ny + 1))) or
                (self.is_obstacle((nx, ny - 1)) and not self.is_obstacle((nx + dx, ny - 1)))):
                return neighbor
                
        # For vertical movement
        elif dy != 0:
            nx, ny = neighbor
            if ((self.is_obstacle((nx + 1, ny)) and not self.is_obstacle((nx + 1, ny + dy))) or
                (self.is_obstacle((nx - 1, ny)) and not self.is_obstacle((nx - 1, ny + dy)))):
                return neighbor
        
        # In simplified JPS, we return the direct neighbor
        # This makes the algorithm work more like A* but with pruning
        return neighbor
    
    def is_obstacle(self, node):
        """
        Check if a node is an obstacle or out of bounds
        :param node: Node to check
        :return: True if obstacle or out of bounds
        """
        x, y = node
        
        # Check if out of bounds
        if x < 0 or x >= self.Env.x_range or y < 0 or y >= self.Env.y_range:
            return True
            
        # Check if in obstacle set
        if node in self.obs:
            return True
            
        return False

    def cost(self, s_start, s_goal):
        """
        Calculate cost between two nodes
        :param s_start: starting node
        :param s_goal: end node
        :return: Cost for this motion
        """
        if self.is_obstacle(s_start) or self.is_obstacle(s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def f_value(self, s):
        """
        Calculate f value (f = g + h)
        :param s: current state
        :return: f value
        """
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set
        :return: The planning path
        """
        # Check if a path was found
        if self.s_goal not in PARENT:
            return []
            
        # Reconstruct path
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return path  # Return path from goal to start (consistent with other algorithms)

    def heuristic(self, s):
        """
        Calculate heuristic
        :param s: current node
        :return: heuristic value
        """
        goal = self.s_goal
        heuristic_type = self.heuristic_type

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:  # euclidean
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def run_jps(s_start, s_goal, title=""):
    """
    Run Jump Point Search (JPS)
    :param s_start: Start point
    :param s_goal: Goal point
    :param title: Title for the JPS run
    """
    if title:
        print(f"\n===== {title} =====")
    
    # Create plotting object
    plot = plotting.Plotting(s_start, s_goal)
    
    # Create JPS object
    jps = JPS(s_start, s_goal, "euclidean")
    
    # Display environment info
    print(f"Grid size: {jps.Env.x_range} × {jps.Env.y_range}")
    print(f"Start: {s_start}, Goal: {s_goal}")
    print(f"Number of obstacles: {len(jps.Env.obs)}")
    
    # Run JPS
    print("\nRunning Jump Point Search (JPS)...")
    start_time = time.time()
    jps_path, jps_visited = jps.searching()
    end_time = time.time()
    jps_time = end_time - start_time
    
    print(f"JPS Runtime: {jps_time:.4f} seconds")
    print(f"JPS Nodes explored: {len(jps_visited)}")
    
    if jps_path:
        print(f"JPS found a path with {len(jps_path)} nodes")
        print("Visualizing JPS path...")
        plot.animation(jps_path, jps_visited, "Jump Point Search (JPS)")
    else:
        print("JPS could not find a path.")


def main():
    """
    Testing JPS implementation
    """
    print("Jump Point Search (JPS) Implementation")
    print("--------------------------------------")

    s_start = (5, 5)
    s_goal = (45, 25)
    run_jps(s_start, s_goal, "Test Case 2: Long Distance")


if __name__ == '__main__':
    main()