"""
JPS
@author: clark bai
"""

import os
import sys
import math
import heapq
import time
import matplotlib.pyplot as plt

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
        
        # Record jump points
        self.jump_points = []

    def searching(self):
        """
        Jump Point Search - Simplified
        :return: path, visited order
        """
        # Create figure and axes for dynamic plotting
        fig, ax = plt.subplots()
        plot = plotting.Plotting(self.s_start, self.s_goal)
        plot.plot_grid("Jump Point Search (JPS) - Live Demo")
        
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
                # Plot final path
                path = self.extract_path(self.PARENT)
                plot.plot_path(path)
                plt.pause(0.5)
                break
            
            # Debug - print current position periodically
            if nodes_processed % 50 == 0:
                print(f"Processing node {nodes_processed}: {current}")
            
            # Find all successors
            neighbors = self.get_neighbors(current)
            
            # Dynamic plotting - plot current node and visited nodes
            plt.plot(current[0], current[1], 'ro', markersize=6) # Current node
            
            # Plot visited nodes
            for node_v in visited: # Renamed 'node' to 'node_v' to avoid conflict
                if node_v != self.s_start and node_v != self.s_goal and node_v != current:
                    plt.plot(node_v[0], node_v[1], 'gray', marker='.', markersize=2)
            
            # Plot currently considered neighbors
            for neighbor_n in neighbors: # Renamed 'neighbor' to 'neighbor_n'
                if not self.is_obstacle(neighbor_n):
                    plt.plot(neighbor_n[0], neighbor_n[1], 'yo', markersize=4, alpha=0.5) # Neighbor nodes
            
            # Update display
            plt.pause(0.01)
            
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
                    plt.plot(jp[0], jp[1], 'bo', markersize=7) # Jump point
                    plt.plot([current[0], jp[0]], [current[1], jp[1]], 'g-', linewidth=1.5, alpha=0.7) # Connection to jump point
                    plt.pause(0.05)  # Pause longer when a jump point is found
                    
                    # Calculate cost to this jump point
                    new_cost = self.g[current] + self.cost(current, jp)
                    
                    # Update if better path found
                    if jp not in self.g or new_cost < self.g[jp]:
                        self.g[jp] = new_cost
                        self.PARENT[jp] = current
                        heapq.heappush(self.OPEN, (self.f_value(jp), jp))
                        
                        # Plot temporary path
                        temp_path = self.extract_temp_path(jp)
                        # Clear previous path lines
                        # Be careful with line removal, ensure it targets the correct lines
                        # This part might need adjustment based on how lines are stored or identified
                        lines_to_remove = [line for line in ax.get_lines() if line.get_color() == 'blue' and line.get_linestyle() == '-']
                        for line in lines_to_remove:
                            line.remove()
                        # Plot new temporary path
                        if temp_path:
                            xs = [x_coord for x_coord, y_coord in temp_path] # Renamed x, y to x_coord, y_coord
                            ys = [y_coord for x_coord, y_coord in temp_path]
                            plt.plot(xs, ys, 'b-', linewidth=2) # Temporary path
                            plt.pause(0.5)
        
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
            for node_v in visited: # Renamed 'node' to 'node_v'
                if node_v != self.s_start and node_v != self.s_goal:
                    plt.plot(node_v[0], node_v[1], 'gray', marker='.', markersize=2)
            
            # Plot jump points
            jump_points_only = list(set([jp_node for _, jp_node in self.jump_points])) # Renamed jp[1] to jp_node
            for jp_node_item in jump_points_only: # Renamed jp to jp_node_item
                if jp_node_item != self.s_start and jp_node_item != self.s_goal:
                    plt.plot(jp_node_item[0], jp_node_item[1], 'bo', markersize=7)
            
            # Plot jump point connection lines
            for start_node, end_node in self.jump_points: # Renamed start, end
                plt.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'g-', linewidth=1.5, alpha=0.7)
            
            # Plot the final path
            plot.plot_path(path)
            # Add legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='r', label='Start Point', markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='g', label='Goal Point', markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='.', color='gray', label='Visited Node', markersize=2, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='b', label='Jump Point', markersize=7, linestyle='None'),
                plt.Line2D([0], [0], color='g', label='Jump Connection', linewidth=1.5, alpha=0.7),
                plt.Line2D([0], [0], color='r', label='Final Path', linewidth=2)
            ]
            plt.legend(handles=handles)
  
            plt.pause(1) # Pause for 1 second to view final result
        else:
            print(f"No path found after processing {nodes_processed} nodes")
            
        plt.show()
        return path, visited

    def extract_temp_path(self, current_node): # Renamed current to current_node
        """
        Extract temporary path from start to current node
        :param current_node: Current node
        :return: Temporary path
        """
        path = [current_node]
        s_node = current_node # Renamed s to s_node
        
        while s_node != self.s_start:
            if s_node not in self.PARENT:
                return [] # Path does not exist
            s_node = self.PARENT[s_node]
            path.append(s_node)
        
        return list(reversed(path))

    def get_neighbors(self, s_node): # Renamed s to s_node
        """
        Find neighbors of state s_node that are not in obstacles
        :param s_node: State
        :return: Neighbors
        """
        nei_list = []
        for u_motion in self.u_set: # Renamed u to u_motion
            s_next = (s_node[0] + u_motion[0], s_node[1] + u_motion[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list
    
    def find_jump_point(self, current_node, neighbor_node): # Renamed current, neighbor
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
        node_to_check = neighbor_node # Renamed node to node_to_check
        steps = 0
        max_steps = 1000  # Increase the maximum number of steps to prevent early return
        
        while steps < max_steps:
            steps += 1
            x_coord, y_coord = node_to_check # Renamed x, y
            
            # Diagonal movement
            if dx != 0 and dy != 0:
                # Check forced neighbors
                if ((self.is_obstacle((x_coord - dx, y_coord)) and not self.is_obstacle((x_coord - dx, y_coord + dy))) or
                    (self.is_obstacle((x_coord, y_coord - dy)) and not self.is_obstacle((x_coord + dx, y_coord - dy)))):
                    return node_to_check
                    
                # Recursively check horizontal and vertical directions
                # Note: Recursive calls within an iterative function can be complex.
                # This part of JPS is crucial and often involves careful state management.
                # For simplicity in this translation, the recursive calls are kept,
                # but a fully iterative JPS might handle this differently.
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
            
        # If max_steps reached or no jump point condition met along the straight line,
        # this implies no jump point was found by forced neighbor or goal conditions.
        # The original JPS might return None here if only straight line check fails without forced neighbor.
        # The provided code returns 'node_to_check' (the last valid node in the line).
        # This behavior might differ from a strict JPS implementation.
        # For a strict JPS, if no jump point is found by the rules, it should return None.
        # However, the previous logic was "return node", so keeping it consistent.
        return node_to_check # Or return None if strict JPS behavior is desired here.
    
    def is_forced_neighbor(self, node_to_check, direction_vec): # Renamed node, direction
        """
        Check if the node_to_check has a forced neighbor
        :param node_to_check: Current node
        :param direction_vec: Movement direction (dx, dy)
        :return: Boolean, True if there is a forced neighbor
        """
        x_coord, y_coord = node_to_check # Renamed x, y
        dx_dir, dy_dir = direction_vec # Renamed dx, dy
        
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
    
    def is_obstacle(self, node_to_check): # Renamed node
        """
        Check if a node_to_check is an obstacle or out of bounds
        :param node_to_check: Node to check
        :return: True if obstacle or out of bounds, False otherwise
        """
        x_coord, y_coord = node_to_check # Renamed x, y
        
        # Check if out of bounds
        if not (0 <= x_coord < self.Env.x_range and 0 <= y_coord < self.Env.y_range):
            return True
            
        # Check if in obstacle set
        if node_to_check in self.obs:
            return True
            
        return False

    def cost(self, s_start_node, s_goal_node): # Renamed s_start, s_goal
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

    def f_value(self, s_node): # Renamed s
        """
        Calculate f value (f = g + h)
        :param s_node: current state/node
        :return: f value
        """
        return self.g[s_node] + self.heuristic(s_node)

    def extract_path(self, parent_map): # Renamed PARENT
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
        current_s = self.s_goal # Renamed s to current_s

        while current_s != self.s_start:
            current_s = parent_map[current_s]
            path.append(current_s)

            if current_s == self.s_start: # Should be caught by while condition, but good for clarity
                break
        
        path.reverse() # Reverse the path to be from start to goal
        return path

    def heuristic(self, s_node): # Renamed s
        """
        Calculate heuristic (estimated cost from s_node to goal)
        :param s_node: current node
        :return: heuristic value
        """
        goal_node = self.s_goal # Renamed goal to goal_node for clarity
        
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - s_node[0]) + abs(goal_node[1] - s_node[1])
        else:  # Default to euclidean
            return math.hypot(goal_node[0] - s_node[0], goal_node[1] - s_node[1])


def run_jps(s_start_coord, s_goal_coord, run_title=""): # Renamed s_start, s_goal, title
    """
    Run Jump Point Search (JPS)
    :param s_start_coord: Start point coordinates
    :param s_goal_coord: Goal point coordinates
    :param run_title: Title for the JPS run
    """
    if run_title:
        print(f"\n===== {run_title} =====")
    
    # Create JPS object
    jps_solver = JPS(s_start_coord, s_goal_coord, "euclidean") # Renamed jps to jps_solver
    
    # Display environment info
    print(f"Grid size: {jps_solver.Env.x_range} × {jps_solver.Env.y_range}")
    print(f"Start: {s_start_coord}, Goal: {s_goal_coord}")
    print(f"Number of obstacles: {len(jps_solver.Env.obs)}")
    
    # Run JPS
    print("\nRunning Jump Point Search (JPS)...")
    start_time = time.time()
    jps_path, jps_visited_nodes = jps_solver.searching() # Renamed jps_visited
    end_time = time.time()
    jps_run_time = end_time - start_time # Renamed jps_time
    
    print(f"JPS Runtime: {jps_run_time:.4f} seconds")
    print(f"JPS Nodes explored: {len(jps_visited_nodes)}")
    
    if jps_path:
        print(f"JPS found a path with {len(jps_path)} nodes")
    else:
        print("JPS could not find a path.")


def main():
    """
    Testing JPS implementation
    """
    print("Jump Point Search (JPS) Implementation")
    print("--------------------------------------")

    s_start_main = (5, 5) # Renamed s_start
    s_goal_main = (45, 25) # Renamed s_goal
    run_jps(s_start_main, s_goal_main, "Test Case JPS") # Changed title slightly


if __name__ == '__main__':
    main()
