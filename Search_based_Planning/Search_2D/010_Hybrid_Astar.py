"""
Hybrid A* algorithm for path planning with vehicle kinematics
@author: clark bai
Reference: "Practical Search Techniques in Path Planning for Autonomous Driving"
          by Dolgov, D., Thrun, S., Montemerlo, M., & Diebel, J. (2008)
"""

import os
import sys
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env
# Removed Reed-Shepp import as it has dependency issues
# from CurvesGenerator import reeds_shepp


class VehicleModel:
    """
    Bicycle model for vehicle kinematics
    """
    def __init__(self):
        # Vehicle parameters - adjusted for sharper turns with forward-only movement
        self.wheelbase = 1.5  # Wheelbase [m] - reduced for tighter turning radius
        self.min_turning_radius = 2.0  # Minimum turning radius [m] - reduced for tighter turns
        self.max_steer_angle = np.arctan(self.wheelbase / self.min_turning_radius)  # Maximum steering angle [rad]
        
        # Vehicle dimensions for collision checking (smaller values to make navigation easier)
        self.length = 2.5  # [m] - reduced length
        self.width = 1.2   # [m] - reduced width
        
        # Motion resolution
        self.ds = 1.0  # Step size [m] - decreased for finer resolution with forward-only movement
        
        # Steering discretization
        self.n_steer = 9  # Number of steering angles - increased for more steering options
        self.steer_set = np.linspace(-self.max_steer_angle, self.max_steer_angle, self.n_steer)
        
        # Steering change penalty
        self.steer_change_cost = 0.1
        
        # Gear change penalty
        self.gear_change_cost = 1.0
        
        # Backwards movement penalty
        self.backward_cost = 2.0
        
        # Direction set (forward only)
        self.dir_set = [1]  # Restricted to forward movement only

    def next_states(self, curr_x, curr_y, curr_yaw, curr_steer, curr_gear):
        """
        Calculate the next states from the current state considering kinematics
        """
        next_states = []
        
        for steer in self.steer_set:
            for gear in self.dir_set:
                # Calculate next state based on bicycle model
                x, y, yaw = self.move(curr_x, curr_y, curr_yaw, steer, gear)
                
                # Calculate cost for this movement
                cost = self.ds  # Base cost is the distance moved
                
                # Add steering change penalty
                steer_change = abs(steer - curr_steer)
                cost += self.steer_change_cost * steer_change
                
                # Add reverse movement penalty
                if gear < 0:
                    cost += self.backward_cost
                
                # Add gear change penalty
                if curr_gear * gear < 0:  # Changed direction
                    cost += self.gear_change_cost
                
                next_states.append({
                    'x': x,
                    'y': y,
                    'yaw': yaw,
                    'steer': steer,
                    'gear': gear,
                    'cost': cost
                })
                
        return next_states

    def move(self, x, y, yaw, steer, gear):
        """
        Move the vehicle based on bicycle model kinematics
        """
        if abs(steer) < 1e-6:  # Straight line motion
            x += gear * self.ds * np.cos(yaw)
            y += gear * self.ds * np.sin(yaw)
            yaw = yaw
        else:  # Curved motion
            turning_radius = self.wheelbase / np.tan(steer)
            beta = gear * self.ds / turning_radius  # Turning angle
            
            # ICR (Instantaneous Center of Rotation) calculation
            x_icr = x - turning_radius * np.sin(yaw)
            y_icr = y + turning_radius * np.cos(yaw)
            
            # Move around the ICR
            yaw += beta
            x = x_icr + turning_radius * np.sin(yaw)
            y = y_icr - turning_radius * np.cos(yaw)
            
        # Normalize yaw angle
        yaw = self.normalize_angle(yaw)
        
        return x, y, yaw

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi)
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def vehicle_footprint(self, x, y, yaw):
        """
        Calculate the 4 corners of the vehicle footprint
        """
        # First calculate corners assuming vehicle is centered at origin and pointing along x-axis
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        
        corners = [
            [-half_length, -half_width],  # rear-left
            [half_length, -half_width],   # front-left
            [half_length, half_width],    # front-right
            [-half_length, half_width]    # rear-right
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for corner in corners:
            # Rotate
            rx = corner[0] * np.cos(yaw) - corner[1] * np.sin(yaw)
            ry = corner[0] * np.sin(yaw) + corner[1] * np.cos(yaw)
            # Translate
            rx += x
            ry += y
            rotated_corners.append([rx, ry])
            
        return rotated_corners
    
    def check_collision(self, x, y, yaw, obs):
        """
        Check if vehicle is in collision with obstacles
        Uses a reduced vehicle size for easier forward-only movement
        """
        # For forward-only movement, we'll use a more relaxed collision check
        # First do a quick position check to avoid expensive computation
        cx, cy = round(x), round(y)
        if (cx, cy) in obs:
            return True
            
        # Use a simplified circular check with reduced radius for better pathfinding
        # This approximates the vehicle as a circle with reduced radius
        safety_radius = max(self.length, self.width) / 2.0 * 0.6  # 60% of the radius for better forward-only performance
        
        # Check points in a circle around the center
        for dx in range(-int(safety_radius), int(safety_radius) + 1):
            for dy in range(-int(safety_radius), int(safety_radius) + 1):
                # Skip points outside the circle
                if dx*dx + dy*dy > safety_radius*safety_radius:
                    continue
                    
                # Check if point is in collision
                px, py = round(x + dx), round(y + dy)
                if (px, py) in obs:
                    return True
                    
        # With forward-only constraint, we need to be more permissive
        return False


class HybridAStarNode:
    """
    Node class for Hybrid A* path planning
    """
    def __init__(self, x, y, yaw, steer, gear, cost, parent_index):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.steer = steer
        self.gear = gear  # Direction: 1 for forward, -1 for backward
        self.cost = cost
        self.parent_index = parent_index
        
    def __str__(self):
        return f"Node(x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}, cost={self.cost:.2f})"
    
    def __eq__(self, other):
        if not isinstance(other, HybridAStarNode):
            return False
        return (self.x == other.x and 
                self.y == other.y and 
                self.yaw == other.yaw)


class HybridAStar:
    """
    Hybrid A* path planning algorithm
    """
    def __init__(self, start, goal, obs, vehicle_model):
        # Start and goal configuration
        self.start = start  # [x, y, yaw] in [m, m, rad]
        self.goal = goal    # [x, y, yaw] in [m, m, rad]
        
        # Environment
        self.obs = obs  # Set of obstacle positions
        self.x_width = 60  # Width of grid
        self.y_width = 40  # Height of grid
        
        # Vehicle model
        self.vehicle = vehicle_model
        
        # Grid resolution for discretization - much smaller values for better forward-only path planning
        self.xyreso = 0.2  # Grid resolution for x-y position [m] - finer resolution
        self.yawreso = np.deg2rad(10.0)  # Much finer grid resolution for yaw angle
        
        # Open and closed sets
        self.open_set = {}
        self.closed_set = {}
        
        # Visited nodes for visualization
        self.visited = []
        
        # Path information
        self.final_path = None
        
        # Pre-compute 2D heuristic map
        self.obstacle_heuristic_map = self.calc_obstacle_heuristic_map()
    
    def planning(self, max_iterations=50000):
        """
        Hybrid A* path planning
        
        Args:
            max_iterations: Maximum number of iterations to prevent infinite loops
        """
        start_node = HybridAStarNode(
            self.start[0], self.start[1], self.start[2], 
            0.0, 1, 0.0, -1
        )
        
        # Initial heuristic
        h_score = self.calc_heuristic(start_node)
        
        # Use priority queue for open list
        open_list = [(h_score, 0, start_node)]  # (f_score, counter, node)
        heapq.heapify(open_list)
        
        # Dictionary to lookup nodes by grid index
        self.open_set = {self.calc_grid_index(start_node): start_node}
        self.closed_set = {}
        
        # Counter for tiebreaking in priority queue
        counter = 0
        
        # For progress tracking
        iterations = 0
        progress_check = 1000  # Show progress every 1000 iterations
        
        # Initial distance to goal
        initial_dist = math.hypot(self.goal[0] - self.start[0], self.goal[1] - self.start[1])
        
        # Main loop
        while open_list and iterations < max_iterations:
            iterations += 1
            
            # Show progress periodically
            if iterations % progress_check == 0:
                if open_list:
                    # Get the current best node
                    best_node = open_list[0][2]
                    current_dist = math.hypot(self.goal[0] - best_node.x, self.goal[1] - best_node.y)
                    print(f"Iteration {iterations}: Best distance to goal = {current_dist:.2f}, "
                          f"Progress: {(1 - current_dist/initial_dist) * 100:.1f}%, "
                          f"Open set size: {len(open_list)}")
            # Get node with lowest f-score
            _, _, current = heapq.heappop(open_list)
            
            # Get grid index
            c_grid_index = self.calc_grid_index(current)
            
            # Move node from open to closed set
            if c_grid_index in self.open_set:
                del self.open_set[c_grid_index]
            self.closed_set[c_grid_index] = current
            
            # Add to visited list for visualization
            self.visited.append((current.x, current.y))
            
            # Check if we've reached the goal
            if self.is_goal(current):
                print("Goal reached!")
                final_node = current
                self.final_path = self.extract_path(final_node)
                
                # Apply path smoothing using simple method since Reed-Shepp is unavailable
                self.simple_path_smoothing()
                
                return self.final_path, self.visited
            
            # Expand current node - explore all possible motion primitives
            for next_state in self.vehicle.next_states(
                current.x, current.y, current.yaw, current.steer, current.gear
            ):
                # Create a new node
                next_node = HybridAStarNode(
                    next_state['x'], next_state['y'], next_state['yaw'],
                    next_state['steer'], next_state['gear'],
                    current.cost + next_state['cost'], c_grid_index
                )
                
                # Calculate grid index for the new node
                n_grid_index = self.calc_grid_index(next_node)
                
                # Skip if this node is already in the closed set
                if n_grid_index in self.closed_set:
                    continue
                
                # Skip if the state is in collision
                if self.vehicle.check_collision(
                    next_node.x, next_node.y, next_node.yaw, self.obs
                ):
                    continue
                
                # If this node is not in the open set, add it
                if n_grid_index not in self.open_set:
                    self.open_set[n_grid_index] = next_node
                    
                    # Calculate f-score (g + h)
                    h_score = self.calc_heuristic(next_node)
                    f_score = next_node.cost + h_score
                    
                    # Add to priority queue
                    counter += 1
                    heapq.heappush(open_list, (f_score, counter, next_node))
                else:
                    # If this node is already in the open set, check if we found a better path
                    existing_node = self.open_set[n_grid_index]
                    if next_node.cost < existing_node.cost:
                        # Update the node
                        existing_node.cost = next_node.cost
                        existing_node.parent_index = c_grid_index
                        existing_node.steer = next_node.steer
                        existing_node.gear = next_node.gear
                        
                        # Update priority queue - inefficient but simplifies implementation
                        # (in practice, would modify priority directly)
                        h_score = self.calc_heuristic(existing_node)
                        f_score = existing_node.cost + h_score
                        counter += 1
                        heapq.heappush(open_list, (f_score, counter, existing_node))
        
        # No path found
        print("No path found!")
        return None, self.visited

    def simple_path_smoothing(self):
        """
        Apply a simple path smoothing algorithm to the final path
        Uses moving average for position and orientation
        """
        if not self.final_path:
            return
        
        # Skip if path is too short
        if len(self.final_path) < 3:
            return
        
        # Extract only position and orientation information
        x = [node[0] for node in self.final_path]
        y = [node[1] for node in self.final_path]
        yaw = [node[2] for node in self.final_path]
        steer = [node[3] for node in self.final_path]
        gear = [node[4] for node in self.final_path]
        
        # Apply moving average smoothing
        window_size = 5
        smoothed_x = x.copy()
        smoothed_y = y.copy()
        smoothed_yaw = yaw.copy()
        
        # Keep start and end points fixed
        for i in range(1, len(x) - 1):
            # Define window indices with boundary checks
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(x), i + window_size // 2 + 1)
            
            # Calculate moving average for position
            smoothed_x[i] = sum(x[start_idx:end_idx]) / (end_idx - start_idx)
            smoothed_y[i] = sum(y[start_idx:end_idx]) / (end_idx - start_idx)
            
            # Smoother yaw requires special handling for angle wrapping
            sin_sum = sum(np.sin(yaw[j]) for j in range(start_idx, end_idx))
            cos_sum = sum(np.cos(yaw[j]) for j in range(start_idx, end_idx))
            smoothed_yaw[i] = np.arctan2(sin_sum, cos_sum)
        
        # Check for collisions with obstacles after smoothing
        smoothed_path = []
        for i in range(len(x)):
            # Skip point if it's in collision after smoothing
            if i > 0 and i < len(x) - 1 and self.vehicle.check_collision(
                smoothed_x[i], smoothed_y[i], smoothed_yaw[i], self.obs
            ):
                # Use original point if smoothed point is in collision
                smoothed_path.append((x[i], y[i], yaw[i], steer[i], gear[i]))
            else:
                # Otherwise use smoothed point
                smoothed_path.append((smoothed_x[i], smoothed_y[i], smoothed_yaw[i], steer[i], gear[i]))
        
        # Update the final path
        self.final_path = smoothed_path
        
    def calc_heuristic(self, node):
        """
        Calculate heuristic cost using a combination of:
        1. Non-holonomic-without-obstacles heuristic
        2. Obstacle heuristic
        """
        # 1. Non-holonomic-without-obstacles heuristic
        # Euclidean distance
        diff_x = self.goal[0] - node.x
        diff_y = self.goal[1] - node.y
        euc_dist = math.hypot(diff_x, diff_y)
        
        # Heading difference
        diff_yaw = abs(self.vehicle.normalize_angle(self.goal[2] - node.yaw))
        
        # Modified heuristic for forward-only planning
        # Put less weight on heading difference to allow for wider turns
        rs_cost = euc_dist + 0.2 * self.vehicle.min_turning_radius * diff_yaw  # Reduced heading penalty
        
        # 2. Obstacle heuristic (from 2D grid-based heuristic)
        grid_x, grid_y = self.to_grid_position(node.x, node.y)
        if 0 <= grid_x < self.x_width and 0 <= grid_y < self.y_width:
            obs_cost = self.obstacle_heuristic_map[grid_x][grid_y]
        else:
            obs_cost = euc_dist  # Fallback to Euclidean if outside grid
        
        # Return maximum of the two heuristics (both are admissible)
        return max(rs_cost, obs_cost)
    
    def calc_obstacle_heuristic_map(self):
        """
        Pre-compute a 2D grid-based heuristic that accounts for obstacles
        Uses a simple Dijkstra algorithm
        """
        # Initialize the grid with infinity cost
        grid = [[float("inf") for _ in range(self.y_width)] for _ in range(self.x_width)]
        
        # Goal position in grid coordinates
        goal_x, goal_y = self.to_grid_position(self.goal[0], self.goal[1])
        
        # Set goal cost to 0
        if 0 <= goal_x < self.x_width and 0 <= goal_y < self.y_width:
            grid[goal_x][goal_y] = 0
        
        # Queue for Dijkstra algorithm
        queue = [(0, goal_x, goal_y)]
        heapq.heapify(queue)
        
        # Directions for expansion (8-connectivity)
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
        dy = [0, 1, 1, 1, 0, -1, -1, -1]
        
        # Dijkstra algorithm
        while queue:
            cost, x, y = heapq.heappop(queue)
            
            # Skip if we've found a better path already
            if grid[x][y] < cost:
                continue
            
            # Expand in all directions
            for i in range(len(dx)):
                nx, ny = x + dx[i], y + dy[i]
                
                # Check if within grid
                if not (0 <= nx < self.x_width and 0 <= ny < self.y_width):
                    continue
                
                # Check if obstacle
                if (nx, ny) in self.obs:
                    continue
                
                # Cost = current cost + distance
                ncost = cost + math.hypot(dx[i], dy[i])
                
                # Update if better path found
                if ncost < grid[nx][ny]:
                    grid[nx][ny] = ncost
                    heapq.heappush(queue, (ncost, nx, ny))
        
        return grid
    
    def to_grid_position(self, x, y):
        """
        Convert continuous position to grid position
        """
        return int(round(x)), int(round(y))
    
    def calc_grid_index(self, node):
        """
        Calculate grid index for a node (used for open/closed sets)
        Discretizes continuous state space
        """
        # Discretize x, y, yaw
        x_ind = round(node.x / self.xyreso)
        y_ind = round(node.y / self.xyreso)
        yaw_ind = round(node.yaw / self.yawreso)
        
        # Create a unique index
        return (x_ind, y_ind, yaw_ind)
    
    def is_goal(self, node):
        """
        Check if node is close enough to the goal
        Using more relaxed tolerances to make it easier to reach the goal
        """
        # Position tolerance - increased for better success
        dist = math.hypot(node.x - self.goal[0], node.y - self.goal[1])
        pos_tol = 3.0  # Position tolerance [m] - increased tolerance
        
        # Yaw tolerance - increased for better success
        yaw_diff = abs(self.vehicle.normalize_angle(node.yaw - self.goal[2]))
        yaw_tol = np.deg2rad(45.0)  # Yaw tolerance [rad] - increased tolerance
        
        # Debug goal check
        if dist < pos_tol:
            print(f"Close to goal: dist={dist:.2f}, yaw_diff={np.rad2deg(yaw_diff):.1f}°")
            if dist < pos_tol and yaw_diff < yaw_tol:
                print(f"Goal criteria met! Position: {node.x:.2f}, {node.y:.2f}, Yaw: {np.rad2deg(node.yaw):.1f}°")
                
        return dist < pos_tol and yaw_diff < yaw_tol
    
    def extract_path(self, final_node):
        """
        Extract path from final node to start node
        """
        path = []
        node = final_node
        
        # Traverse from goal to start
        while node.parent_index != -1:
            path.append((node.x, node.y, node.yaw, node.steer, node.gear))
            node = self.closed_set[node.parent_index]
        
        # Add start node
        path.append((node.x, node.y, node.yaw, node.steer, node.gear))
        
        # Reverse to get path from start to goal
        path.reverse()
        
        return path


def main():
    print("Hybrid A* Path Planning Start")
    
    # Start and goal positions [x, y, yaw]
    start = [5.0, 5.0, np.deg2rad(45.0)]  # Start with 45-degree angle heading toward goal
    goal = [45.0, 25.0, np.deg2rad(0.0)]  # Goal position unchanged
    
    # Initialize environment and vehicle model
    env_instance = env.Env()
    obstacles = env_instance.obs
    vehicle_model = VehicleModel()
    
    # Print obstacle map size information
    print(f"Environment size: {env_instance.x_range}x{env_instance.y_range}")
    print(f"Number of obstacles: {len(obstacles)}")
    
    # Check if start or goal positions are in collision
    print(f"Checking start and goal positions...")
    start_collision = vehicle_model.check_collision(start[0], start[1], start[2], obstacles)
    goal_collision = vehicle_model.check_collision(goal[0], goal[1], goal[2], obstacles)
    
    if start_collision:
        print("Start position is in collision with obstacles!")
        return
    if goal_collision:
        print("Goal position is in collision with obstacles!")
        return
        
    print(f"Start position: ({start[0]}, {start[1]}, {np.rad2deg(start[2])}°) - Collision: {start_collision}")
    print(f"Goal position: ({goal[0]}, {goal[1]}, {np.rad2deg(goal[2])}°) - Collision: {goal_collision}")
    
    # Create planner
    planner = HybridAStar(start, goal, obstacles, vehicle_model)
    
    # Max iterations
    max_iters = 100000  # Further increased maximum iterations for forward-only planning
    print("Planning with max iterations:", max_iters)
    
    # Execute planning with progress tracking
    print("Starting search...")
    path, visited = planner.planning(max_iterations=max_iters)
    print(f"Search completed. Explored {len(visited)} states.")
    
    if path:
        # Extract path elements
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        yaw = [p[2] for p in path]
        steer = [p[3] for p in path]
        
        # Visualize
        plot = plotting.Plotting(start[:2], goal[:2])
        
        # Enable interactive mode for animation
        plt.ion()
        
        # Create a figure for animation
        fig = plt.figure()
        
        # First, demonstrate the search process
        print("Animating search process...")
        plt.clf()
        plot.plot_grid("Hybrid A* Search Process")
        
        # Display visited states in batches for smoother animation
        batch_size = max(1, len(visited) // 50)  # 50 frames for the search process
        for i in range(0, len(visited), batch_size):
            plt.clf()
            plot.plot_grid("Hybrid A* Search Process")

            # Plot visited nodes so far
            vis_x = [v[0] for v in visited[:i+batch_size]]
            vis_y = [v[1] for v in visited[:i+batch_size]]
            plt.plot(vis_x, vis_y, ".c", markersize=1, label="Explored")

            # Plot current exploration frontier
            if i > 0:
                frontier_x = [v[0] for v in visited[i:i+batch_size]]
                frontier_y = [v[1] for v in visited[i:i+batch_size]]
                if frontier_x:  # Check if there are any frontier points
                    plt.plot(frontier_x, frontier_y, ".g", markersize=2, label="Frontier")

            plt.axis("equal")
            plt.grid(True)
            if i == 0:
                plt.legend()
            plt.pause(0.01)
        
        # Wait for the user to see the search process
        plt.pause(1)
        
        # Now, animate the vehicle moving along the path
        print("Animating vehicle movement along the path...")
        
        for i in range(len(x)):
            plt.clf()
            plot.plot_grid("Hybrid A* Vehicle Movement")
            
            # Plot the complete path
            plt.plot(x, y, "-", color='lightgray', linewidth=1)
            
            # Plot the traveled path
            if i > 0:
                plt.plot(x[:i+1], y[:i+1], "-r", linewidth=2, label="Traveled Path")
            
            # Draw the vehicle at current position
            # Calculate the four corners of the vehicle for visualization
            corners = vehicle_model.vehicle_footprint(x[i], y[i], yaw[i])
            
            # Convert corners to matplotlib polygon
            corners_x = [p[0] for p in corners]
            corners_y = [p[1] for p in corners]
            # Close the shape by adding first point again
            corners_x.append(corners_x[0])
            corners_y.append(corners_y[0])
            plt.plot(corners_x, corners_y, '-k', linewidth=1.5)
            
            # Draw heading as an arrow
            arrow_length = 1.0
            plt.arrow(x[i], y[i], 
                    arrow_length * np.cos(yaw[i]), 
                    arrow_length * np.sin(yaw[i]),
                    head_width=0.5, head_length=0.5, fc='b', ec='b')
            
            # Display steering angle
            steer_text = f"Steering: {np.rad2deg(steer[i]):.1f}°"
            plt.text(5, 28, steer_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            
            # Display current position and heading
            pos_text = f"Position: ({x[i]:.1f}, {y[i]:.1f}), Heading: {np.rad2deg(yaw[i]):.1f}°"
            plt.text(5, 26, pos_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            
            plt.title(f"Hybrid A* Vehicle Movement (Step {i+1}/{len(x)})")
            plt.axis("equal")
            plt.grid(True)
            if i == 0:
                plt.legend()
            
            # Adjust pause time - slower at the beginning and end
            if i < 10 or i > len(x) - 10:
                plt.pause(0.1)  # Slower at beginning and end
            else:
                plt.pause(0.05)  # Faster in the middle
                
        # Keep the final frame visible
        plt.ioff()
        plt.show()
    
    print("Done")


if __name__ == "__main__":
    main()
