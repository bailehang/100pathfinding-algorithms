"""
Multi-Agent Theta* 2D: Cooperative path planning for multiple agents
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import time
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class MultiAgentThetaStar:
    """
    Multi-Agent Theta*: Cooperative path planning for multiple agents
    
    This algorithm extends Theta* to handle multiple agents navigating through
    the same environment while avoiding collisions with obstacles and each other.
    """
    def __init__(self, agent_starts, agent_goals, heuristic_type="euclidean", 
                 collision_radius=1.0, priority_method="distance_to_goal"):
        """
        Initialize the Multi-Agent Theta* planner
        
        Args:
            agent_starts: List of start positions for each agent
            agent_goals: List of goal positions for each agent
            heuristic_type: Heuristic function type
            collision_radius: Minimum distance between agents to avoid collisions
            priority_method: Method to determine agent planning priority
        """
        self.agent_starts = agent_starts
        self.agent_goals = agent_goals
        self.num_agents = len(agent_starts)
        self.heuristic_type = heuristic_type
        self.collision_radius = collision_radius
        self.priority_method = priority_method
        
        # Environment setup
        self.Env = env.Env()  # class Env
        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs.copy()  # position of obstacles
        
        # Agent colors for visualization
        self.agent_colors = self.generate_agent_colors(self.num_agents)
        
        # Agent paths and data
        self.paths = [[] for _ in range(self.num_agents)]
        self.path_costs = [0] * self.num_agents
        self.priorities = self.calculate_agent_priorities()
        
        # Time-indexed reservations for coordination
        self.reserved_positions = {}  # {time_step: [(x, y, agent_id), ...]}
        
        # Visualization settings
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.plot = plotting.Plotting(agent_starts[0], agent_goals[0])  # Will be updated per agent
        
        # Animation frames
        self.animation_frames = []
        
        # Stats
        self.planning_time = 0
        self.collision_checks = 0
        self.search_nodes_expanded = 0

    def plan_paths(self):
        """
        Plan paths for all agents using prioritized planning
        """
        start_time = time.time()
        
        # Sort agents by priority
        agent_indices = sorted(range(self.num_agents), key=lambda i: self.priorities[i])
        
        for idx in agent_indices:
            print(f"Planning path for agent {idx} (priority: {self.priorities[idx]:.2f})")
            
            # Setup agent-specific planning environment
            dynamic_obstacles = self.create_dynamic_obstacles(idx)
            
            # Plan using Theta* for this agent
            path, expanded_nodes = self.theta_star_plan(
                self.agent_starts[idx], 
                self.agent_goals[idx], 
                dynamic_obstacles, 
                idx
            )
            
            if path:
                self.search_nodes_expanded += len(expanded_nodes)
                self.paths[idx] = path
                self.path_costs[idx] = self.calculate_path_cost(path)
                
                # Update reservations based on new path
                self.update_reserved_positions(path, idx)
            else:
                print(f"WARNING: No path found for agent {idx}")
                self.paths[idx] = [self.agent_starts[idx]]  # Stay at start position
        
        self.planning_time = time.time() - start_time
        
        # Create visualization
        self.visualize_paths()
        
        return self.paths, self.path_costs

    def calculate_agent_priorities(self):
        """
        Calculate priorities for agents based on the selected method
        """
        priorities = []
        
        for i in range(self.num_agents):
            if self.priority_method == "distance_to_goal":
                # Shorter distance to goal = higher priority
                distance = math.hypot(
                    self.agent_goals[i][0] - self.agent_starts[i][0],
                    self.agent_goals[i][1] - self.agent_starts[i][1]
                )
                priorities.append(-distance)  # Negate for sorting (higher value = higher priority)
                
            elif self.priority_method == "random":
                priorities.append(random.random())
                
            elif self.priority_method == "sequential":
                priorities.append(-i)  # First agent gets highest priority
                
            else:  # Default: equal priority
                priorities.append(0)
                
        return priorities

    def create_dynamic_obstacles(self, agent_idx):
        """
        Create dynamic obstacles based on paths of higher-priority agents
        
        Returns: dict mapping time step to list of positions that are blocked
        """
        dynamic_obstacles = {}
        
        for t in self.reserved_positions:
            positions = []
            for pos_x, pos_y, other_idx in self.reserved_positions[t]:
                if other_idx != agent_idx:  # Avoid self-collision
                    positions.append((pos_x, pos_y))
            
            if positions:
                dynamic_obstacles[t] = positions
        
        return dynamic_obstacles

    def update_reserved_positions(self, path, agent_idx):
        """
        Update the reserved positions based on an agent's path
        """
        for t, pos in enumerate(path):
            if t not in self.reserved_positions:
                self.reserved_positions[t] = []
            
            self.reserved_positions[t].append((pos[0], pos[1], agent_idx))
            
        # For the final position, reserve it indefinitely
        # This prevents other agents from planning paths through the goal of this agent
        final_pos = path[-1]
        for t in range(len(path), len(path) + 100):  # Reserve for a long time
            if t not in self.reserved_positions:
                self.reserved_positions[t] = []
            self.reserved_positions[t].append((final_pos[0], final_pos[1], agent_idx))

    def theta_star_plan(self, start, goal, dynamic_obstacles, agent_idx):
        """
        Plan a path for a single agent using Theta*
        
        Args:
            start: Start position
            goal: Goal position
            dynamic_obstacles: Time-indexed positions to avoid
            agent_idx: Agent index
            
        Returns:
            path: List of positions
            visited: List of nodes expanded during search
        """
        # Initialize data structures
        open_set = []
        closed_set = set()
        g_costs = {start: 0}
        f_costs = {start: self.heuristic(start, goal)}
        parent = {start: start}
        time_at_node = {start: 0}  # Track the time step at each node
        
        heapq.heappush(open_set, (f_costs[start], start))
        
        visited = []  # Track visited nodes for visualization
        
        while open_set:
            _, current = heapq.heappop(open_set)
            current_time = time_at_node[current]
            
            visited.append(current)
            
            if current == goal:
                path = self.extract_path(parent, current)
                return path, visited
                
            closed_set.add(current)
            
            # Check all neighbors
            for motion in self.u_set:
                neighbor = (current[0] + motion[0], current[1] + motion[1])
                
                # Skip if outside grid or in static obstacle
                if not self.is_valid_position(neighbor):
                    continue
                
                # Time at this neighbor
                neighbor_time = current_time + 1
                
                # Check for collisions with dynamic obstacles at this time
                if self.collides_with_dynamic_obstacle(neighbor, neighbor_time, dynamic_obstacles):
                    continue
                
                # Try for line-of-sight from parent node (Theta* improvement)
                if parent[current] != current:  # Not the start node
                    # Check line-of-sight considering dynamic obstacles
                    if self.line_of_sight(parent[current], neighbor, time_at_node[parent[current]], 
                                         neighbor_time, dynamic_obstacles):
                        # Path 2: Direct connection to parent's parent
                        new_g = g_costs[parent[current]] + self.cost(parent[current], neighbor)
                        
                        if neighbor not in g_costs or new_g < g_costs[neighbor]:
                            g_costs[neighbor] = new_g
                            parent[neighbor] = parent[current]
                            time_at_node[neighbor] = neighbor_time
                            f_value = g_costs[neighbor] + self.heuristic(neighbor, goal)
                            f_costs[neighbor] = f_value
                            
                            if neighbor not in closed_set:
                                heapq.heappush(open_set, (f_value, neighbor))
                        
                        continue
                
                # Path 1: Regular A* path through current node
                new_g = g_costs[current] + self.cost(current, neighbor)
                
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    parent[neighbor] = current
                    time_at_node[neighbor] = neighbor_time
                    f_value = g_costs[neighbor] + self.heuristic(neighbor, goal)
                    f_costs[neighbor] = f_value
                    
                    if neighbor not in closed_set:
                        heapq.heappush(open_set, (f_value, neighbor))
        
        # No path found
        return [], visited

    def is_valid_position(self, pos):
        """
        Check if a position is valid (within grid and not in obstacles)
        """
        x, y = pos
        
        # Check grid boundaries
        if x < 0 or x >= self.Env.x_range or y < 0 or y >= self.Env.y_range:
            return False
            
        # Check static obstacles
        if pos in self.obs:
            return False
            
        return True

    def collides_with_dynamic_obstacle(self, pos, time_step, dynamic_obstacles):
        """
        Check if a position collides with a dynamic obstacle at a given time
        """
        if time_step not in dynamic_obstacles:
            return False
            
        for obs_pos in dynamic_obstacles[time_step]:
            dist = math.hypot(pos[0] - obs_pos[0], pos[1] - obs_pos[1])
            if dist < self.collision_radius:
                self.collision_checks += 1
                return True
                
        return False

    def line_of_sight(self, start, end, start_time, end_time, dynamic_obstacles):
        """
        Check if there is line-of-sight between two positions, considering dynamic obstacles
        """
        # First check for static obstacles using Bresenham's line algorithm
        if self.is_collision(start, end):
            return False
            
        # Then check for dynamic obstacles at intermediate points
        # Approximate the number of steps based on manhattan distance
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        steps = max(dx, dy)
        
        if steps <= 1:  # Adjacent cells
            return True
            
        # Check intermediate points for dynamic obstacles
        for i in range(1, steps):
            t = i / steps  # Interpolation parameter
            
            # Interpolate position
            x = int(round(start[0] + t * (end[0] - start[0])))
            y = int(round(start[1] + t * (end[1] - start[1])))
            
            # Interpolate time
            time_step = int(round(start_time + t * (end_time - start_time)))
            
            if self.collides_with_dynamic_obstacle((x, y), time_step, dynamic_obstacles):
                return False
                
        return True

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment between two positions passes through obstacles
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

    def extract_path(self, parent, current):
        """
        Extract path from parent dictionary
        """
        path = [current]
        while current != parent[current]:
            current = parent[current]
            path.append(current)
            
        return list(reversed(path))

    def heuristic(self, a, b):
        """
        Calculate heuristic distance
        """
        if self.heuristic_type == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:  # Default: Euclidean
            return math.hypot(a[0] - b[0], a[1] - b[1])

    def cost(self, a, b):
        """
        Calculate cost between two adjacent positions
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def calculate_path_cost(self, path):
        """
        Calculate the total cost of a path
        """
        if not path or len(path) < 2:
            return 0
            
        cost = 0
        for i in range(len(path) - 1):
            cost += self.cost(path[i], path[i + 1])
            
        return cost

    def generate_agent_colors(self, num_agents):
        """
        Generate distinct colors for each agent
        """
        # Use a predefined set of colors for better visibility
        base_colors = list(mcolors.TABLEAU_COLORS)
        
        # If we need more colors than available, create additional ones
        if num_agents > len(base_colors):
            # Add more colors by cycling through hue values
            for i in range(len(base_colors), num_agents):
                h = i / num_agents
                s = 0.8
                v = 0.9
                rgb = plt.cm.hsv(h)
                base_colors.append(mcolors.rgb2hex(rgb))
                
        return base_colors[:num_agents]

    def visualize_paths(self):
        """
        Create visualization of all agent paths
        """
        self.animation_frames = []
        
        # Determine the maximum path length
        max_path_length = max([len(path) for path in self.paths])
        
        # Create frames for each time step
        for t in range(max_path_length):
            # Create a copy of the environment
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Draw the grid and obstacles
            self.plot.plot_grid("Multi-Agent Theta*")
            
            # Draw paths up to current time for each agent
            for agent_idx in range(self.num_agents):
                agent_path = self.paths[agent_idx]
                color = self.agent_colors[agent_idx]
                
                # Draw the complete path (translucent)
                if len(agent_path) >= 2:
                    xs = [pos[0] for pos in agent_path]
                    ys = [pos[1] for pos in agent_path]
                    plt.plot(xs, ys, color=color, alpha=0.3, linewidth=2)
                
                # Draw the current position
                current_pos_idx = min(t, len(agent_path) - 1)
                current_pos = agent_path[current_pos_idx]
                
                circle = plt.Circle((current_pos[0], current_pos[1]), 
                                  radius=0.5, 
                                  color=color, 
                                  alpha=0.8)
                ax.add_patch(circle)
                
                # Add agent index label
                plt.text(current_pos[0], current_pos[1], str(agent_idx),
                       color='white', ha='center', va='center',
                       fontsize=8, fontweight='bold')
                
                # Draw start and goal markers
                plt.plot(self.agent_starts[agent_idx][0], self.agent_starts[agent_idx][1],
                       color=color, marker='o', markersize=8)
                plt.plot(self.agent_goals[agent_idx][0], self.agent_goals[agent_idx][1],
                       color=color, marker='*', markersize=12)
            
            # Add time step indicator
            plt.title(f"Multi-Agent Theta* (Time: {t})")
            
            # Save this frame
            self.animation_frames.append(fig)
            plt.close(fig)
        
        # Display the first frame
        plt.figure(figsize=(10, 8))
        self.display_solution_stats()
        
        # Create animation
        self.animate_paths()

    def animate_paths(self):
        """
        Create and display animation of agent paths
        """
        # Create a new figure for the animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Draw the grid and obstacles
        self.plot.plot_grid("Multi-Agent Theta*")
        
        # Initialize agent position markers
        agent_circles = []
        for agent_idx in range(self.num_agents):
            circle = plt.Circle((self.agent_starts[agent_idx][0], self.agent_starts[agent_idx][1]), 
                              radius=0.5, 
                              color=self.agent_colors[agent_idx], 
                              alpha=0.8)
            agent_circles.append(ax.add_patch(circle))
            
            # Add agent index label
            label = plt.text(self.agent_starts[agent_idx][0], self.agent_starts[agent_idx][1], 
                           str(agent_idx),
                           color='white', ha='center', va='center',
                           fontsize=8, fontweight='bold')
            agent_circles.append(label)
            
            # Draw start and goal markers
            plt.plot(self.agent_starts[agent_idx][0], self.agent_starts[agent_idx][1],
                   color=self.agent_colors[agent_idx], marker='o', markersize=8)
            plt.plot(self.agent_goals[agent_idx][0], self.agent_goals[agent_idx][1],
                   color=self.agent_colors[agent_idx], marker='*', markersize=12)
            
            # Draw the complete path (translucent)
            if len(self.paths[agent_idx]) >= 2:
                xs = [pos[0] for pos in self.paths[agent_idx]]
                ys = [pos[1] for pos in self.paths[agent_idx]]
                plt.plot(xs, ys, color=self.agent_colors[agent_idx], alpha=0.3, linewidth=2)
        
        # Add title with time indicator
        title = ax.set_title("Multi-Agent Theta* (Time: 0)")
        
        def update_frame(frame):
            # Update the time in the title
            title.set_text(f"Multi-Agent Theta* (Time: {frame})")
            
            # Update agent positions
            circle_idx = 0
            for agent_idx in range(self.num_agents):
                agent_path = self.paths[agent_idx]
                
                # Get current position
                current_pos_idx = min(frame, len(agent_path) - 1)
                current_pos = agent_path[current_pos_idx]
                
                # Update circle position
                agent_circles[circle_idx].center = current_pos
                circle_idx += 1
                
                # Update label position
                agent_circles[circle_idx].set_position((current_pos[0], current_pos[1]))
                circle_idx += 1
                
            return agent_circles + [title]
        
        # Determine the maximum path length
        max_path_length = max([len(path) for path in self.paths])
        
        # Create animation
        ani = FuncAnimation(fig, update_frame, frames=range(max_path_length),
                          blit=True, interval=200, repeat=True)
        
        # Display animation
        plt.tight_layout()
        plt.show()

    def display_solution_stats(self):
        """
        Display statistics about the solution
        """
        print("\nMulti-Agent Theta* Solution Statistics:")
        print(f"Number of agents: {self.num_agents}")
        print(f"Total planning time: {self.planning_time:.4f} seconds")
        print(f"Total nodes expanded: {self.search_nodes_expanded}")
        print(f"Collision checks performed: {self.collision_checks}")
        
        # Path statistics
        path_lengths = [len(path) for path in self.paths]
        print(f"Average path length: {sum(path_lengths) / self.num_agents:.2f} steps")
        print(f"Average path cost: {sum(self.path_costs) / self.num_agents:.2f}")
        
        # Check for path existence
        paths_found = sum(1 for path in self.paths if len(path) > 1)
        print(f"Paths found: {paths_found}/{self.num_agents}")
        
        # Calculate total solution cost
        total_cost = sum(self.path_costs)
        makespan = max(path_lengths)
        print(f"Solution total cost: {total_cost:.2f}")
        print(f"Solution makespan: {makespan} time steps")


def main():
    """
    Multi-Agent Theta* Demo with 5 agents
    
    This algorithm extends Theta* to handle multiple agents pathfinding
    through a shared environment. It uses prioritized planning to coordinate
    agents and prevent collisions.
    
    Features:
    1. Agent prioritization based on configurable metrics
    2. Theta* pathfinding for each agent 
    3. Dynamic obstacle avoidance between agents
    4. Temporal coordination to prevent collisions
    5. Visualization of all agent paths and interactions
    """
    # Create sample agent starts and goals
    agent_starts = [
        (5, 5),    # Agent 0
        (15, 5),   # Agent 1
        (25, 5),   # Agent 2
        (35, 5),   # Agent 3
        (45, 5),   # Agent 4
    ]
    
    agent_goals = [
        (45, 25),  # Agent 0
        (35, 25),  # Agent 1
        (25, 25),  # Agent 2
        (15, 25),  # Agent 3
        (5, 25),   # Agent 4
    ]
    
    # Create Multi-Agent Theta* planner
    planner = MultiAgentThetaStar(
        agent_starts=agent_starts,
        agent_goals=agent_goals,
        heuristic_type="euclidean",
        collision_radius=1.0,
        priority_method="distance_to_goal"  # Options: distance_to_goal, random, sequential
    )
    
    # Plan paths for all agents
    paths, costs = planner.plan_paths()
    
    # Output solution quality
    for i in range(len(paths)):
        print(f"Agent {i} path cost: {costs[i]:.2f}")


if __name__ == '__main__':
    main()
