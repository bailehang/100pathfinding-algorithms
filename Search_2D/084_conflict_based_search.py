"""
CONFLICT_BASED_SEARCH_2D (CBS)
@author: huiming zhou (original code)
@author: clark bai
Based on: Sharon, Stern, Felner, Sturtevant (2012, 2015)
"""

import io
import os
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict


class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]
        return obs_cir


class Agent:
    def __init__(self, start, goal, agent_id):
        self.start = start
        self.goal = goal
        self.id = agent_id


class Constraint:
    def __init__(self, agent, location, time):
        self.agent = agent
        self.location = location
        self.time = time

    def __str__(self):
        return f"Agent {self.agent} cannot be at {self.location} at time {self.time}"


class Conflict:
    def __init__(self, agent1, agent2, location, time):
        self.agent1 = agent1
        self.agent2 = agent2
        self.location = location
        self.time = time

    def __str__(self):
        return f"Conflict between agents {self.agent1} and {self.agent2} at {self.location} at time {self.time}"


class CTNode:
    def __init__(self):
        self.constraints = []
        self.solution = {}
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost


class AStar:
    def __init__(self, env, cbs_instance=None):
        self.env = env
        self.motions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # Including wait action
        self.cbs = cbs_instance
        self.explored_nodes = set()

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid_position(self, pos):
        x, y = pos
        
        # Check boundaries
        if x < 1 or x >= 49 or y < 1 or y >= 29:
            return False
            
        # Check circle obstacles
        for (ox, oy, r) in self.env.obs_circle:
            if math.hypot(x - ox, y - oy) <= r + 0.5:
                return False
                
        # Check rectangle obstacles
        for (ox, oy, w, h) in self.env.obs_rectangle:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return False
                
        return True

    def plan(self, start, goal, constraints, agent_id=0, show_search=False):
        constraint_dict = defaultdict(set)
        for constraint in constraints:
            constraint_dict[constraint.time].add(constraint.location)

        open_list = [(self.heuristic(start, goal), 0, start, [start])]
        closed_set = set()
        self.explored_nodes = set()
        
        while open_list:
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path
                
            if (current, len(path) - 1) in closed_set:
                continue
                
            closed_set.add((current, len(path) - 1))
            self.explored_nodes.add(current)
            
            # Visualize search process if enabled
            if show_search and self.cbs and len(self.explored_nodes) % 20 == 0:
                self.cbs.plot_grid(f"A* Search for Agent {agent_id} - Exploring node {len(self.explored_nodes)}")
                self.cbs.plot_search_progress(agent_id, self.explored_nodes, current, goal)
                self.cbs.capture_frame()
            
            for dx, dy in self.motions:
                next_pos = (current[0] + dx, current[1] + dy)
                next_time = len(path)
                
                # Check if position is valid
                if not self.is_valid_position(next_pos):
                    continue
                    
                # Check constraints
                if next_pos in constraint_dict[next_time]:
                    continue
                    
                # Check edge constraints (for future enhancement)
                if (next_time > 0 and 
                    (current, next_pos) in constraint_dict.get(next_time, set())):
                    continue
                
                new_path = path + [next_pos]
                new_cost = cost + 1
                priority = new_cost + self.heuristic(next_pos, goal)
                
                heapq.heappush(open_list, (priority, new_cost, next_pos, new_path))
        
        return None  # No path found


class CBS:
    def __init__(self, agents):
        self.agents = agents
        self.env = Env()
        self.astar = AStar(self.env, self)
        self.fig_size = (12, 8)
        self.frames = []
        self.fig, self.ax = plt.subplots(figsize=self.fig_size, dpi=120)
        
        # Colors for different agents - optimized for 4 agents
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.current_conflicts = []
        self.tree_depth = 0

    def detect_conflicts(self, solution):
        conflicts = []
        
        for agent1_id in solution:
            for agent2_id in solution:
                if agent1_id >= agent2_id:
                    continue
                    
                path1 = solution[agent1_id]
                path2 = solution[agent2_id]
                
                # Check vertex conflicts
                max_len = max(len(path1), len(path2))
                for t in range(max_len):
                    pos1 = path1[min(t, len(path1) - 1)]
                    pos2 = path2[min(t, len(path2) - 1)]
                    
                    if pos1 == pos2:
                        conflicts.append(Conflict(agent1_id, agent2_id, pos1, t))
                        
                # Check edge conflicts
                min_len = min(len(path1), len(path2))
                for t in range(min_len - 1):
                    if (path1[t] == path2[t + 1] and path1[t + 1] == path2[t]):
                        conflicts.append(Conflict(agent1_id, agent2_id, (path1[t], path1[t + 1]), t))
        
        return conflicts

    def generate_child_nodes(self, node, conflict):
        children = []
        
        # Create constraint for agent 1
        child1 = CTNode()
        child1.constraints = node.constraints + [Constraint(conflict.agent1, conflict.location, conflict.time)]
        
        # Create constraint for agent 2  
        child2 = CTNode()
        child2.constraints = node.constraints + [Constraint(conflict.agent2, conflict.location, conflict.time)]
        
        children.append(child1)
        children.append(child2)
        
        return children

    def solve_low_level(self, node, show_search=False):
        solution = {}
        
        for agent in self.agents:
            # Get constraints for this agent
            agent_constraints = [c for c in node.constraints if c.agent == agent.id]
            
            # Plan path with constraints
            path = self.astar.plan(agent.start, agent.goal, agent_constraints, 
                                 agent.id, show_search)
            
            if path is None:
                return None
                
            solution[agent.id] = path
            
        return solution

    def planning(self):
        # Initialize root node
        root = CTNode()
        
        # Show initial setup without detailed A* search visualization
        print("Starting initial path planning for each agent...")
        self.plot_grid("CBS - Initial Setup with 8 Agents")
        self.plot_agents_state()
        self.capture_frame()
        
        # Solve initial paths without detailed visualization
        root.solution = self.solve_low_level(root, show_search=False)
        
        if root.solution is None:
            print("No initial solution found!")
            return None
            
        # Show initial solution
        self.plot_grid("CBS - Initial Solution (No Constraints)")
        self.plot_paths(root.solution)
        self.capture_frame()
            
        root.cost = sum(len(path) for path in root.solution.values())
        
        open_list = [root]
        iteration = 0
        self.tree_depth = 0
        
        while open_list and iteration < 30:  # Increased limit for 4 agents demo
            iteration += 1
            current = heapq.heappop(open_list)
            
            print(f"CBS Iteration {iteration}: Processing node with cost {current.cost}")
            
            # Detect conflicts first
            conflicts = self.detect_conflicts(current.solution)
            self.current_conflicts = conflicts
            
            if not conflicts:
                # Solution found
                print("Conflict-free solution found!")
                self.plot_grid("CBS - Final Solution Found!")
                self.plot_paths(current.solution)
                self.capture_frame()
                self.animate_solution(current.solution)
                self.save_animation_as_gif("084_conflict_based_search")
                plt.show()
                return current.solution
            
            # Visualize current solution with conflicts
            self.plot_grid(f"CBS - Iteration {iteration} (Found {len(conflicts)} conflicts)")
            self.plot_paths(current.solution)
            self.plot_conflicts(conflicts[:5])  # Show up to 5 conflicts for 4 agents
            self.capture_frame()
                
            # Choose first conflict to resolve
            conflict = conflicts[0]
            print(f"Resolving conflict between agents {conflict.agent1} and {conflict.agent2} at {conflict.location}")
            
            children = self.generate_child_nodes(current, conflict)
            self.tree_depth += 1
            
            valid_children = 0
            for i, child in enumerate(children):
                child.solution = self.solve_low_level(child, show_search=False)
                
                if child.solution is not None:
                    child.cost = sum(len(path) for path in child.solution.values())
                    heapq.heappush(open_list, child)
                    valid_children += 1
                    
                    # Show new solution occasionally
                    if iteration % 3 == 0:  # Show every 3rd iteration
                        self.plot_grid(f"New Solution - Constraint {i+1} Applied")
                        self.plot_paths(child.solution)
                        self.plot_constraints([child.constraints[-1]])
                        self.capture_frame()
            
            print(f"Generated {valid_children} valid child nodes")
            
            if not open_list:
                print("No more nodes to explore - no solution found")
                break
        
        print("Reached iteration limit or no solution found")
        return None

    def plot_grid(self, name):
        self.ax.clear()
        
        # Plot obstacles
        for (ox, oy, w, h) in self.env.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.env.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.env.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    facecolor='gray',
                    fill=True
                )
            )

        # Plot start and goal positions
        for i, agent in enumerate(self.agents):
            color = self.colors[i % len(self.colors)]
            plt.plot(agent.start[0], agent.start[1], 's', color=color, markersize=10, 
                    label=f'Agent {agent.id} Start', markeredgecolor='black', markeredgewidth=1)
            plt.plot(agent.goal[0], agent.goal[1], '^', color=color, markersize=10, 
                    label=f'Agent {agent.id} Goal', markeredgecolor='black', markeredgewidth=1)

        plt.title(name, fontsize=14, fontweight='bold')
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

    def plot_agents_state(self):
        """Plot just the agents' start and goal positions"""
        for i, agent in enumerate(self.agents):
            color = self.colors[i % len(self.colors)]
            plt.plot(agent.start[0], agent.start[1], 's', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1)
            plt.plot(agent.goal[0], agent.goal[1], '^', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1)

    def plot_search_progress(self, agent_id, explored_nodes, current_node, goal):
        """Visualize A* search progress"""
        color = self.colors[agent_id % len(self.colors)]
        
        # Plot explored nodes
        for node in explored_nodes:
            plt.plot(node[0], node[1], 'o', color=color, markersize=2, alpha=0.3)
        
        # Highlight current node
        plt.plot(current_node[0], current_node[1], 'o', color=color, markersize=6, 
                markeredgecolor='black', markeredgewidth=1)
        
        # Draw line to goal
        plt.plot([current_node[0], goal[0]], [current_node[1], goal[1]], 
                '--', color=color, alpha=0.5, linewidth=1)

    def plot_paths(self, solution):
        if not solution:
            return
            
        for agent_id, path in solution.items():
            color = self.colors[agent_id % len(self.colors)]
            
            # Plot path
            x_coords = [pos[0] for pos in path]
            y_coords = [pos[1] for pos in path]
            plt.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.8,
                    label=f'Agent {agent_id} Path')
            
            # Plot waypoints
            for i, pos in enumerate(path):
                plt.plot(pos[0], pos[1], 'o', color=color, markersize=4, alpha=0.6)
                # Add time labels for first few waypoints
                if i < 5:
                    plt.text(pos[0], pos[1], str(i), fontsize=8, ha='center', va='center')

    def plot_conflicts(self, conflicts):
        """Visualize conflicts between agents"""
        for conflict in conflicts:
            if isinstance(conflict.location, tuple) and len(conflict.location) == 2:
                # Vertex conflict
                x, y = conflict.location
                plt.plot(x, y, 'X', color='red', markersize=15, markeredgecolor='black', 
                        markeredgewidth=2, label=f'Conflict at t={conflict.time}')
                # Add conflict text
                plt.text(x, y-1, f'Conflict\nt={conflict.time}', fontsize=10, ha='center', 
                        va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))

    def plot_constraints(self, constraints):
        """Visualize constraints"""
        for constraint in constraints:
            if isinstance(constraint.location, tuple) and len(constraint.location) == 2:
                x, y = constraint.location
                color = self.colors[constraint.agent % len(self.colors)]
                plt.plot(x, y, 'D', color=color, markersize=8, markeredgecolor='red', 
                        markeredgewidth=2, alpha=0.7)
                plt.text(x, y+1, f'Constraint\nAgent {constraint.agent}\nt={constraint.time}', 
                        fontsize=8, ha='center', va='bottom', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))

    def animate_solution(self, solution):
        """Create animation showing agents moving along their paths"""
        if not solution:
            return
        
        # Find maximum path length
        max_length = max(len(path) for path in solution.values())
        
        # Reduce animation frames for performance - show every 2nd time step
        for t in range(0, max_length, 2):
            self.plot_grid(f"Solution Animation - Time Step {t}")
            
            # Plot complete paths
            for agent_id, path in solution.items():
                color = self.colors[agent_id % len(self.colors)]
                x_coords = [pos[0] for pos in path]
                y_coords = [pos[1] for pos in path]
                plt.plot(x_coords, y_coords, '-', color=color, linewidth=2, alpha=0.3)
            
            # Plot current positions
            for agent_id, path in solution.items():
                color = self.colors[agent_id % len(self.colors)]
                current_pos = path[min(t, len(path) - 1)]
                plt.plot(current_pos[0], current_pos[1], 'o', color=color, markersize=12,
                        markeredgecolor='black', markeredgewidth=2,
                        label=f'Agent {agent_id} at t={t}')
                
                # Show trajectory up to current time
                if t > 0:
                    traj_x = [path[min(i, len(path) - 1)][0] for i in range(t + 1)]
                    traj_y = [path[min(i, len(path) - 1)][1] for i in range(t + 1)]
                    plt.plot(traj_x, traj_y, '-', color=color, linewidth=4, alpha=0.8)
            
            self.capture_frame()

    def save_animation_as_gif(self, name, fps=2):
        if not os.path.exists('gif'):
            os.makedirs('gif')
        
        gif_path = os.path.join('gif', f"{name}.gif")
        if self.frames:
            print(f"Saving {len(self.frames)} frames to {gif_path}")
            # Optimize GIF creation
            self.frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            print(f"GIF saved successfully with {len(self.frames)} frames")

    def capture_frame(self):
        buf = io.BytesIO()
        # Reduce DPI for smaller file size and faster processing
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
        buf.seek(0)
        # Copy the image data before closing the buffer
        img = Image.open(buf).copy()
        self.frames.append(img)
        buf.close()


def main():
    # Define agents with start and goal positions - 4 agents for optimal demo
    agents = [
        Agent((5, 5), (45, 25), 0),      # Bottom-left to top-right
        Agent((5, 25), (45, 5), 1),      # Top-left to bottom-right
        Agent((25, 2), (25, 27), 2),     # Bottom-center to top-center
        Agent((45, 15), (5, 15), 3)      # Right-center to left-center
    ]
    
    print("Starting Conflict-Based Search (CBS) algorithm...")
    print(f"Number of agents: {len(agents)}")
    for agent in agents:
        print(f"Agent {agent.id}: Start {agent.start} -> Goal {agent.goal}")
    
    cbs = CBS(agents)
    solution = cbs.planning()
    
    if solution is None:
        print("No solution found!")
    else:
        print("Solution found!")
        total_cost = 0
        for agent_id, path in solution.items():
            path_length = len(path)
            total_cost += path_length
            print(f"Agent {agent_id} path length: {path_length}")
            print(f"Agent {agent_id} path: {path[:5]}{'...' if len(path) > 5 else ''}")
        print(f"Total cost: {total_cost}")


if __name__ == '__main__':
    main() 