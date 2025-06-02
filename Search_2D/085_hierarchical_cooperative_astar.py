"""
HIERARCHICAL_COOPERATIVE_A_STAR (HCA*)
@author: clark bai
Based on: Silver (2005) - Cooperative Pathfinding

ALGORITHM OVERVIEW:
===================
Hierarchical Cooperative A* (HCA*) is a multi-agent pathfinding algorithm that uses 
abstraction to efficiently plan paths for multiple agents in a coordinated manner.

KEY CONCEPTS:
1. Hierarchical Planning: Uses two levels of abstraction
   - Abstract Level: Coarse-grained representation using clusters
   - Concrete Level: Fine-grained paths within the original environment

2. Cooperative Planning: Agents plan sequentially based on priority
   - Higher priority agents plan first
   - Lower priority agents avoid conflicts using reservation tables

3. Abstract Graph: Environment is divided into clusters
   - Each cluster becomes an abstract node if it contains sufficient free space
   - Abstract edges connect neighboring clusters with valid paths

ALGORITHM PHASES:
=================
Phase 1: Abstract Planning
- Create abstract graph by clustering the environment
- Plan abstract paths for each agent in the abstract space
- Abstract paths provide high-level guidance

Phase 2: Concrete Planning
- Convert abstract paths to concrete paths segment by segment
- Use reservation tables to avoid conflicts between agents
- Higher priority agents reserve space first

ADVANTAGES:
===========
- Efficient for large environments due to abstraction
- Scalable with number of agents through hierarchical decomposition
- Guarantees conflict-free paths through cooperative planning
- Balances optimality with computational efficiency

IMPLEMENTATION FEATURES:
========================
- Real-time visualization of abstract graph construction
- Step-by-step visualization of planning phases
- Agent priority system for conflict resolution
- Comprehensive animation of final solution
- GIF export for analysis and presentation
"""

import io
import os
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict, deque


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
    def __init__(self, start, goal, agent_id, priority=0):
        self.start = start
        self.goal = goal
        self.id = agent_id
        self.priority = priority  # Higher priority agents plan first
        self.abstract_path = []
        self.concrete_path = []
        self.reservation_table = {}  # Time -> position reservations


class AbstractGraph:
    """Creates an abstract representation of the environment"""
    
    def __init__(self, env, cluster_size=4):
        self.env = env
        self.cluster_size = cluster_size
        self.width = env.x_range[1]
        self.height = env.y_range[1]
        self.abstract_width = math.ceil(self.width / cluster_size)
        self.abstract_height = math.ceil(self.height / cluster_size)
        self.abstract_nodes = {}
        self.abstract_edges = {}
        self.build_abstract_graph()
    
    def build_abstract_graph(self):
        """Build the abstract graph by clustering the environment"""
        # Create abstract nodes
        for ax in range(self.abstract_width):
            for ay in range(self.abstract_height):
                if self.is_abstract_node_valid(ax, ay):
                    self.abstract_nodes[(ax, ay)] = self.get_cluster_center(ax, ay)
        
        # Create abstract edges between adjacent valid nodes
        for (ax, ay) in self.abstract_nodes:
            neighbors = [(ax+1, ay), (ax-1, ay), (ax, ay+1), (ax, ay-1)]
            self.abstract_edges[(ax, ay)] = []
            for (nx, ny) in neighbors:
                if (nx, ny) in self.abstract_nodes:
                    if self.is_abstract_edge_valid((ax, ay), (nx, ny)):
                        self.abstract_edges[(ax, ay)].append((nx, ny))
    
    def is_abstract_node_valid(self, ax, ay):
        """Check if an abstract node represents a valid cluster"""
        # Get the concrete coordinates of this cluster
        start_x = ax * self.cluster_size
        start_y = ay * self.cluster_size
        end_x = min(start_x + self.cluster_size, self.width)
        end_y = min(start_y + self.cluster_size, self.height)
        
        # Check if at least 50% of the cluster is free space
        total_cells = (end_x - start_x) * (end_y - start_y)
        free_cells = 0
        
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                if self.is_concrete_position_valid(x, y):
                    free_cells += 1
        
        return free_cells >= total_cells * 0.5
    
    def is_concrete_position_valid(self, x, y):
        """Check if a concrete position is valid (not in obstacle)"""
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
    
    def is_abstract_edge_valid(self, node1, node2):
        """Check if there's a valid path between two abstract nodes"""
        center1 = self.get_cluster_center(*node1)
        center2 = self.get_cluster_center(*node2)
        
        # Simple line-of-sight check between cluster centers
        return self.line_of_sight(center1, center2)
    
    def line_of_sight(self, pos1, pos2):
        """Check if there's a clear line of sight between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            if not self.is_concrete_position_valid(int(x), int(y)):
                return False
                
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return True
    
    def get_cluster_center(self, ax, ay):
        """Get the center coordinates of an abstract cluster"""
        start_x = ax * self.cluster_size
        start_y = ay * self.cluster_size
        end_x = min(start_x + self.cluster_size, self.width)
        end_y = min(start_y + self.cluster_size, self.height)
        
        center_x = (start_x + end_x) // 2
        center_y = (start_y + end_y) // 2
        
        # Find the nearest valid position to the center
        for radius in range(self.cluster_size):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    test_x = center_x + dx
                    test_y = center_y + dy
                    if (start_x <= test_x < end_x and start_y <= test_y < end_y and 
                        self.is_concrete_position_valid(test_x, test_y)):
                        return (test_x, test_y)
        
        return (center_x, center_y)  # Fallback
    
    def get_abstract_position(self, concrete_pos):
        """Convert concrete position to abstract position"""
        x, y = concrete_pos
        ax = x // self.cluster_size
        ay = y // self.cluster_size
        return (ax, ay)


class AStar:
    def __init__(self, env, abstract_graph=None):
        self.env = env
        self.abstract_graph = abstract_graph
        self.motions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # Including wait action
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

    def plan_abstract(self, start_abstract, goal_abstract):
        """Plan a path in abstract space"""
        if not self.abstract_graph:
            return None
        
        if start_abstract not in self.abstract_graph.abstract_nodes:
            return None
        if goal_abstract not in self.abstract_graph.abstract_nodes:
            return None
        
        open_list = [(self.heuristic(start_abstract, goal_abstract), 0, start_abstract, [start_abstract])]
        closed_set = set()
        
        while open_list:
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal_abstract:
                return path
                
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            for neighbor in self.abstract_graph.abstract_edges.get(current, []):
                if neighbor not in closed_set:
                    new_path = path + [neighbor]
                    new_cost = cost + 1
                    priority = new_cost + self.heuristic(neighbor, goal_abstract)
                    heapq.heappush(open_list, (priority, new_cost, neighbor, new_path))
        
        return None

    def plan_concrete(self, start, goal, reservation_table=None, time_offset=0):
        """Plan a concrete path considering reservations from other agents"""
        if reservation_table is None:
            reservation_table = {}
        
        open_list = [(self.heuristic(start, goal), 0, start, [start])]
        closed_set = set()
        self.explored_nodes = set()
        
        while open_list:
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path
                
            current_time = len(path) - 1 + time_offset
            if (current, current_time) in closed_set:
                continue
                
            closed_set.add((current, current_time))
            self.explored_nodes.add(current)
            
            for dx, dy in self.motions:
                next_pos = (current[0] + dx, current[1] + dy)
                next_time = current_time + 1
                
                # Check if position is valid
                if not self.is_valid_position(next_pos):
                    continue
                
                # Check vertex conflicts (position reservations)
                if next_time in reservation_table:
                    if next_pos in reservation_table[next_time]:
                        continue
                
                # Check edge conflicts (swapping positions)
                edge_conflict = False
                if current_time in reservation_table and next_time in reservation_table:
                    # Check if another agent is moving from next_pos to current at the same time
                    if (next_pos in reservation_table[current_time] and 
                        current in reservation_table[next_time]):
                        edge_conflict = True
                
                # Check for head-on collision (two agents moving towards each other)
                if 'edges' in reservation_table:
                    reverse_edge = (next_pos, current)
                    if (next_time in reservation_table['edges'] and 
                        reverse_edge in reservation_table['edges'][next_time]):
                        edge_conflict = True
                
                if edge_conflict:
                    continue
                
                # Additional check: ensure we don't conflict with agents staying at next_pos
                if (current_time in reservation_table and 
                    next_pos in reservation_table[current_time] and 
                    next_time in reservation_table and 
                    next_pos in reservation_table[next_time]):
                    continue
                
                new_path = path + [next_pos]
                new_cost = cost + 1
                priority = new_cost + self.heuristic(next_pos, goal)
                
                heapq.heappush(open_list, (priority, new_cost, next_pos, new_path))
        
        return None


class HCA:
    """Hierarchical Cooperative A* implementation"""
    
    def __init__(self, agents):
        self.agents = sorted(agents, key=lambda a: a.priority, reverse=True)  # Sort by priority
        self.env = Env()
        self.abstract_graph = AbstractGraph(self.env, cluster_size=6)
        self.astar = AStar(self.env, self.abstract_graph)
        self.global_reservation_table = defaultdict(set)
        
        # Visualization
        self.fig_size = (12, 8)
        self.frames = []
        self.fig, self.ax = plt.subplots(figsize=self.fig_size, dpi=120)
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.planning_phase = "initialization"

    def detect_conflicts_in_solution(self, solution):
        """Detect any conflicts in the final solution"""
        conflicts = []
        
        # Get maximum path length
        max_length = max(len(path) for path in solution.values()) if solution else 0
        
        for t in range(max_length):
            # Check vertex conflicts at time t
            positions_at_time = {}
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t]
                else:
                    pos = path[-1]  # Agent stays at goal
                
                if pos in positions_at_time:
                    conflicts.append(f"Vertex conflict at time {t}: Agents {positions_at_time[pos]} and {agent_id} both at {pos}")
                else:
                    positions_at_time[pos] = agent_id
            
            # Check edge conflicts (swapping) at time t
            if t > 0:
                for agent1_id, path1 in solution.items():
                    for agent2_id, path2 in solution.items():
                        if agent1_id >= agent2_id:
                            continue
                        
                        # Get positions at t-1 and t
                        pos1_prev = path1[min(t-1, len(path1)-1)]
                        pos1_curr = path1[min(t, len(path1)-1)]
                        pos2_prev = path2[min(t-1, len(path2)-1)]
                        pos2_curr = path2[min(t, len(path2)-1)]
                        
                        # Check if agents are swapping positions
                        if pos1_prev == pos2_curr and pos1_curr == pos2_prev and pos1_prev != pos1_curr:
                            conflicts.append(f"Edge conflict at time {t}: Agents {agent1_id} and {agent2_id} swapping positions {pos1_prev} <-> {pos1_curr}")
        
        return conflicts

    def solve(self):
        """Main solving function implementing HCA*"""
        print("Starting Hierarchical Cooperative A* (HCA*) algorithm...")
        print(f"Number of agents: {len(self.agents)}")
        for agent in self.agents:
            print(f"Agent {agent.id} (Priority {agent.priority}): Start {agent.start} -> Goal {agent.goal}")
        
        # Phase 1: Abstract planning
        self.planning_phase = "abstract"
        print("\nPhase 1: Abstract Planning")
        self.plot_state("HCA* - Phase 1: Abstract Planning")
        self.plot_abstract_graph()
        self.capture_frame()
        
        for agent in self.agents:
            start_abstract = self.abstract_graph.get_abstract_position(agent.start)
            goal_abstract = self.abstract_graph.get_abstract_position(agent.goal)
            
            print(f"Planning abstract path for Agent {agent.id}")
            abstract_path = self.astar.plan_abstract(start_abstract, goal_abstract)
            
            if abstract_path is None:
                print(f"No abstract path found for Agent {agent.id}")
                return None
            
            agent.abstract_path = abstract_path
            print(f"Agent {agent.id} abstract path: {abstract_path}")
            
            # Visualize abstract path
            self.plot_state(f"HCA* - Agent {agent.id} Abstract Path")
            self.plot_abstract_graph()
            self.plot_abstract_path(agent)
            self.capture_frame()
        
        # Phase 2: Concrete planning with cooperation
        self.planning_phase = "concrete"
        print("\nPhase 2: Concrete Planning with Cooperation")
        
        for agent in self.agents:
            print(f"Planning concrete path for Agent {agent.id} (Priority {agent.priority})")
            
            # Plan concrete path for each segment of the abstract path
            concrete_path = []
            current_pos = agent.start
            current_time = 0
            
            for i in range(len(agent.abstract_path)):
                abstract_node = agent.abstract_path[i]
                target_pos = self.abstract_graph.abstract_nodes[abstract_node]
                
                if i == len(agent.abstract_path) - 1:
                    # Last segment - go to actual goal
                    target_pos = agent.goal
                
                # Plan concrete path segment with proper time offset
                segment_path = self.astar.plan_concrete(
                    current_pos, target_pos, 
                    self.global_reservation_table, 
                    current_time
                )
                
                if segment_path is None:
                    print(f"Failed to find concrete path for Agent {agent.id}")
                    return None
                
                # Add segment (excluding first position if not the first segment)
                if concrete_path:
                    # Skip the first position to avoid duplication
                    concrete_path.extend(segment_path[1:])
                    current_time += len(segment_path) - 1
                else:
                    # First segment, include all positions
                    concrete_path.extend(segment_path)
                    current_time += len(segment_path) - 1
                
                current_pos = target_pos
            
            agent.concrete_path = concrete_path
            
            # Add comprehensive reservations to global table
            for t, pos in enumerate(concrete_path):
                self.global_reservation_table[t].add(pos)
                
                # Also reserve edges for moving actions
                if t > 0:
                    prev_pos = concrete_path[t-1]
                    # Create edge reservation: (from_pos, to_pos) at time t
                    edge_key = (prev_pos, pos)
                    if 'edges' not in self.global_reservation_table:
                        self.global_reservation_table['edges'] = defaultdict(set)
                    self.global_reservation_table['edges'][t].add(edge_key)
            
            print(f"Agent {agent.id} concrete path length: {len(concrete_path)}")
            
            # Visualize progress
            self.plot_state(f"HCA* - Agent {agent.id} Concrete Path Planned")
            self.plot_all_concrete_paths()
            self.capture_frame()
        
        # Final solution
        print("\nPlanning complete!")
        solution = {agent.id: agent.concrete_path for agent in self.agents}
        
        self.plot_state("HCA* - Final Solution")
        self.plot_all_concrete_paths()
        self.capture_frame()
        
        # Animation
        self.animate_solution(solution)
        self.save_animation_as_gif("085_hierarchical_cooperative_astar")
        # plt.show()  # Disable interactive display
        
        # Validate solution
        conflicts = self.detect_conflicts_in_solution(solution)
        if conflicts:
            print("\nConflicts detected in the solution:")
            for conflict in conflicts:
                print(conflict)
            return None
        else:
            print("\n✅ COLLISION VALIDATION PASSED!")
            print("   No conflicts detected in the final solution")
        
        return solution

    def plot_state(self, title):
        """Plot the current state"""
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

        # Plot agents
        for i, agent in enumerate(self.agents):
            color = self.colors[i % len(self.colors)]
            plt.plot(agent.start[0], agent.start[1], 's', color=color, markersize=10, 
                    label=f'Agent {agent.id} Start (P{agent.priority})', 
                    markeredgecolor='black', markeredgewidth=1)
            plt.plot(agent.goal[0], agent.goal[1], '^', color=color, markersize=10, 
                    label=f'Agent {agent.id} Goal', 
                    markeredgecolor='black', markeredgewidth=1)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()

    def plot_abstract_graph(self):
        """Visualize the abstract graph"""
        # Plot abstract nodes
        for abstract_pos, concrete_pos in self.abstract_graph.abstract_nodes.items():
            plt.plot(concrete_pos[0], concrete_pos[1], 'o', color='lightblue', 
                    markersize=8, alpha=0.6, markeredgecolor='navy', markeredgewidth=1)
            # Add cluster boundaries
            ax, ay = abstract_pos
            cluster_size = self.abstract_graph.cluster_size
            x_start = ax * cluster_size
            y_start = ay * cluster_size
            x_end = min(x_start + cluster_size, self.env.x_range[1])
            y_end = min(y_start + cluster_size, self.env.y_range[1])
            
            rect = patches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=1, edgecolor='lightblue', facecolor='none', alpha=0.3
            )
            self.ax.add_patch(rect)
        
        # Plot abstract edges
        for node, neighbors in self.abstract_graph.abstract_edges.items():
            node_pos = self.abstract_graph.abstract_nodes[node]
            for neighbor in neighbors:
                neighbor_pos = self.abstract_graph.abstract_nodes[neighbor]
                plt.plot([node_pos[0], neighbor_pos[0]], [node_pos[1], neighbor_pos[1]], 
                        '-', color='lightblue', alpha=0.4, linewidth=1)

    def plot_abstract_path(self, agent):
        """Plot an agent's abstract path"""
        if not agent.abstract_path:
            return
        
        color = self.colors[agent.id % len(self.colors)]
        path_positions = [self.abstract_graph.abstract_nodes[node] for node in agent.abstract_path]
        
        x_coords = [pos[0] for pos in path_positions]
        y_coords = [pos[1] for pos in path_positions]
        
        plt.plot(x_coords, y_coords, '-', color=color, linewidth=4, alpha=0.8,
                label=f'Agent {agent.id} Abstract Path')
        
        for i, pos in enumerate(path_positions):
            plt.plot(pos[0], pos[1], 'D', color=color, markersize=8, alpha=0.8,
                    markeredgecolor='black', markeredgewidth=1)

    def plot_all_concrete_paths(self):
        """Plot all agents' concrete paths"""
        for agent in self.agents:
            if agent.concrete_path:
                color = self.colors[agent.id % len(self.colors)]
                x_coords = [pos[0] for pos in agent.concrete_path]
                y_coords = [pos[1] for pos in agent.concrete_path]
                
                plt.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.8,
                        label=f'Agent {agent.id} Path')
                
                # Plot waypoints
                for i, pos in enumerate(agent.concrete_path):
                    if i % 5 == 0:  # Show every 5th waypoint
                        plt.plot(pos[0], pos[1], 'o', color=color, markersize=4, alpha=0.6)

    def animate_solution(self, solution):
        """Create animation showing agents moving along their paths"""
        if not solution:
            return
        
        max_length = max(len(path) for path in solution.values())
        
        for t in range(0, max_length, 2):  # Show every 2nd time step
            self.plot_state(f"HCA* Solution Animation - Time Step {t}")
            
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
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).copy()
        self.frames.append(img)
        buf.close()


def main():
    # Define agents with different priorities
    agents = [
        Agent((5, 5), (45, 25), 0, priority=3),    # Highest priority
        Agent((5, 25), (45, 5), 1, priority=2),   # High priority
        Agent((25, 2), (25, 27), 2, priority=1),  # Medium priority
        Agent((45, 15), (5, 15), 3, priority=0)   # Lowest priority
    ]
    
    hca = HCA(agents)
    solution = hca.solve()
    
    if solution is None:
        print("No solution found!")
    else:
        print("\n" + "="*50)
        print("SOLUTION ANALYSIS")
        print("="*50)
        
        # Path lengths
        total_cost = 0
        for agent_id, path in solution.items():
            path_length = len(path)
            total_cost += path_length
            print(f"Agent {agent_id} path length: {path_length}")
        print(f"Total cost: {total_cost}")
        
        # Collision detection
        print("\n" + "-"*30)
        print("COLLISION DETECTION RESULTS")
        print("-"*30)
        conflicts = hca.detect_conflicts_in_solution(solution)
        
        if not conflicts:
            print("✅ NO COLLISIONS DETECTED!")
            print("   - All vertex conflicts avoided")
            print("   - All edge conflicts (swapping) avoided")
            print("   - Solution is completely collision-free")
        else:
            print("❌ COLLISIONS DETECTED:")
            for conflict in conflicts:
                print(f"   - {conflict}")
        
        print(f"\nFinal validation: {'PASSED' if not conflicts else 'FAILED'}")
        print("="*50)


if __name__ == '__main__':
    main() 