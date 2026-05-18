"""
JPS+ (Jump Point Search Plus)
@author: Trae AI (based on JPS by clark bai)
"""

import os
import sys
import math
import heapq
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class JPSPlus:
    """
    Jump Point Search Plus (JPS+) Algorithm.
    It involves an offline precomputation of a jump point graph,
    followed by an online A* search on this graph.
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()
        self.u_set = self.Env.motions  # All 8 directions
        self.obs = self.Env.obs

        # For online search
        self.OPEN = []
        self.CLOSED_online = set()
        self.PARENT = dict()
        self.g = dict()

        # For precomputation
        self.precomputed_graph = {}  # Stores {jp: [(neighbor_jp, cost), ...]}
        self.all_jump_points = set() # Stores all identified jump points

        # Initialize plotting - create one shared figure for the entire process
        self.plot_util = plotting.Plotting(self.s_start, self.s_goal)
        self.fig, self.ax = plt.subplots()
        self.plot_util.plot_grid("Jump Point Search Plus (JPS+)")


    def _is_valid(self, node):
        """Check if a node is within bounds."""
        return 0 <= node[0] < self.Env.x_range and 0 <= node[1] < self.Env.y_range

    def _is_obstacle(self, node):
        """Check if a node is an obstacle or out of bounds."""
        if not self._is_valid(node):
            return True
        return node in self.obs

    def _jps_core_find_jump_point(self, parent, current):
        """
        Core JPS logic to find the next jump point from 'parent' towards 'current'.
        This is adapted from the JPS find_jump_point method.
        Returns the jump point if found, otherwise None.
        """
        if not self._is_valid(current) or self._is_obstacle(current):
            return None

        if current == self.s_goal:
            return current

        dx = current[0] - parent[0]
        dy = current[1] - parent[1]

        # Normalize direction
        norm_dx = dx // abs(dx) if dx != 0 else 0
        norm_dy = dy // abs(dy) if dy != 0 else 0

        # Check for forced neighbors
        # Diagonal movement
        if norm_dx != 0 and norm_dy != 0:
            # Check horizontally and vertically for forced neighbors
            # Forced neighbor along x-axis?
            if self._is_obstacle((current[0] - norm_dx, current[1])) and \
               not self._is_obstacle((current[0] - norm_dx, current[1] + norm_dy)):
                return current
            # Forced neighbor along y-axis?
            if self._is_obstacle((current[0], current[1] - norm_dy)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] - norm_dy)):
                return current

            # Recursive calls for diagonal movement
            if self._jps_core_find_jump_point(current, (current[0] + norm_dx, current[1])): # Horizontal search
                return current
            if self._jps_core_find_jump_point(current, (current[0], current[1] + norm_dy)): # Vertical search
                return current
        # Straight movement (Horizontal)
        elif norm_dx != 0: # dx != 0, dy == 0
            # Forced neighbor above?
            if self._is_obstacle((current[0], current[1] + 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] + 1)):
                return current
            # Forced neighbor below?
            if self._is_obstacle((current[0], current[1] - 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] - 1)):
                return current
        # Straight movement (Vertical)
        else: # dx == 0, dy != 0
            # Forced neighbor to the right?
            if self._is_obstacle((current[0] + 1, current[1])) and \
               not self._is_obstacle((current[0] + 1, current[1] + norm_dy)):
                return current
            # Forced neighbor to the left?
            if self._is_obstacle((current[0] - 1, current[1])) and \
               not self._is_obstacle((current[0] - 1, current[1] + norm_dy)):
                return current
        
        # Continue jumping in the same direction
        next_node = (current[0] + norm_dx, current[1] + norm_dy)
        return self._jps_core_find_jump_point(current, next_node)


    def precompute_graph(self):
        """
        Offline precomputation of the jump point graph.
        """
        print("Starting precomputation of jump point graph...")
        plt.title("JPS+ - Precomputation Phase")

        # Start with s_start. Goal is implicitly a jump point.
        # All nodes could potentially be jump points or lead to one.
        # We iterate from known jump points to find successors.
        
        jp_queue = []
        if self._is_valid(self.s_start) and not self._is_obstacle(self.s_start):
            jp_queue.append(self.s_start)
            self.all_jump_points.add(self.s_start)
            plt.plot(self.s_start[0], self.s_start[1], 'bs', markersize=8, label="Start (JP)")

        # Add goal as a jump point if valid
        if self._is_valid(self.s_goal) and not self._is_obstacle(self.s_goal):
             self.all_jump_points.add(self.s_goal) # Goal is always a jump point
             plt.plot(self.s_goal[0], self.s_goal[1], 'gs', markersize=8, label="Goal (JP)")


        head = 0
        processed_count = 0
        while head < len(jp_queue):
            current_jp = jp_queue[head]
            head += 1
            processed_count +=1

            if processed_count % 20 == 0:
                print(f"Precomputing from JP {processed_count}/{len(jp_queue)}: {current_jp}")
                plt.pause(0.01)


            if current_jp not in self.precomputed_graph:
                self.precomputed_graph[current_jp] = []

            # For each of the 8 directions from current_jp
            for dx, dy in self.u_set:
                initial_step_node = (current_jp[0] + dx, current_jp[1] + dy)

                if not self._is_valid(initial_step_node) or self._is_obstacle(initial_step_node):
                    continue
                
                # Find the next jump point in this direction
                successor_jp = self._jps_core_find_jump_point(current_jp, initial_step_node)

                if successor_jp:
                    cost_val = self.cost(current_jp, successor_jp)
                    
                    # Add edge to graph if not already present
                    is_duplicate = any(entry[0] == successor_jp for entry in self.precomputed_graph.get(current_jp, []))
                    if not is_duplicate:
                        self.precomputed_graph[current_jp].append((successor_jp, cost_val))
                        plt.plot([current_jp[0], successor_jp[0]], [current_jp[1], successor_jp[1]],
                                 'c-', linewidth=0.5, alpha=0.3) # Connection line

                    if successor_jp not in self.all_jump_points:
                        self.all_jump_points.add(successor_jp)
                        jp_queue.append(successor_jp)
                        if successor_jp != self.s_goal and successor_jp != self.s_start:
                             plt.plot(successor_jp[0], successor_jp[1], 'mo', markersize=4, alpha=0.6) # New JP

        # Plot all identified jump points distinctly
        for jp_node in self.all_jump_points:
            if jp_node != self.s_start and jp_node != self.s_goal:
                 plt.plot(jp_node[0], jp_node[1], 'mo', markersize=5, label='_nolegend_') # Magenta for JPs

        print(f"Precomputation finished. Found {len(self.all_jump_points)} jump points.")
        print(f"Graph has {sum(len(adj) for adj in self.precomputed_graph.values())} edges.")
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: # Avoid error if no labels
            self.ax.legend(by_label.values(), by_label.keys())
        plt.title("JPS+ - Precomputation Complete")
        plt.pause(1) # Pause briefly to see the precomputation result

    def searching_online(self):
        """
        Online A* search on the precomputed jump point graph.
        """
        if not self.all_jump_points:
            print("No jump points found or precomputation not run. Cannot search.")
            return [], []

        if self.s_start not in self.all_jump_points:
            print(f"Start node {self.s_start} is not a known jump point. Cannot start search.")
            # Attempt to connect s_start to the nearest jump point if desired, or fail.
            # For now, we assume s_start must be a jump point from precomputation.
            return [], []

        # Update the title for the online search phase
        plt.title("JPS+ - Online Search Phase")

        # Plot all precomputed jump points
        for jp_node in self.all_jump_points:
            if jp_node != self.s_start and jp_node != self.s_goal:
                plt.plot(jp_node[0], jp_node[1], 'mo', markersize=5, alpha=0.5, label='_nolegend_')
        
        # Plot precomputed graph edges lightly
        for u, adj in self.precomputed_graph.items():
            for v, _ in adj:
                plt.plot([u[0], v[0]], [u[1], v[1]], 'c-', linewidth=0.5, alpha=0.2, label='_nolegend_')


        self.PARENT.clear()
        self.g.clear()
        self.OPEN = []
        self.CLOSED_online = set()
        
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        
        visited_online_nodes = [self.s_start]
        nodes_processed = 0

        print(f"Starting JPS+ online search from {self.s_start} to {self.s_goal}")

        while self.OPEN:
            _, current_jp = heapq.heappop(self.OPEN)
            nodes_processed += 1

            if current_jp in self.CLOSED_online:
                continue
            self.CLOSED_online.add(current_jp)
            
            if current_jp != self.s_start and current_jp != self.s_goal :
                 visited_online_nodes.append(current_jp)
                 plt.plot(current_jp[0], current_jp[1], 'yo', markersize=6) # Visited JP in online search

            if current_jp == self.s_goal:
                print(f"Goal reached in online search after processing {nodes_processed} jump points!")
                path = self.extract_path(self.PARENT)
                self.plot_util.plot_path(path) # Uses self.ax by default if Plotting is adapted
                plt.pause(0.25) # User's highlighted pause
                break
            
            if nodes_processed % 10 == 0:
                 print(f"Online search: processing JP {nodes_processed}, current: {current_jp}")
                 plt.pause(0.01)

            # Successors are from the precomputed graph
            if current_jp in self.precomputed_graph:
                for successor_jp, cost_val in self.precomputed_graph[current_jp]:
                    if successor_jp in self.CLOSED_online: # Successor already processed
                        continue

                    new_cost = self.g[current_jp] + cost_val
                    if successor_jp not in self.g or new_cost < self.g[successor_jp]:
                        self.g[successor_jp] = new_cost
                        self.PARENT[successor_jp] = current_jp
                        heapq.heappush(self.OPEN, (self.f_value(successor_jp), successor_jp))
                        
                        # Plot edge being considered in A*
                        plt.plot([current_jp[0], successor_jp[0]], [current_jp[1], successor_jp[1]],
                                 'b-', linewidth=1, alpha=0.7)
                        plt.pause(0.5)
        else: # Loop finished without break (goal not found)
            if self.s_goal not in self.PARENT:
                 print(f"No path found to {self.s_goal} in online search.")
                 path = []
            else: # Should be caught by break, but as a fallback
                 path = self.extract_path(self.PARENT)


        if path:
            print(f"Online path found with {len(path)} nodes (segments between JPs).")

            plt.title("JPS+ - Final Path Result")
            # 明确设置图例中各个元素的颜色
            start_marker = plt.Line2D([], [], marker='s', color='b', label='Start Point', markerfacecolor='b', markersize=8)
            goal_marker = plt.Line2D([], [], marker='s', color='g', label='Goal Point', markerfacecolor='g', markersize=8)
            jp_marker = plt.Line2D([], [], marker='o', color='m', label='Jump Point (Precomputed)', markerfacecolor='m', markersize=5)
            visited_jp_marker = plt.Line2D([], [], marker='o', color='y', label='Visited Jump Point (Online)', markerfacecolor='y', markersize=6)
            path_segment = plt.Line2D([], [], color='r', label='Final Path', linewidth=2)
            plt.legend(handles=[start_marker, goal_marker, jp_marker, visited_jp_marker, path_segment], loc='best')

        else:
            print(f"No path found by JPS+ online search after processing {nodes_processed} jump points.")
        
        plt.show() # Keep plot open
        return path, visited_online_nodes


    def cost(self, s_start_node, s_goal_node):
        if self._is_obstacle(s_start_node) or self._is_obstacle(s_goal_node):
            return math.inf
        return math.hypot(s_goal_node[0] - s_start_node[0], s_goal_node[1] - s_start_node[1])

    def f_value(self, s_node):
        # Ensure g-value exists, default to infinity if not (though should be set before f_value call)
        g_val = self.g.get(s_node, math.inf)
        return g_val + self.heuristic(s_node)

    def extract_path(self, parent_map):
        if self.s_goal not in parent_map:
            return []
            
        path = [self.s_goal]
        current_s = self.s_goal
        while current_s != self.s_start:
            if current_s not in parent_map: # Should not happen if goal is in parent_map
                print(f"Error: Path reconstruction failed. Node {current_s} not in parent_map.")
                return [] # Path broken
            current_s = parent_map[current_s]
            path.append(current_s)
        path.reverse()
        
        # The path extracted is a sequence of jump points.
        # For a full grid path, one would need to interpolate between these jump points.
        # For this demo, we'll show the path connecting jump points.
        return path

    def heuristic(self, s_node):
        goal_node = self.s_goal
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - s_node[0]) + abs(goal_node[1] - s_node[1])
        else:  # Default to euclidean
            return math.hypot(goal_node[0] - s_node[0], goal_node[1] - s_node[1])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    # s_start = (1,1)
    # s_goal = (5,6) # Small map for testing
    heuristic_type = "euclidean"

    jps_plus_algo = JPSPlus(s_start, s_goal, heuristic_type)
    
    # --- Offline Precomputation Phase ---
    start_time_precompute = time.time()
    jps_plus_algo.precompute_graph()
    end_time_precompute = time.time()
    print(f"Precomputation Time: {end_time_precompute - start_time_precompute:.4f} seconds")

    # --- Online Search Phase ---
    if jps_plus_algo.all_jump_points: # Proceed only if precomputation was successful
        start_time_online = time.time()
        path, visited_online = jps_plus_algo.searching_online()
        end_time_online = time.time()
        print(f"Online Search Time: {end_time_online - start_time_online:.4f} seconds")
        
        if path:
            print(f"Path found: {path}")
            print(f"Visited jump points during online search: {len(visited_online)}")
        else:
            print("No path found in online search.")
    else:
        print("Skipping online search due to precomputation issues.")

if __name__ == '__main__':
    main()
