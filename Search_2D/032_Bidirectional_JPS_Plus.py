"""
Bidirectional JPS++ (Jump Point Search Plus)
@author: Cline (based on JPS+ by Trae AI and JPS by clark bai)
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


class BidirectionalJPSPlus:
    """
    Bidirectional Jump Point Search Plus (JPS++) Algorithm.
    
    This algorithm combines JPS+ (which includes an offline precomputation phase)
    with bidirectional search strategy. The search proceeds from both start and goal
    simultaneously, which can significantly reduce search time in many cases.
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()
        self.u_set = self.Env.motions  # All 8 directions
        self.obs = self.Env.obs

        # For online search - Forward (start to goal)
        self.OPEN_fwd = []
        self.CLOSED_fwd = set()
        self.PARENT_fwd = dict()
        self.g_fwd = dict()

        # For online search - Backward (goal to start)
        self.OPEN_bwd = []
        self.CLOSED_bwd = set()
        self.PARENT_bwd = dict()
        self.g_bwd = dict()

        # For precomputation
        self.precomputed_graph = {}  # Stores {jp: [(neighbor_jp, cost), ...]}
        self.reverse_graph = {}      # Stores reverse edges for backward search
        self.all_jump_points = set()  # Stores all identified jump points
        
        # For meeting point detection
        self.meeting_point = None
        self.best_total_cost = float('inf')

        # Initialize plotting - create one shared figure for the entire process
        self.fig, self.ax = plt.subplots()
        self.plot_util = plotting.Plotting(self.s_start, self.s_goal)
        self.plot_util.plot_grid("Bidirectional Jump Point Search Plus (JPS++)")
        #plt.ion()  # Turn on interactive mode for real-time plotting
        
        # For visualization - use different colors for forward and backward search
        self.forward_color = 'cornflowerblue'
        self.backward_color = 'salmon'
        self.path_color = 'limegreen'

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

        if current == self.s_goal or current == self.s_start:
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
            if self._jps_core_find_jump_point(current, (current[0] + norm_dx, current[1])):  # Horizontal search
                return current
            if self._jps_core_find_jump_point(current, (current[0], current[1] + norm_dy)):  # Vertical search
                return current
        # Straight movement (Horizontal)
        elif norm_dx != 0:  # dx != 0, dy == 0
            # Forced neighbor above?
            if self._is_obstacle((current[0], current[1] + 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] + 1)):
                return current
            # Forced neighbor below?
            if self._is_obstacle((current[0], current[1] - 1)) and \
               not self._is_obstacle((current[0] + norm_dx, current[1] - 1)):
                return current
        # Straight movement (Vertical)
        else:  # dx == 0, dy != 0
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
        print(f"Start: {self.s_start}, Goal: {self.s_goal}")
        plt.title("Bidirectional JPS++ - Precomputation Phase")

        # Start with s_start and s_goal as initial jump points
        jp_queue = []
        if self._is_valid(self.s_start) and not self._is_obstacle(self.s_start):
            jp_queue.append(self.s_start)
            self.all_jump_points.add(self.s_start)
            plt.plot(self.s_start[0], self.s_start[1], 'bs', markersize=8, label="Start (JP)")

        if self._is_valid(self.s_goal) and not self._is_obstacle(self.s_goal):
            jp_queue.append(self.s_goal)  # Add goal to queue to expand from it too
            self.all_jump_points.add(self.s_goal)  # Goal is always a jump point
            plt.plot(self.s_goal[0], self.s_goal[1], 'gs', markersize=8, label="Goal (JP)")

        head = 0
        processed_count = 0
        while head < len(jp_queue):
            current_jp = jp_queue[head]
            head += 1
            processed_count += 1

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
                        
                        # Also add to reverse graph for backward search
                        if successor_jp not in self.reverse_graph:
                            self.reverse_graph[successor_jp] = []
                        self.reverse_graph[successor_jp].append((current_jp, cost_val))
                        
                        plt.plot([current_jp[0], successor_jp[0]], [current_jp[1], successor_jp[1]],
                                 'c-', linewidth=0.5, alpha=0.3)  # Connection line

                    if successor_jp not in self.all_jump_points:
                        self.all_jump_points.add(successor_jp)
                        jp_queue.append(successor_jp)
                        if successor_jp != self.s_goal and successor_jp != self.s_start:
                            plt.plot(successor_jp[0], successor_jp[1], 'mo', markersize=4, alpha=0.6)  # New JP

        # Plot all identified jump points distinctly
        for jp_node in self.all_jump_points:
            if jp_node != self.s_start and jp_node != self.s_goal:
                plt.plot(jp_node[0], jp_node[1], 'mo', markersize=5, label='_nolegend_')  # Magenta for JPs

        print(f"Precomputation finished. Found {len(self.all_jump_points)} jump points.")
        print(f"Graph has {sum(len(adj) for adj in self.precomputed_graph.values())} edges.")
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:  # Avoid error if no labels
            self.ax.legend(by_label.values(), by_label.keys())
        plt.title("Bidirectional JPS++ - Precomputation Complete")
        plt.pause(1)  # Pause briefly to see the precomputation result

    def searching_bidirectional(self):
        """
        Bidirectional search on the precomputed jump point graph.
        
        This method runs two simultaneous searches - one from start to goal and
        one from goal to start. The search terminates when the two frontiers meet.
        """
        if not self.all_jump_points:
            print("No jump points found or precomputation not run. Cannot search.")
            return [], [], []

        if self.s_start not in self.all_jump_points or self.s_goal not in self.all_jump_points:
            print(f"Either start {self.s_start} or goal {self.s_goal} is not a known jump point. Cannot search.")
            return [], [], []

        # Update the title for the bidirectional search phase
        plt.title("Bidirectional JPS++ - Online Search Phase")

        # Plot all precomputed jump points
        for jp_node in self.all_jump_points:
            if jp_node != self.s_start and jp_node != self.s_goal:
                plt.plot(jp_node[0], jp_node[1], 'mo', markersize=5, alpha=0.3, label='_nolegend_')
        
        # Plot precomputed graph edges lightly
        for u, adj in self.precomputed_graph.items():
            for v, _ in adj:
                plt.plot([u[0], v[0]], [u[1], v[1]], 'c-', linewidth=0.5, alpha=0.1, label='_nolegend_')

        # Initialize forward search (start to goal)
        self.PARENT_fwd.clear()
        self.g_fwd.clear()
        self.OPEN_fwd = []
        self.CLOSED_fwd = set()
        
        self.PARENT_fwd[self.s_start] = self.s_start
        self.g_fwd[self.s_start] = 0
        heapq.heappush(self.OPEN_fwd, (self.f_value(self.s_start, self.s_goal, "forward"), self.s_start))
        
        # Initialize backward search (goal to start)
        self.PARENT_bwd.clear()
        self.g_bwd.clear()
        self.OPEN_bwd = []
        self.CLOSED_bwd = set()
        
        self.PARENT_bwd[self.s_goal] = self.s_goal
        self.g_bwd[self.s_goal] = 0
        heapq.heappush(self.OPEN_bwd, (self.f_value(self.s_goal, self.s_start, "backward"), self.s_goal))
        
        # Track visited nodes for visualization
        visited_fwd = [self.s_start]
        visited_bwd = [self.s_goal]
        
        # For statistics
        nodes_processed_fwd = 0
        nodes_processed_bwd = 0
        
        print(f"Starting bidirectional search from {self.s_start} to {self.s_goal}")
        
        # Main search loop - alternating between forward and backward search
        while self.OPEN_fwd and self.OPEN_bwd:
            # Check if we should terminate the search
            if self.meeting_point and (
                (not self.OPEN_fwd or self.f_value(self.OPEN_fwd[0][1], self.s_goal, "forward") >= self.best_total_cost) and
                (not self.OPEN_bwd or self.f_value(self.OPEN_bwd[0][1], self.s_start, "backward") >= self.best_total_cost)
            ):
                print(f"Search terminated - optimal meeting point found: {self.meeting_point}")
                print(f"Best path cost: {self.best_total_cost}")
                break
                
            # ---- Forward search step ----
            if self.OPEN_fwd:
                # Process one node from forward search
                _, current_jp_fwd = heapq.heappop(self.OPEN_fwd)
                nodes_processed_fwd += 1
                
                if current_jp_fwd in self.CLOSED_fwd:
                    continue
                self.CLOSED_fwd.add(current_jp_fwd)
                
                if current_jp_fwd != self.s_start:
                    visited_fwd.append(current_jp_fwd)
                    plt.plot(current_jp_fwd[0], current_jp_fwd[1], 'o', color=self.forward_color, markersize=6)
                
                # Check if this node has been visited by backward search
                if current_jp_fwd in self.CLOSED_bwd:
                    total_cost = self.g_fwd[current_jp_fwd] + self.g_bwd[current_jp_fwd]
                    self.best_total_cost = total_cost
                    self.meeting_point = current_jp_fwd
                    print(f"Meeting point found: {current_jp_fwd} with cost {total_cost}")
                    plt.plot(current_jp_fwd[0], current_jp_fwd[1], 'yo', markersize=10)  # Highlight meeting point
                    plt.pause(0.5)
                    # Immediately break out of the search loop
                    break
                
                # Expand successors
                if current_jp_fwd in self.precomputed_graph:
                    for successor_jp, cost_val in self.precomputed_graph[current_jp_fwd]:
                        if successor_jp in self.CLOSED_fwd:
                            continue
                        
                        new_cost = self.g_fwd[current_jp_fwd] + cost_val
                        if successor_jp not in self.g_fwd or new_cost < self.g_fwd[successor_jp]:
                            self.g_fwd[successor_jp] = new_cost
                            self.PARENT_fwd[successor_jp] = current_jp_fwd
                            heapq.heappush(self.OPEN_fwd, (self.f_value(successor_jp, self.s_goal, "forward"), successor_jp))
                            
                            # Plot edge being considered in forward search
                            plt.plot([current_jp_fwd[0], successor_jp[0]], [current_jp_fwd[1], successor_jp[1]],
                                     '-', color=self.forward_color, linewidth=1, alpha=0.7)
                            # Show temporary path from start to this node
                            temp_path = self.extract_partial_path(successor_jp, self.PARENT_fwd, "forward")
                            if len(temp_path) > 1:  # Only plot if path has more than one node
                                xs = [x for x, _ in temp_path]
                                ys = [y for _, y in temp_path]
                                plt.plot(xs, ys, '--', color=self.forward_color, linewidth=0.8, alpha=0.5)
                            plt.draw()
                            plt.pause(0.1)  # Small pause to show search progress
                            
                            # Check if this successor has been visited by backward search
                            if successor_jp in self.CLOSED_bwd:
                                total_cost = new_cost + self.g_bwd[successor_jp]
                                self.best_total_cost = total_cost
                                self.meeting_point = successor_jp
                                print(f"Meeting point found: {successor_jp} with cost {total_cost}")
                                plt.plot(successor_jp[0], successor_jp[1], 'yo', markersize=10)  # Highlight meeting point
                                plt.pause(0.5)
                                # Instead of immediately returning, compute and display the full path
                                # Extract forward path from start to meeting point
                                fwd_path = self.extract_partial_path(successor_jp, self.PARENT_fwd, "forward")
                                
                                # Extract backward path from goal to meeting point
                                bwd_path = self.extract_partial_path(successor_jp, self.PARENT_bwd, "backward")
                                
                                # Combine the paths - the backward path needs to be reversed to get meeting to goal order
                                bwd_path_reversed = bwd_path[::-1]  # Reverse the backward path
                                
                                # Only use forward path from start to meeting point
                                full_path = fwd_path
                                
                                print(f"full_path {len(full_path)} nodes")
                                print(f"{full_path}")
                                # Plot the final path
                                plt.title("Bidirectional JPS++ - Final Path Result")
                                self.plot_util.plot_path(full_path)
                                plt.pause(1)  # Ensure the path is visible
                                
                                # Create proper legend with custom colors for the final display
                                self.create_legend()
                                plt.show()
                                
                                return full_path, visited_fwd, visited_bwd
                
            # ---- Backward search step ----
            if self.OPEN_bwd:
                # Process one node from backward search
                _, current_jp_bwd = heapq.heappop(self.OPEN_bwd)
                nodes_processed_bwd += 1
                
                if current_jp_bwd in self.CLOSED_bwd:
                    continue
                self.CLOSED_bwd.add(current_jp_bwd)
                
                if current_jp_bwd != self.s_goal:
                    visited_bwd.append(current_jp_bwd)
                    plt.plot(current_jp_bwd[0], current_jp_bwd[1], 'o', color=self.backward_color, markersize=6)
                
                # Check if this node has been visited by forward search
                if current_jp_bwd in self.CLOSED_fwd:
                    total_cost = self.g_fwd[current_jp_bwd] + self.g_bwd[current_jp_bwd]
                    self.best_total_cost = total_cost
                    self.meeting_point = current_jp_bwd
                    print(f"Meeting point found: {current_jp_bwd} with cost {total_cost}")
                    plt.plot(current_jp_bwd[0], current_jp_bwd[1], 'yo', markersize=10)  # Highlight meeting point
                    plt.pause(0.5)
                    # Immediately break out of the search loop
                    break
                
                # Expand predecessors using the reverse graph for backward search
                if current_jp_bwd in self.reverse_graph:
                    for successor_jp, cost_val in self.reverse_graph[current_jp_bwd]:
                        if successor_jp in self.CLOSED_bwd:
                            continue
                        
                        new_cost = self.g_bwd[current_jp_bwd] + cost_val
                        if successor_jp not in self.g_bwd or new_cost < self.g_bwd[successor_jp]:
                            self.g_bwd[successor_jp] = new_cost
                            self.PARENT_bwd[successor_jp] = current_jp_bwd
                            heapq.heappush(self.OPEN_bwd, (self.f_value(successor_jp, self.s_start, "backward"), successor_jp))
                            
                            # Plot edge being considered in backward search
                            plt.plot([current_jp_bwd[0], successor_jp[0]], [current_jp_bwd[1], successor_jp[1]],
                                     '-', color=self.backward_color, linewidth=1, alpha=0.7)
                            
                            # Show temporary path from goal to this node
                            temp_path = self.extract_partial_path(successor_jp, self.PARENT_bwd, "backward")
                            if len(temp_path) > 1:  # Only plot if path has more than one node
                                xs = [x for x, _ in temp_path]
                                ys = [y for _, y in temp_path]
                                plt.plot(xs, ys, '--', color=self.backward_color, linewidth=0.8, alpha=0.5)
                            plt.draw()
                            plt.pause(0.1)  # Small pause to show search progress
                            
                            # Check if this successor has been visited by forward search
                            if successor_jp in self.CLOSED_fwd:
                                total_cost = self.g_fwd[successor_jp] + new_cost
                                self.best_total_cost = total_cost
                                self.meeting_point = successor_jp
                                print(f"Meeting point found: {successor_jp} with cost {total_cost}")
                                plt.plot(successor_jp[0], successor_jp[1], 'yo', markersize=10)  # Highlight meeting point
                                plt.pause(0.5)
                                # Instead of immediately returning, compute and display the full path
                                # Extract forward path from start to meeting point
                                fwd_path = self.extract_partial_path(successor_jp, self.PARENT_fwd, "forward")
                                
                                # Extract backward path from goal to meeting point
                                bwd_path = self.extract_partial_path(successor_jp, self.PARENT_bwd, "backward")
                                
                                # Combine the paths - the backward path needs to be reversed to get meeting to goal order
                                bwd_path_reversed = bwd_path[::-1]  # Reverse the backward path
                                
                                # Now combine: start -> meeting -> goal
                                full_path = fwd_path + bwd_path[1:]
                     
                                print(f"Combined path with {len(full_path)} nodes")
                                print(f"{full_path}")
                                # Plot the final path
                                plt.title("Bidirectional JPS++ - Final Path Result")
                                self.plot_util.plot_path(full_path)
                                plt.pause(1)  # Ensure the path is visible
                                
                                # Create legend for the final display
                                self.create_legend()
                                plt.show()
                                
                                return full_path, visited_fwd, visited_bwd
            
            # Update status occasionally and pause for visualization
            if (nodes_processed_fwd + nodes_processed_bwd) % 10 == 0:
                print(f"Processed: {nodes_processed_fwd} forward, {nodes_processed_bwd} backward nodes")
                plt.pause(0.5)
        
        # Path extraction
        full_path = []
        if self.meeting_point:
            print(f"Extracting path through meeting point: {self.meeting_point}")
            
            # Extract forward path from start to meeting point
            fwd_path = self.extract_partial_path(self.meeting_point, self.PARENT_fwd, "forward")
            print(f"Forward path (start to meeting): {fwd_path}")
            
            # Extract backward path from goal to meeting point
            bwd_path = self.extract_partial_path(self.meeting_point, self.PARENT_bwd, "backward")
            print(f"Backward path (goal to meeting): {bwd_path}")
            
            # Combine the paths - the backward path needs to be reversed to get meeting to goal order
            bwd_path_reversed = bwd_path[::-1]  # Reverse the backward path
            
            # Now combine: start -> meeting -> goal
            if len(fwd_path) > 0 and len(bwd_path_reversed) > 0:
                if fwd_path[-1] == bwd_path_reversed[0]:  # If meeting point appears in both paths
                    full_path = fwd_path + bwd_path_reversed[1:]  # Skip duplicate meeting point
                else:
                    full_path = fwd_path + bwd_path_reversed  # No duplication
                    
            print(f"Combined path with {len(full_path)} nodes: {full_path}")
            
            # Plot the final path
            plt.title("Bidirectional JPS++ - Final Path Result")
            self.plot_util.plot_path(full_path)
            
            print(f"Bidirectional search complete. Path found with {len(full_path)} nodes.")
            print(f"Nodes processed: {nodes_processed_fwd} forward, {nodes_processed_bwd} backward. "
                  f"Total: {nodes_processed_fwd + nodes_processed_bwd}")
        else:
            print("No path found by bidirectional search.")
        
        # Create legend using the create_legend method
        self.create_legend()
        plt.show()
        
        return full_path, visited_fwd, visited_bwd

    def extract_partial_path(self, meeting_node, parent_dict, direction):
        """
        Extract path from start or goal to the meeting point.
        
        Parameters:
            meeting_node: Node where forward and backward searches meet
            parent_dict: Parent dictionary (PARENT_fwd or PARENT_bwd)
            direction: "forward" or "backward" to determine path orientation
        
        Returns:
            List of nodes representing the path
        """
        path = [meeting_node]
        current = meeting_node
        
        if direction == "forward":
            # Start to meeting point
            while current != self.s_start and current in parent_dict:
                current = parent_dict[current]
                path.append(current)
            path.reverse()  # Reverse to get start to meeting point order
        else:
            # Goal to meeting point
            while current != self.s_goal and current in parent_dict:
                current = parent_dict[current]
                path.append(current)
            # For the backward path, the order is goal->...->meeting point
            # which will be reversed later when joining paths
        
        # Check if we actually found a path
        if direction == "forward" and self.s_start not in path:
            print(f"Warning: Forward path does not contain start node!")
            return []
        elif direction == "backward" and self.s_goal not in path:
            print(f"Warning: Backward path does not contain goal node!")
            return []
            
        return path

    def cost(self, s_start_node, s_goal_node):
        """Calculate cost between two nodes (Euclidean distance)."""
        if self._is_obstacle(s_start_node) or self._is_obstacle(s_goal_node):
            return math.inf
        return math.hypot(s_goal_node[0] - s_start_node[0], s_goal_node[1] - s_start_node[1])

    def f_value(self, s_node, goal, direction):
        """
        Calculate f value (f = g + h) for either forward or backward search.
        
        Parameters:
            s_node: Current node
            goal: Goal node for this search direction
            direction: "forward" or "backward" to determine which g-value to use
        
        Returns:
            f-value for the node
        """
        if direction == "forward":
            g_val = self.g_fwd.get(s_node, math.inf)
        else:  # backward
            g_val = self.g_bwd.get(s_node, math.inf)
            
        return g_val + self.heuristic(s_node, goal)

    def heuristic(self, s_node, goal_node=None):
        """
        Calculate heuristic (estimated cost from s_node to goal).
        
        Parameters:
            s_node: Current node
            goal_node: Goal node (defaults to self.s_goal if not specified)
        
        Returns:
            Heuristic value
        """
        if goal_node is None:
            goal_node = self.s_goal
            
        if self.heuristic_type == "manhattan":
            return abs(goal_node[0] - s_node[0]) + abs(goal_node[1] - s_node[1])
        else:  # Default to euclidean
            return math.hypot(goal_node[0] - s_node[0], goal_node[1] - s_node[1])
            
    def create_legend(self):
        """
        Create a custom legend for the plot visualization with all elements.
        """
        # Create proper legend with custom colors
        start_marker = plt.Line2D([], [], marker='s', color='b', label='Start Point', markerfacecolor='b', markersize=8)
        goal_marker = plt.Line2D([], [], marker='s', color='g', label='Goal Point', markerfacecolor='g', markersize=8)
        jp_marker = plt.Line2D([], [], marker='o', color='m', label='Jump Point (Precomputed)', markerfacecolor='m', markersize=5)
        fwd_marker = plt.Line2D([], [], marker='o', color=self.forward_color, label='Forward Search Node', markerfacecolor=self.forward_color, markersize=6)
        bwd_marker = plt.Line2D([], [], marker='o', color=self.backward_color, label='Backward Search Node', markerfacecolor=self.backward_color, markersize=6)
        meeting_marker = plt.Line2D([], [], marker='o', color='y', label='Meeting Point', markerfacecolor='y', markersize=10)
        path_line = plt.Line2D([], [], color='r', label='Final Path', linewidth=2)
        
        plt.legend(handles=[start_marker, goal_marker, jp_marker, fwd_marker, bwd_marker, meeting_marker, path_line], loc='best')


def main():
    """
    Testing the Bidirectional JPS++ implementation.
    """
    print("Bidirectional Jump Point Search Plus (JPS++) Implementation")
    print("----------------------------------------------------------")

    s_start = (5, 5)
    s_goal = (45, 25)
    heuristic_type = "euclidean"

    bidirectional_jps_plus = BidirectionalJPSPlus(s_start, s_goal, heuristic_type)
    
    # --- Offline Precomputation Phase ---
    start_time_precompute = time.time()
    bidirectional_jps_plus.precompute_graph()
    end_time_precompute = time.time()
    precompute_time = end_time_precompute - start_time_precompute
    print(f"Precomputation Time: {precompute_time:.4f} seconds")

    # --- Bidirectional Online Search Phase ---
    if bidirectional_jps_plus.all_jump_points:  # Proceed only if precomputation was successful
        start_time_online = time.time()
        path, visited_fwd, visited_bwd = bidirectional_jps_plus.searching_bidirectional()
        end_time_online = time.time()
        online_time = end_time_online - start_time_online
        print(f"Online Search Time: {online_time:.4f} seconds")
        print(f"Total Algorithm Time: {precompute_time + online_time:.4f} seconds")
        
        if path:
            print(f"Path length: {len(path)}")
            print(f"Forward search visited: {len(visited_fwd)} nodes")
            print(f"Backward search visited: {len(visited_bwd)} nodes")
            print(f"Total nodes visited: {len(visited_fwd) + len(visited_bwd)}")
        else:
            print("No path found.")
    else:
        print("Skipping bidirectional search due to precomputation issues.")
   

if __name__ == '__main__':
    main()
