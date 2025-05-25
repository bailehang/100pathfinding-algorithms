"""
S-Theta* 2D: Low steering path-planning algorithm
@author: clark bai
"""

import os
import sys
import math
import heapq
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class SThetaStar:
    """
    S-Theta*: Low steering path-planning algorithm
    S-Theta* improves on Theta* by considering steering constraints and minimizing steering changes,
    producing smoother paths suitable for vehicles with turning constraints.
    """
    def __init__(self, s_start, s_goal, heuristic_type, steering_weight=1.0):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.steering_weight = steering_weight  # Weight for steering cost

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.heading = dict()  # heading direction at each node
        
        # For visualization
        self.los_checks = []
        self.fig = plt.figure()
        self.plot = plotting.Plotting(s_start, s_goal)
        
        # Current search state
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []

    def searching(self):
        """
        S-Theta* path searching.
        :return: path, visited order
        """
        # Initialize plot
        self.plot.plot_grid("S-Theta*")

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        self.heading[self.s_start] = (0, 0)  # Initial heading (no specific direction)
        
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            self.current_visited.append(s)

            # Update current path and visualization
            if s == self.s_goal:
                self.current_path = self.extract_path(self.PARENT)
            else:
                # Show path from start to current node
                temp_path = self.extract_temp_path(s)
                self.current_path = temp_path
            
            # Update visualization periodically
            if len(self.CLOSED) % 5 == 0 or s == self.s_goal:
                self.update_plot()

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                # Calculate heading to neighbor
                heading_to_neighbor = self.calculate_heading(s, s_n)
                
                # Path 1 - Regular path through current node
                new_cost_regular = self.g[s] + self.cost(s, s_n)
                
                # Add steering cost for Path 1
                if s != self.s_start:
                    steering_change_regular = self.steering_cost(self.heading[s], heading_to_neighbor)
                    new_cost_regular += self.steering_weight * steering_change_regular
                
                # Path 2 - Try to use line-of-sight from parent
                new_cost_los = math.inf
                
                # Check line-of-sight from parent
                los_result = self.line_of_sight(self.PARENT[s], s_n)
                self.los_checks.append((self.PARENT[s], s_n, los_result))
                self.current_los_checks.append((self.PARENT[s], s_n, los_result))
                
                if los_result:
                    # Line-of-sight exists, consider path from parent
                    heading_from_parent = self.calculate_heading(self.PARENT[s], s_n)
                    new_cost_los = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)
                    
                    # Add steering cost for Path 2
                    if self.PARENT[s] != self.s_start:
                        steering_change_los = self.steering_cost(self.heading[self.PARENT[s]], heading_from_parent)
                        new_cost_los += self.steering_weight * steering_change_los

                # Choose the better path
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # Compare costs and update if better path found
                if new_cost_regular <= new_cost_los:
                    # Path 1 is better or equal
                    if new_cost_regular < self.g[s_n]:
                        self.g[s_n] = new_cost_regular
                        self.PARENT[s_n] = s
                        self.heading[s_n] = heading_to_neighbor
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                else:
                    # Path 2 is better
                    if new_cost_los < self.g[s_n]:
                        self.g[s_n] = new_cost_los
                        self.PARENT[s_n] = self.PARENT[s]  # Skip a step in the path
                        self.heading[s_n] = heading_from_parent
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # Final update
        self.update_plot(final=True)
        plt.show()
        
        return self.extract_path(self.PARENT), self.CLOSED

    def calculate_heading(self, s_start, s_end):
        """
        Calculate heading direction from s_start to s_end
        :param s_start: start node
        :param s_end: end node
        :return: normalized heading vector
        """
        dx = s_end[0] - s_start[0]
        dy = s_end[1] - s_start[1]
        distance = math.hypot(dx, dy)
        
        if distance == 0:
            return (0, 0)  # No specific direction if same point
        
        return (dx/distance, dy/distance)

    def steering_cost(self, heading1, heading2):
        """
        Calculate steering cost between two headings
        :param heading1: first heading vector
        :param heading2: second heading vector
        :return: steering cost (angle change)
        """
        # Calculate dot product
        dot_product = heading1[0] * heading2[0] + heading1[1] * heading2[1]
        # Clamp to avoid numerical issues
        dot_product = max(-1, min(1, dot_product))
        # Calculate angle in radians
        angle = math.acos(dot_product)
        
        return angle

    def update_plot(self, final=False):
        """
        Update the plot to show current search state
        """
        # Clear current figure
        plt.cla()
        
        # Draw grid using plotting.py methods
        self.plot.plot_grid("S-Theta*")
        
        # Draw visited nodes
        if self.current_visited:
            for node in self.current_visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], color='gray', marker='o')
        
        # Draw line-of-sight checks
        if self.current_los_checks:
            for start, end, result in self.current_los_checks:
                color = 'g' if result else 'r'
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.3)
        
        # Draw current path
        if self.current_path:
            self.plot.plot_path(self.current_path)
            
            # Draw heading arrows for current path (optional)
            if len(self.current_path) > 1:
                for i in range(len(self.current_path) - 1):
                    if self.current_path[i] in self.heading and self.heading[self.current_path[i]] != (0, 0):
                        x, y = self.current_path[i]
                        dx, dy = self.heading[self.current_path[i]]
                        # Scale arrows
                        scale = 2.0
                        plt.arrow(x, y, dx*scale, dy*scale, head_width=0.5, head_length=0.5, 
                                 fc='blue', ec='blue', alpha=0.5)
        
        # Update figure
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        
        # Pause longer for final result
        if final:
            plt.pause(0.5)
        else:
            plt.pause(0.01)
    
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

    def get_neighbor(self, s):
        """
        Find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        nei_list = []
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        """
        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # Check if points are within grid boundaries
        x_range, y_range = self.Env.x_range, self.Env.y_range
        
        # Check boundary constraints
        if (s_start[0] < 0 or s_start[0] >= x_range or 
            s_start[1] < 0 or s_start[1] >= y_range or
            s_end[0] < 0 or s_end[0] >= x_range or
            s_end[1] < 0 or s_end[1] >= y_range):
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        # Basic check for diagonal segments
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        # Bresenham's line algorithm for more thorough checking
        x0, y0 = s_start
        x1, y1 = s_end
        
        # If the line is steep, transpose the grid
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        # Swap points if needed to ensure x increases
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        y = y0
        
        # Determine step direction
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # Check each point on the line
        for x in range(x0, x1 + 1):
            if steep:
                # If steep, the coordinates are transposed
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

    def line_of_sight(self, s_start, s_end):
        """
        Check if there is a line-of-sight between two nodes
        :param s_start: start node
        :param s_end: end node
        :return: True if line-of-sight exists
        """
        return not self.is_collision(s_start, s_end)

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(reversed(path))

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """
        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def path_smoothing(self, path):
        """
        Apply path smoothing to reduce unnecessary heading changes
        :param path: original path
        :return: smoothed path
        """
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]
        current = path[0]
        i = 1
        
        while i < len(path) - 1:
            # Try to connect current node to nodes further ahead in the path
            for j in range(len(path) - 1, i, -1):
                if not self.is_collision(current, path[j]):
                    # Found a valid shortcut
                    current = path[j]
                    smoothed_path.append(current)
                    i = j + 1
                    break
            else:
                # No shortcuts found, add the next node
                current = path[i]
                smoothed_path.append(current)
                i += 1
        
        # Make sure goal is in the path
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
            
        return smoothed_path


def main():
    """
    S-Theta*: Low Steering Path-Planning Algorithm
    
    S-Theta* extends Theta* by considering steering constraints and minimizing steering changes.
    This makes it especially suitable for vehicle navigation where smooth paths with minimal
    heading changes are preferred. The algorithm considers not just path length but also
    steering costs to produce paths that are more suitable for vehicles with turning constraints.
    
    Like Theta*, S-Theta* allows for paths that can go through any angle, not just along grid edges.
    It checks for line-of-sight between non-adjacent nodes but adds a penalty for steering changes,
    resulting in smoother, more natural paths with fewer heading changes.
    
    The steering_weight parameter controls how much to prioritize minimizing steering changes
    versus path length. Higher values will result in smoother paths with fewer turns,
    potentially at the cost of slightly longer paths.
    """
    s_start = (5, 5)
    s_goal = (45, 25)
    
    # Create S-Theta* instance with euclidean heuristic and steering weight of 2.0
    s_theta_star = SThetaStar(s_start, s_goal, "euclidean", steering_weight=2.0)
    path, visited = s_theta_star.searching()
    
    # Optional: perform additional path smoothing
    # smoothed_path = s_theta_star.path_smoothing(path)
    # s_theta_star.plot.plot_path(smoothed_path)
    # plt.title("S-Theta* with Additional Path Smoothing")
    # plt.show()


if __name__ == '__main__':
    main()
