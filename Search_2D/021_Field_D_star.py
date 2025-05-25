"""
Field D* algorithm for 2D path planning
@author: clarkbai 
Based on the work by Dave Ferguson and Anthony Stentz
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

from Search_2D import plotting, env


class FieldDStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env
        self.Plot = plotting.Plotting(s_start, s_goal)

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range

        # Initialize values
        self.g = {}          # Cost to come
        self.rhs = {}        # One-step lookahead value
        self.U = {}          # Priority queue
        self.km = 0          # Accumulated heuristic
        self.path = []       # Final path
        
        # Initialize costs
        for i in range(self.x_range):
            for j in range(self.y_range):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        # Initialize goal
        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.calculate_key(self.s_goal)
        
        # For visualization
        self.visited = set()
        self.count = 0
        self.fig = plt.figure()

    def run(self):
        """
        Main run function to execute the Field D* algorithm
        """
        self.Plot.plot_grid("Field D*")
        self.compute_path()
        self.path = self.extract_field_path()
        self.plot_visited(self.visited)
        self.plot_path(self.path)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def on_press(self, event):
        """
        Handle mouse click events for interactive replanning
        """
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x_range - 1 or y < 0 or y > self.y_range - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: x =", x, ",", "y =", y)

            # Update the obstacle map
            if (x, y) not in self.obs:
                # Add obstacle
                self.obs.add((x, y))
                self.Plot.update_obs(self.obs)
            else:
                # Remove obstacle
                self.obs.remove((x, y))
                self.Plot.update_obs(self.obs)
            
            # Full reset of Field D* search
            self.km = 0
            self.U = {}
            self.visited = set()
            self.count += 1
            
            # Reset all costs
            for i in range(self.x_range):
                for j in range(self.y_range):
                    self.rhs[(i, j)] = float("inf")
                    self.g[(i, j)] = float("inf")
            
            # Re-initialize goal
            self.rhs[self.s_goal] = 0.0
            self.U[self.s_goal] = self.calculate_key(self.s_goal)
            
            # Recompute path from scratch
            self.compute_path()
            self.path = self.extract_field_path()
            
            # Clear the plot and redraw everything
            plt.cla()
            self.Plot.plot_grid("Field D*")
            self.plot_visited(self.visited)
            self.plot_path(self.path)
            self.fig.canvas.draw_idle()

    def compute_path(self):
        """
        Main computation function for Field D*
        """
        while True:
            s, v = self.top_key()
            # print(f"Current state: {s}, Key value: {v}")  # 添加打印语句
            # If we've reached the start node or no path exists
            if v >= self.calculate_key(self.s_start) and \
               self.rhs[self.s_start] == self.g[self.s_start]:
                print("Path computation terminated early.")
                break

            # Pop current node
            k_old = v
            self.U.pop(s)
            self.visited.add(s)

            # Update key
            if k_old < self.calculate_key(s):
                self.U[s] = self.calculate_key(s)
            
            # Lower state
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbors(s):
                    self.update_vertex(x)
            
            # Raise state
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                for x in self.get_neighbors(s):
                    self.update_vertex(x)

    def update_vertex(self, s):
        """
        Update vertex value using the Field D* algorithm
        """
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            
            # Consider all neighbors
            for neighbor in self.get_neighbors(s):
                # Direct cost between s and neighbor
                direct_cost = self.g[neighbor] + self.cost(s, neighbor)
                self.rhs[s] = min(self.rhs[s], direct_cost)
            
            # Consider all paths through cell edges (interpolation)
            for i in range(len(self.u_set)):
                u1 = self.u_set[i]
                u2 = self.u_set[(i + 1) % len(self.u_set)]
                
                # Get the two neighbors that form the edge
                s1 = (s[0] + u1[0], s[1] + u1[1])
                s2 = (s[0] + u2[0], s[1] + u2[1])
                
                # Skip if either neighbor is an obstacle
                if s1 in self.obs or s2 in self.obs:
                    continue
                
                # Calculate interpolated cost
                interp_cost = self.compute_interpolated_cost(s, s1, s2)
                self.rhs[s] = min(self.rhs[s], interp_cost)
        
        # Update priority queue
        if s in self.U:
            self.U.pop(s)
        
        if self.g[s] != self.rhs[s]:
            self.U[s] = self.calculate_key(s)

    def compute_interpolated_cost(self, s, s1, s2):
        """
        Compute the cost of moving from s to the edge between s1 and s2
        using linear interpolation
        """
        # Get costs at the endpoints
        g1 = self.g[s1]
        g2 = self.g[s2]
        
        # If both costs are infinite, no path exists
        if g1 == float("inf") and g2 == float("inf"):
            return float("inf")
        
        # Calculate distances
        d1 = math.hypot(s[0] - s1[0], s[1] - s1[1])
        d2 = math.hypot(s[0] - s2[0], s[1] - s2[1])
        d12 = math.hypot(s1[0] - s2[0], s1[1] - s2[1])
        
        # Optimal interpolation parameter
        if g1 <= g2:
            # If going directly to s1 is cheapest
            b = 0
            cost = g1 + d1
        elif d12 == 0:
            # If s1 and s2 are the same point
            b = 0
            cost = g1 + d1
        else:
            # Optimal interpolation point
            b = min(max((g1 - g2 + d12) / (2 * d12), 0), 1)
            
            # Calculate interpolated point
            ix = s1[0] + b * (s2[0] - s1[0])
            iy = s1[1] + b * (s2[1] - s1[1])
            
            # Calculate distance and cost
            dist_to_interp = math.hypot(s[0] - ix, s[1] - iy)
            interp_cost = (1 - b) * g1 + b * g2
            
            cost = interp_cost + dist_to_interp
        
        return cost

    def extract_field_path(self):
        """
        Extract the path using the Field D* algorithm
        """
        path = [self.s_start]
        s = self.s_start
        
        # Safety counter to prevent infinite loops
        safety_count = 0
        max_iterations = 1000
        
        while s != self.s_goal and safety_count < max_iterations:
            safety_count += 1
            
            # Get all valid neighbors
            neighbors = self.get_neighbors(s)
            if not neighbors:
                break
                
            # Find the neighbor with minimum cost
            min_cost = float("inf")
            next_s = None
            
            # Check all regular neighbors
            for neighbor in neighbors:
                if neighbor in self.obs:
                    continue
                    
                cost = self.g[neighbor] + self.cost(s, neighbor)
                if cost < min_cost:
                    min_cost = cost
                    next_s = neighbor
            
            # Check interpolated paths through cell edges
            for i in range(len(self.u_set)):
                u1 = self.u_set[i]
                u2 = self.u_set[(i + 1) % len(self.u_set)]
                
                # Calculate edge endpoints
                s1 = (s[0] + u1[0], s[1] + u1[1])
                s2 = (s[0] + u2[0], s[1] + u2[1])
                
                # Skip if out of bounds or obstacles
                if (not (0 <= s1[0] < self.x_range and 0 <= s1[1] < self.y_range) or
                    not (0 <= s2[0] < self.x_range and 0 <= s2[1] < self.y_range) or
                    s1 in self.obs or s2 in self.obs):
                    continue
                
                # Get costs at endpoints
                g1, g2 = self.g[s1], self.g[s2]
                if g1 == float("inf") and g2 == float("inf"):
                    continue
                
                # Calculate edge length
                d12 = math.hypot(s1[0] - s2[0], s1[1] - s2[1])
                if d12 <= 0.001:  # Avoid division by near-zero
                    continue
                
                # Calculate interpolation parameter
                b = min(max((g1 - g2 + d12) / (2 * d12), 0), 1)
                
                # Calculate interpolated point
                ix = s1[0] + b * (s2[0] - s1[0])
                iy = s1[1] + b * (s2[1] - s1[1])
                
                # Calculate total cost through interpolated point
                interp_cost = (1 - b) * g1 + b * g2
                dist_to_interp = math.hypot(s[0] - ix, s[1] - iy)
                total_cost = interp_cost + dist_to_interp
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    # Round to the nearest grid cell for visualization
                    next_x = round(ix)
                    next_y = round(iy)
                    next_s = (next_x, next_y)
            
            # If no valid next state, break
            if next_s is None or next_s in self.obs:
                break
                
            # Add to path and continue
            path.append(next_s)
            s = next_s
            
            # Check if we've reached the goal
            if s == self.s_goal:
                break
        
        # If path finding failed to reach the goal
        if s != self.s_goal:
            print("Warning: Path could not reach the goal")
        
        return path

    def calculate_key(self, s):
        """
        Calculate the priority key for a state
        """
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def top_key(self):
        """
        Get the state with the minimum key value from the priority queue
        """
        if not self.U:
            return None, [float("inf"), float("inf")]
        
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        """
        Heuristic function
        """
        if self.heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:  # euclidean
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def cost(self, s_start, s_goal):
        """
        Calculate cost between two adjacent states
        """
        if self.is_collision(s_start, s_goal):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if there is a collision between two states
        """
        if s_start in self.obs or s_end in self.obs:
            return True
        
        # Check diagonal movement
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            
            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def get_neighbors(self, s):
        """
        Get all neighbors of a state
        """
        neighbors = set()
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check if the neighbor is within the grid bounds
            if 0 <= s_next[0] < self.x_range and 0 <= s_next[1] < self.y_range:
                if s_next not in self.obs:
                    neighbors.add(s_next)
        return neighbors

    def plot_path(self, path):
        """
        Plot the path
        """
        print(f"path = {path}")  # Debugging statement
        if not path:
            return
            
        px = [x[0] for x in path]
        py = [x[1] for x in path]
   
        plt.plot(px, py, linewidth=2, color='r')
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        
    def plot_visited(self, visited):
        """
        Plot visited nodes
        """
        colors = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']
                 
        if self.count >= len(colors):
            self.count = 0
            
        for x in visited:
            if x != self.s_start and x != self.s_goal:
                plt.plot(x[0], x[1], marker='o', color=colors[self.count % len(colors)])


def main():
    # Define start and goal positions
    s_start = (5, 5)
    s_goal = (45, 25)
    
    # Create Field D* instance
    field_dstar = FieldDStar(s_start, s_goal, "euclidean")
    
    # Run the algorithm
    field_dstar.run()


if __name__ == '__main__':
    main()
