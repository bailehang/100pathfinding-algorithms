"""
Repairing_Astar 2D
@author: clark bai
"""

import os
import sys
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class RepairingAStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()
        self.Plot = plotting.Plotting(self.s_start, self.s_goal)

        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        # Cost to come and parent dictionaries
        self.g = {}  # Cost to come from start
        self.parent = {}  # Parent node mapping

        # Current optimal path
        self.path = []
        self.affected_nodes = set()  # Nodes affected by environment changes

        # Initialize costs
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.g[(i, j)] = float("inf")

        self.g[self.s_start] = 0
        self.parent[self.s_start] = self.s_start
        self.visited = set()
        self.count = 0

        self.fig = plt.figure()

    def run(self):
        """
        Main run function for the interactive planner
        """
        self.Plot.plot_grid("Repairing A*")

        # Initial A* search
        self.path = self.searching()
        self.plot_path(self.path)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)

        plt.show()

    def on_press(self, event):
        """
        Mouse click event handler to add/remove obstacles
        """
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            self.visited = set()
            self.count += 1

            # Update obstacle map
            if (x, y) not in self.obs:
                self.obs.add((x, y))
            else:
                self.obs.remove((x, y))

            self.Plot.update_obs(self.obs)

            # Identify affected nodes
            self.identify_affected_nodes((x, y))
            
            # Repair the path if necessary
            if self.is_path_affected():
                self.repair_path()
            
            # Plot the result
            plt.cla()
            self.Plot.plot_grid("Repairing A*")
            self.plot_visited(self.visited)
            self.plot_path(self.path)
            self.fig.canvas.draw_idle()

    def searching(self):
        """
        Standard A* search
        :return: path
        """
        open_set = {self.s_start}
        closed_set = set()
        
        # Initialize g and parent
        self.g = {}
        self.parent = {}
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.g[(i, j)] = float("inf")
                
        self.g[self.s_start] = 0
        self.parent[self.s_start] = self.s_start
        
        while open_set:
            # Select node with minimum f_value
            s = min(open_set, key=lambda x: self.g[x] + self.h(x))
            self.visited.add(s)
            
            # Check if goal is reached
            if s == self.s_goal:
                return self.extract_path()
                
            # Remove s from open_set and add to closed_set
            open_set.remove(s)
            closed_set.add(s)
            
            # Expand neighbors
            for s_n in self.get_neighbor(s):
                # Skip if in closed set
                if s_n in closed_set:
                    continue
                    
                new_cost = self.g[s] + self.cost(s, s_n)
                
                # Found a better path to neighbor
                if s_n not in open_set or new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.parent[s_n] = s
                    if s_n not in open_set:
                        open_set.add(s_n)
                        
        return []  # No path found

    def identify_affected_nodes(self, changed_node):
        """
        Identify nodes affected by environment changes
        :param changed_node: node where obstacle was added/removed
        """
        self.affected_nodes = set()
        
        # Find nodes that have the changed node as their neighbor
        for i in range(max(0, changed_node[0] - 2), min(self.x, changed_node[0] + 3)):
            for j in range(max(0, changed_node[1] - 2), min(self.y, changed_node[1] + 3)):
                node = (i, j)
                if node in self.parent and node not in self.obs:
                    if changed_node in self.get_neighbor(node) or node == changed_node:
                        self.affected_nodes.add(node)
        
        # Check if any part of the path is affected
        for i in range(len(self.path) - 1):
            s1, s2 = self.path[i], self.path[i + 1]
            if self.is_collision(s1, s2):
                self.affected_nodes.add(s1)
                self.affected_nodes.add(s2)

    def is_path_affected(self):
        """
        Check if the current path is affected by environment changes
        :return: True if affected, False otherwise
        """
        if not self.path:
            return True
            
        # Check if any node in the path is in affected_nodes
        for node in self.path:
            if node in self.affected_nodes:
                return True
                
        # Check if any segment of the path is now in collision
        for i in range(len(self.path) - 1):
            if self.is_collision(self.path[i], self.path[i + 1]):
                return True
                
        return False

    def repair_path(self):
        """
        Repair the existing path by performing a partial search
        """
        # Find the first affected node in the path
        affected_index = -1
        for i, node in enumerate(self.path):
            if node in self.affected_nodes:
                affected_index = i
                break
                
        if affected_index == -1:
            # If no specific node is affected, check path segments
            for i in range(len(self.path) - 1):
                if self.is_collision(self.path[i], self.path[i + 1]):
                    affected_index = i
                    break
        
        if affected_index != -1:
            # Keep the valid part of the path and recompute from the affected node
            valid_path = self.path[:affected_index]
            
            # If valid_path is empty, perform a complete A* search
            if not valid_path:
                self.path = self.searching()
                return
            
            # Otherwise, start search from the last valid node
            new_start = valid_path[-1]
            
            # Initialize search from new_start to goal
            open_set = {new_start}
            closed_set = set()
            
            # Update g values for the search
            temp_g = {}
            temp_parent = {}
            for i in range(self.Env.x_range):
                for j in range(self.Env.y_range):
                    temp_g[(i, j)] = float("inf")
                    
            temp_g[new_start] = self.g[new_start]
            temp_parent[new_start] = self.parent[new_start]
            
            # A* search from new_start to goal
            while open_set and self.s_goal not in closed_set:
                s = min(open_set, key=lambda x: temp_g[x] + self.h(x))
                self.visited.add(s)
                
                # Check if goal is reached
                if s == self.s_goal:
                    # Reconstruct the partial path
                    partial_path = [self.s_goal]
                    while partial_path[-1] != new_start:
                        partial_path.append(temp_parent[partial_path[-1]])
                    partial_path.reverse()
                    
                    # Update the full path
                    self.path = valid_path + partial_path[1:]
                    
                    # Update the parent and g values
                    for i in range(len(partial_path) - 1):
                        self.parent[partial_path[i + 1]] = partial_path[i]
                        self.g[partial_path[i + 1]] = temp_g[partial_path[i + 1]]
                    
                    return
                
                # Remove s from open_set and add to closed_set
                open_set.remove(s)
                closed_set.add(s)
                
                # Expand neighbors
                for s_n in self.get_neighbor(s):
                    if s_n in closed_set:
                        continue
                        
                    new_cost = temp_g[s] + self.cost(s, s_n)
                    
                    if s_n not in open_set or new_cost < temp_g[s_n]:
                        temp_g[s_n] = new_cost
                        temp_parent[s_n] = s
                        if s_n not in open_set:
                            open_set.add(s_n)
            
            # If no path found, perform complete A* search
            self.path = self.searching()
        else:
            # If no affected node found but path is affected, do a complete search
            self.path = self.searching()

    def get_neighbor(self, s):
        """
        Find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        s_list = set()

        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if 0 <= s_next[0] < self.x and 0 <= s_next[1] < self.y and s_next not in self.obs:
                s_list.add(s_next)

        return s_list

    def h(self, s):
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

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        if self.is_collision(s_start, s_goal):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment between s_start and s_end collides with obstacles
        :param s_start: start node
        :param s_end: end node
        :return: True if in collision, False otherwise
        """
        if s_start in self.obs or s_end in self.obs:
            return True

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

    def extract_path(self):
        """
        Extract the path based on the parent dictionary.
        :return: The planning path
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:
            if s not in self.parent:
                return []  # No path found
                
            s = self.parent[s]
            path.append(s)
            
            if s == self.s_start:
                break

        return list(reversed(path))

    def plot_path(self, path):
        if not path:
            return
            
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")

    def plot_visited(self, visited):
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])


def main():
    x_start = (5, 5)
    x_goal = (45, 25)

    repairing_astar = RepairingAStar(x_start, x_goal, "Euclidean")
    repairing_astar.run()


if __name__ == '__main__':
    main()
