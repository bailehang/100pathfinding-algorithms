"""
Weighted_A_star 2D
@author: clark bai
"""

import os
import sys
import math
import heapq

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class WeightedAStar:
    """Weighted A* sets the cost + weighted heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, weight=1.0):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.weight = weight  # weight for heuristic

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        Weighted A* Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def get_neighbor(self, s):
        """
        Find neighbors of state s that are not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) collides with obstacles.
        :param s_start: start node
        :param s_end: end node
        :return: True: collision / False: no collision
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

    def f_value(self, s):
        """
        Calculate f value: f = g + weight * h
        :param s: current state
        :return: f value
        """

        return self.g[s] + self.weight * self.heuristic(s)

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

    def compare_with_standard_astar(self):
        """
        Compare with standard A* by running both algorithms and returning both paths
        :return: weighted_path, standard_path, weighted_visited, standard_visited
        """
        # Run Weighted A*
        weighted_path, weighted_visited = self.searching()
        
        # Run standard A* (weight = 1.0)
        standard_astar = WeightedAStar(self.s_start, self.s_goal, self.heuristic_type, 1.0)
        standard_path, standard_visited = standard_astar.searching()
        
        return weighted_path, standard_path, weighted_visited, standard_visited


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    weight = 2.0  # Example weight, can be adjusted

    weighted_astar = WeightedAStar(s_start, s_goal, "euclidean", weight)
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = weighted_astar.searching()
    plot.animation(path, visited, f"Weighted A* (w={weight})")  # animation
    
    # Uncomment below to run a comparison with standard A*
    # weighted_path, standard_path, weighted_visited, standard_visited = weighted_astar.compare_with_standard_astar()
    # plot.animation_multi_path([weighted_path, standard_path], [weighted_visited, standard_visited], 
    #                         [f"Weighted A* (w={weight})", "Standard A*"])


if __name__ == '__main__':
    main()
