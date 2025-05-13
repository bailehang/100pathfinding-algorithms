"""
RRT* (RRT Star) Algorithm for Search_based_Planning
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import env, plotting_rrt, queue


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting_rrt.PlottingRRT(x_start, x_goal)
        self.obs = self.env.obs_map()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def planning(self):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 500 == 0:
                print(k)

            if node_new and not self.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])

        # Use original-style animation
        self.plotting.animation(self.vertex, self.path, "RRT*", True)

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                         if not self.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() > goal_sample_rate:
            return Node((np.random.randint(1, self.x_range - 1),
                         np.random.randint(1, self.y_range - 1)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.is_collision(node_new, self.vertex[ind])]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def is_collision(self, node_start, node_end):
        """
        Check if the path between two nodes collides with obstacles
        :param node_start: start node
        :param node_end: end node
        :return: True if the new node is valid, False otherwise
        """
        if self.is_inside_obs(node_end):
            return True

        line = self.get_ray(node_start, node_end)
        for point in line:
            point_tuple = (round(point[0]), round(point[1]))
            if point_tuple in self.obs:
                return True

        return False

    def is_inside_obs(self, node):
        """
        Check if the node is inside any obstacle
        :param node: node
        :return: True if the node is inside obstacle, False otherwise
        """
        point = (round(node.x), round(node.y))
        return point in self.obs

    def get_ray(self, node_start, node_end):
        """
        Get all points in a line between node_start and node_end
        :param node_start: start node
        :param node_end: end node
        :return: list of points in the line
        """
        start = np.array([node_start.x, node_start.y])
        end = np.array([node_end.x, node_end.y])
        
        # Get all points in the line
        dist, theta = self.get_distance_and_angle(node_start, node_end)
        dist = math.ceil(dist)
        
        line = []
        for i in range(dist):
            point = start + i / dist * (end - start)
            line.append(point)
            
        return line


def main():
    x_start = (5, 5)  # Starting node
    x_goal = (45, 25)  # Goal node

    rrt_star = RrtStar(x_start, x_goal, 2.0, 0.10, 10.0, 5000)
    rrt_star.planning()


if __name__ == '__main__':
    main()
