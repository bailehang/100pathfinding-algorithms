"""
Rapidly-Exploring Random Tree (RRT) Algorithm for Search_based_Planning
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import env, plotting_rrt


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()
        self.plotting = plotting_rrt.PlottingRRT(s_start, s_goal)
        self.obs = self.env.obs_map()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and not self.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    return self.extract_path(node_new)

        return None

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() > goal_sample_rate:
            return Node((np.random.randint(1, self.x_range - 1),
                         np.random.randint(1, self.y_range - 1)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

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

    rrt = Rrt(x_start, x_goal, 2.0, 0.05, 10000)
    path = rrt.planning()

    if path:
        # Use original-style animation
        rrt.plotting.animation(rrt.vertex, path, "RRT", True)
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()
