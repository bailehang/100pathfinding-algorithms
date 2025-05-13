"""
RRT-Connect Algorithm for Search_based_Planning
@author: huiming zhou
"""

import os
import sys
import math
import copy
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


class RrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = env.Env()
        self.plotting = plotting_rrt.PlottingRRT(s_start, s_goal)
        self.obs = self.env.obs_map()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.x == node_new.x and \
                node_new_prim.y == node_new.y:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        if np.random.random() > goal_sample_rate:
            return Node((np.random.randint(1, self.x_range - 1),
                         np.random.randint(1, self.y_range - 1)))

        return sample_goal

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

    def extract_path(self, node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

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

    rrt_conn = RrtConnect(x_start, x_goal, 2.0, 0.05, 5000)
    path = rrt_conn.planning()

    if path:
        # Use original-style connect animation which shows the bidirectional trees
        rrt_conn.plotting.animation_connect(rrt_conn.V1, rrt_conn.V2, path, "RRT-Connect")
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()
