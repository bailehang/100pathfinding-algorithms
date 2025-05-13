"""
Plotting tools for RRT-based algorithms in Search_2D
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import env


class PlottingRRT:
    def __init__(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal
        self.env = env.Env()
        self.obs = self.env.obs_map()
        
        # Extract boundary, rectangular, and circular obstacles
        self.obs_boundary = []
        self.obs_rectangle = []
        self.obs_circle = []
        
        # Boundary: Add the four edges as rectangles
        x_range = self.env.x_range
        y_range = self.env.y_range
        self.obs_boundary = [
            [0, 0, 1, y_range],
            [0, 0, x_range, 1],
            [0, y_range-1, x_range, 1],
            [x_range-1, 0, 1, y_range]
        ]
        
        # Identify rectangular obstacles (horizontal and vertical lines)
        # Some heuristics to recognize horizontal and vertical lines in the obstacles
        horizontal_lines = {}
        vertical_lines = {}
        
        for (x, y) in self.obs:
            if 0 < x < x_range-1 and 0 < y < y_range-1:  # Not boundary points
                # Check if point is part of horizontal line
                if (x-1, y) in self.obs and (x+1, y) in self.obs:
                    if y not in horizontal_lines:
                        horizontal_lines[y] = []
                    horizontal_lines[y].append(x)
                
                # Check if point is part of vertical line
                if (x, y-1) in self.obs and (x, y+1) in self.obs:
                    if x not in vertical_lines:
                        vertical_lines[x] = []
                    vertical_lines[x].append(y)
        
        # Process horizontal lines to get rectangles
        for y, xs in horizontal_lines.items():
            xs_sorted = sorted(xs)
            start_x = xs_sorted[0]
            prev_x = start_x
            
            for i in range(1, len(xs_sorted)):
                if xs_sorted[i] != prev_x + 1:
                    # End of a segment
                    width = prev_x - start_x + 1
                    self.obs_rectangle.append([start_x, y, width, 1])
                    start_x = xs_sorted[i]
                prev_x = xs_sorted[i]
            
            # Last segment
            width = prev_x - start_x + 1
            self.obs_rectangle.append([start_x, y, width, 1])
        
        # Process vertical lines to get rectangles
        for x, ys in vertical_lines.items():
            ys_sorted = sorted(ys)
            start_y = ys_sorted[0]
            prev_y = start_y
            
            for i in range(1, len(ys_sorted)):
                if ys_sorted[i] != prev_y + 1:
                    # End of a segment
                    height = prev_y - start_y + 1
                    self.obs_rectangle.append([x, start_y, 1, height])
                    start_y = ys_sorted[i]
                prev_y = ys_sorted[i]
            
            # Last segment
            height = prev_y - start_y + 1
            self.obs_rectangle.append([x, start_y, 1, height])
        
        # For simplicity, we're not detecting circular obstacles in this implementation
        # But we could use clustering algorithms to identify them

    def animation(self, nodelist, path, name, animation=True):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)

    def plot_grid(self, name):
        fig, ax = plt.subplots()

        # Plot obstacles from the set
        for (x, y) in self.obs:
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01)
        plt.show()
