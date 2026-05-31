"""
Spline-RRT* 2D path planning demo.

Spline-RRT* uses RRT* to find and rewire a collision-free waypoint route, then
fits a Catmull-Rom spline through the best route when the smoothed curve remains
valid. The GIF keeps the raw orange waypoint route visible while the blue spline
appears as the path is smoothed.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rrt_variant_helpers import Node, RRTVariantDemo, catmull_rom_path, shortcut_path


class SplineRRTStar(RRTVariantDemo):
    def __init__(self):
        super().__init__(step_len=1.3, goal_sample_rate=0.13, search_radius=9.5, iter_max=820)
        self.mode_label = "Spline-RRT*"
        self.raw_path = []
        self.spline_path = []

    def post_process_path(self, path, k):
        self.raw_path = shortcut_path(path, self.utils)
        spline = catmull_rom_path(self.raw_path, samples_per_segment=6)
        if self.path_is_valid(spline):
            self.spline_path = spline
            return spline
        self.spline_path = []
        return self.raw_path

    def path_is_valid(self, path):
        if len(path) < 2:
            return False
        for i in range(1, len(path)):
            if self.utils.is_collision(Node(path[i - 1]), Node(path[i])):
                return False
        return True

    def snapshot(self, iteration, phase, final=False):
        data = super().snapshot(iteration, phase, final=final)
        if self.raw_path:
            data["extra_shapes"].append(
                {"kind": "polyline", "points": self.raw_path, "color": "#f97316", "linewidth": 1.7, "alpha": 0.50, "linestyle": "--"}
            )
        if self.spline_path:
            data["extra_shapes"].append(
                {"kind": "polyline", "points": self.spline_path, "color": "#2563eb", "linewidth": 2.2, "alpha": 0.78}
            )
        return data

    def phase_text(self, k):
        return "rewire waypoints then fit spline" if self.best_path else "grow waypoint tree"

    def solution_text(self):
        return "spline-smoothed route improved"

    def title(self):
        return "058 Spline-RRT* - waypoint tree and smoothed curve"


def main():
    random.seed(58)
    np.random.seed(58)
    planner = SplineRRTStar()
    path = planner.planning(save_gif=True, gif_name="063_spline_rrt_star")
    if not path:
        raise RuntimeError("Spline-RRT* did not reach the goal")


if __name__ == "__main__":
    main()
