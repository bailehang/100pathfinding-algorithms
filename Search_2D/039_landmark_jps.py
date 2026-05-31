"""
Landmark JPS 2D path planning demo.

This JPS variant combines jump pruning with a mild landmark-biased heuristic.
Purple crosses show the corridor landmarks that pull jump ordering through the
map without changing the obstacle model.
"""

from metrics import install_metrics
install_metrics()

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class LandmarkJPSDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("Landmark JPS", weight=1.12)
        self.landmarks = [(21, 16), (29, 14), (31, 24), (41, 17)]
        self.extra_points = list(self.landmarks)

    def heuristic(self, node):
        octile = super().heuristic(node)
        landmark_bias = min(math.hypot(node[0] - p[0], node[1] - p[1]) for p in self.landmarks)
        return octile + 0.10 * landmark_bias

    def jump_limit(self, current, dx, dy):
        return 10

    def phase_text(self, step, current):
        return "weighted landmark bias orders jump points"

    def title(self):
        return "039 Landmark JPS - weighted landmark jumps"


def main():
    random.seed(39)
    np.random.seed(39)
    planner = LandmarkJPSDemo()
    path = planner.search(save_gif=True, gif_name="039_landmark_jps")
    if not path:
        raise RuntimeError("Landmark JPS did not reach the goal")


if __name__ == "__main__":
    main()
