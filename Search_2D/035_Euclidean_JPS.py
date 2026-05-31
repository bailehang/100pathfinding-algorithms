"""
Euclidean JPS (EJPS) 2D path planning demo.

EJPS uses Euclidean cost-to-go while keeping JPS-style jump pruning. The search
therefore favors direct diagonal progress; the translucent circles show
Euclidean heuristic bands around the goal.
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


class EuclideanJPSDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("EJPS")
        self.extra_regions = [
            {"xy": (25, 5), "w": 20, "h": 20, "edge": "#f59e0b", "face": "#fbbf24", "alpha": 0.07},
        ]

    def heuristic(self, node):
        return math.hypot(node[0] - self.goal[0], node[1] - self.goal[1])

    def jump_limit(self, current, dx, dy):
        return 11 if dx and dy else 8

    def phase_text(self, step, current):
        return "euclidean heuristic biases diagonal jumps"

    def title(self):
        return "035 Euclidean JPS - diagonal cost-to-go"


def main():
    random.seed(35)
    np.random.seed(35)
    planner = EuclideanJPSDemo()
    path = planner.search(save_gif=True, gif_name="035_Euclidean_JPS")
    if not path:
        raise RuntimeError("Euclidean JPS did not reach the goal")


if __name__ == "__main__":
    main()
