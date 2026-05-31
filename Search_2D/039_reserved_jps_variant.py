"""
Reserved JPS-family slot: weighted landmark JPS demo.

The README keeps 039 as a reserved JPS-family slot, so this implementation uses
a conservative experimental variant: JPS pruning with a mild weighted heuristic
and landmark-like bias points. Purple crosses show the landmarks that pull jump
ordering through the corridor.
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


class ReservedJPSVariantDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("Reserved JPS", weight=1.12)
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
        return "039 Reserved JPS variant - weighted landmark jumps"


def main():
    random.seed(39)
    np.random.seed(39)
    planner = ReservedJPSVariantDemo()
    path = planner.search(save_gif=True, gif_name="039_reserved_jps_variant")
    if not path:
        raise RuntimeError("Reserved JPS variant did not reach the goal")


if __name__ == "__main__":
    main()
