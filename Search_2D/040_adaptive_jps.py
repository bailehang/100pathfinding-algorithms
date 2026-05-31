"""
Adaptive JPS 2D path planning demo.

Adaptive JPS changes its jump horizon according to search context: long jumps in
open space, short jumps near obstacles and late in the search. The GIF shows
how this adaptivity keeps the search fast while adding detail around corners.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class AdaptiveJPSDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("Adaptive JPS", weight=1.0)

    def jump_limit(self, current, dx, dy):
        near_obs = any((current[0] + ox, current[1] + oy) in self.obs for ox in range(-2, 3) for oy in range(-2, 3))
        if near_obs:
            return 4
        if len(self.closed) > 90:
            return 7
        return 14

    def on_expand(self, step, current, snapshots):
        if self.jump_limit(current, 1, 0) <= 4:
            self.extra_points.append(current)

    def phase_text(self, step, current):
        return "short jumps near obstacles, long jumps in open space"

    def title(self):
        return "040 Adaptive JPS - context-sensitive jump horizon"


def main():
    random.seed(40)
    np.random.seed(40)
    planner = AdaptiveJPSDemo()
    path = planner.search(save_gif=True, gif_name="040_adaptive_jps")
    if not path:
        raise RuntimeError("Adaptive JPS did not reach the goal")


if __name__ == "__main__":
    main()
