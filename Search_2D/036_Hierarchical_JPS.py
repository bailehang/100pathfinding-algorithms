"""
Hierarchical JPS (HJPS) 2D path planning demo.

HJPS reasons over coarse regions first, then uses JPS jumps inside the active
region corridor. The GIF overlays coarse blocks and portal-like jump points so
the hierarchy is visible during the grid search.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class HierarchicalJPSDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("HJPS", weight=1.05)
        self.extra_regions = [
            {"xy": (1, 1), "w": 16, "h": 14, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.08},
            {"xy": (17, 1), "w": 13, "h": 14, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.08},
            {"xy": (21, 15), "w": 9, "h": 14, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.08},
            {"xy": (31, 16), "w": 18, "h": 13, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.08},
        ]

    def jump_limit(self, current, dx, dy):
        return 16 if current[0] in (20, 30, 40) or current[1] in (15, 16) else 9

    def on_expand(self, step, current, snapshots):
        if current[0] in (20, 30, 40) or current[1] in (15, 16):
            self.extra_points.append(current)

    def phase_text(self, step, current):
        return "coarse region portals guide local jumps"

    def title(self):
        return "036 Hierarchical JPS - regions and portals"


def main():
    random.seed(36)
    np.random.seed(36)
    planner = HierarchicalJPSDemo()
    path = planner.search(save_gif=True, gif_name="036_Hierarchical_JPS")
    if not path:
        raise RuntimeError("Hierarchical JPS did not reach the goal")


if __name__ == "__main__":
    main()
