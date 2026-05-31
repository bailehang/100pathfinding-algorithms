"""
Dynamic JPS 2D path planning demo.

Dynamic JPS updates its pruning information when cells change. This demo inserts
a small temporary obstacle marker during search, clears stale frontier entries,
and continues with fresh jumps around the changed cells.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class DynamicJPSDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("Dynamic JPS", weight=1.02)
        self.changed = False

    def on_expand(self, step, current, snapshots):
        if not self.changed and step > 42:
            for cell in [(24, 18), (25, 18), (26, 18)]:
                self.dynamic_events.append(cell)
            self.changed = True
            snapshots.append(self.snapshot(step, "local map update invalidates nearby jumps"))

    def jump_limit(self, current, dx, dy):
        return 7 if self.changed else 12

    def phase_text(self, step, current):
        return "repair jump rays after dynamic map change" if self.changed else "pre-change jump search"

    def title(self):
        return "037 Dynamic JPS - update and repair jump rays"


def main():
    random.seed(37)
    np.random.seed(37)
    planner = DynamicJPSDemo()
    path = planner.search(save_gif=True, gif_name="037_dynamic_jps")
    if not path:
        raise RuntimeError("Dynamic JPS did not reach the goal")


if __name__ == "__main__":
    main()
