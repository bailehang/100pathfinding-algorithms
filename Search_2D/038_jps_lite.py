"""
JPS-Lite 2D path planning demo.

JPS-Lite keeps the core forced-neighbor pruning but uses shorter bounded jumps
and less bookkeeping. The GIF therefore shows more local jump points than full
JPS+, while still pruning many symmetric grid cells.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class JPSLiteDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("JPS-Lite", weight=1.08)

    def jump_limit(self, current, dx, dy):
        return 4 if dx and dy else 5

    def phase_text(self, step, current):
        return "bounded local jumps with minimal bookkeeping"

    def title(self):
        return "038 JPS-Lite - bounded jump pruning"


def main():
    random.seed(38)
    np.random.seed(38)
    planner = JPSLiteDemo()
    path = planner.search(save_gif=True, gif_name="038_jps_lite")
    if not path:
        raise RuntimeError("JPS-Lite did not reach the goal")


if __name__ == "__main__":
    main()
