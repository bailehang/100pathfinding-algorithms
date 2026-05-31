"""
JPS+ 2D path planning demo.

JPS+ precomputes jump directions so runtime search can skip over long symmetric
grid runs. The blue rays in the GIF represent cached jump spans, and orange
diamonds mark jump points discovered by the online search.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import JPSGridDemo


class JPSPlusDemo(JPSGridDemo):
    def __init__(self):
        super().__init__("JPS+")
        self.extra_regions = [
            {"xy": (2, 2), "w": 46, "h": 26, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.06}
        ]

    def jump_limit(self, current, dx, dy):
        return 14

    def phase_text(self, step, current):
        return "cached jump spans skip symmetric cells"

    def title(self):
        return "033 JPS+ - cached jump spans"


def main():
    random.seed(33)
    np.random.seed(33)
    planner = JPSPlusDemo()
    path = planner.search(save_gif=True, gif_name="033_JPS_plus")
    if not path:
        raise RuntimeError("JPS+ did not reach the goal")


if __name__ == "__main__":
    main()
