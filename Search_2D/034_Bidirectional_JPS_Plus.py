"""
JPS++ / Bidirectional JPS+ 2D path planning demo.

Two JPS+ frontiers expand at the same time: one from the start and one from the
goal. The GIF uses gray and green explored sets, blue cached jump spans, and a
purple marker for the meeting jump point.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jps_variant_helpers import BidirectionalGridDemo


class BidirectionalJPSPlusDemo(BidirectionalGridDemo):
    def __init__(self):
        super().__init__("JPS++")

    def jump_limit(self, current, dx, dy):
        return 12

    def title(self):
        return "034 JPS++ - bidirectional cached jump search"


def main():
    random.seed(34)
    np.random.seed(34)
    planner = BidirectionalJPSPlusDemo()
    path = planner.search(save_gif=True, gif_name="034_Bidirectional_JPS_Plus")
    if not path:
        raise RuntimeError("JPS++ did not reach the goal")


if __name__ == "__main__":
    main()
