"""Adaptive Voronoi Field path planning demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from voronoi_variant_helpers import VoronoiVariantPlanner


class AdaptiveVoronoiFieldDemo(VoronoiVariantPlanner):
    def __init__(self):
        super().__init__("adaptive", "075 Adaptive Voronoi Field")


def main():
    planner = AdaptiveVoronoiFieldDemo()
    path = planner.search(save_gif=True, gif_name="075_adaptive_voronoi_field")
    if not path:
        raise RuntimeError("Adaptive Voronoi Field returned no path")


if __name__ == "__main__":
    main()
