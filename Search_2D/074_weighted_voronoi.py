"""Weighted Voronoi Diagram path planning demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from voronoi_variant_helpers import VoronoiVariantPlanner


class WeightedVoronoiDemo(VoronoiVariantPlanner):
    def __init__(self):
        super().__init__("weighted", "074 Weighted Voronoi")


def main():
    planner = WeightedVoronoiDemo()
    path = planner.search(save_gif=True, gif_name="074_weighted_voronoi")
    if not path:
        raise RuntimeError("Weighted Voronoi returned no path")


if __name__ == "__main__":
    main()
