"""Fuzzy Voronoi Diagram path planning demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from voronoi_variant_helpers import VoronoiVariantPlanner


class FuzzyVoronoiDemo(VoronoiVariantPlanner):
    def __init__(self):
        super().__init__("fuzzy", "074 Fuzzy Voronoi")


def main():
    planner = FuzzyVoronoiDemo()
    path = planner.search(save_gif=True, gif_name="074_fuzzy_voronoi")
    if not path:
        raise RuntimeError("Fuzzy Voronoi returned no path")


if __name__ == "__main__":
    main()
