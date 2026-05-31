"""Voronoi Field path planning demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from voronoi_variant_helpers import VoronoiVariantPlanner


class VoronoiFieldDemo(VoronoiVariantPlanner):
    def __init__(self):
        super().__init__("field", "072 Voronoi Field")


def main():
    planner = VoronoiFieldDemo()
    path = planner.search(save_gif=True, gif_name="072_voronoi_field")
    if not path:
        raise RuntimeError("Voronoi Field returned no path")


if __name__ == "__main__":
    main()
