"""Pre-generated irregular polygon cell graph search."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cell_graph_helpers import PrecomputedCellGraph


def main():
    planner = PrecomputedCellGraph("poly", "Irregular Polygon Cell Graph")
    path = planner.search(save_gif=True, gif_name="044_polygon_cell_graph")
    if not path:
        raise RuntimeError("Polygon cell graph search returned no path")


if __name__ == "__main__":
    main()
