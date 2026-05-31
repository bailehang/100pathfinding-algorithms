"""Pre-generated quadrilateral cell graph search."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cell_graph_helpers import PrecomputedCellGraph


def main():
    planner = PrecomputedCellGraph("quad", "Quadrilateral Cell Graph")
    path = planner.search(save_gif=True, gif_name="042_quad_cell_graph")
    if not path:
        raise RuntimeError("Quadrilateral cell graph search returned no path")


if __name__ == "__main__":
    main()
