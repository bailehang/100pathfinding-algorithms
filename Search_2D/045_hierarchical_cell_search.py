"""Hierarchical precomputed cell graph search."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cell_graph_helpers import PrecomputedCellGraph


def main():
    planner = PrecomputedCellGraph("hier", "Hierarchical Cell Search")
    path = planner.search(save_gif=True, gif_name="045_hierarchical_cell_search")
    if not path:
        raise RuntimeError("Hierarchical cell search returned no path")


if __name__ == "__main__":
    main()
