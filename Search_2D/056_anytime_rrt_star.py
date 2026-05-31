"""
Anytime-RRT* 2D path planning demo.

The planner first behaves like RRT* to obtain any feasible route. Once a route
exists, it keeps running, samples near the incumbent path, rewires the tree, and
shortcuts each improved solution. The GIF emphasizes that the best path keeps
changing after the first solution instead of stopping immediately.
"""

from metrics import install_metrics
install_metrics()

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rrt_variant_helpers import RRTVariantDemo, shortcut_path


class AnytimeRRTStar(RRTVariantDemo):
    def __init__(self):
        super().__init__(step_len=1.35, goal_sample_rate=0.13, search_radius=9.0, iter_max=850)
        self.mode_label = "Anytime-RRT*"
        self.improvement_count = 0

    def sample(self, k):
        if self.best_path and random.random() < 0.58:
            return self.sample_near_path(max(0.9, 3.6 - 0.004 * k))
        return super().sample(k)

    def post_process_path(self, path, k):
        if self.best_path:
            return shortcut_path(path, self.utils)
        return path

    def try_update_solution(self, k, node, snapshots):
        before = self.best_cost
        super().try_update_solution(k, node, snapshots)
        if self.best_cost + 0.05 < before:
            self.improvement_count += 1
            radius = max(0.75, 3.8 - 0.22 * self.improvement_count)
            self.extra_shapes = [
                {"kind": "circle", "center": p, "radius": radius, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.10}
                for p in self.best_path[1:-1: max(1, len(self.best_path) // 5)]
            ]

    def phase_text(self, k):
        return "keep sampling near incumbent route" if self.best_path else "search for first feasible route"

    def solution_text(self):
        return "incumbent improved and shortcut"

    def title(self):
        return "056 Anytime-RRT* - keep improving after first route"


def main():
    random.seed(56)
    np.random.seed(56)
    planner = AnytimeRRTStar()
    path = planner.planning(save_gif=True, gif_name="056_anytime_rrt_star")
    if not path:
        raise RuntimeError("Anytime-RRT* did not reach the goal")


if __name__ == "__main__":
    main()
