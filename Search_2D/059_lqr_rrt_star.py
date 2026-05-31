"""
LQR-RRT* 2D path planning demo.

This compact demo approximates the LQR-RRT* idea with a local linear feedback
steering rollout. Each extension follows a proportional state-feedback rollout
toward the sample, and the blue funnel circles show the local region where that
feedback controller is trusted before the RRT* rewiring step accepts an edge.
"""

from metrics import install_metrics
install_metrics()

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rrt_variant_helpers import Node, RRTVariantDemo


class LQRRRTStar(RRTVariantDemo):
    def __init__(self):
        super().__init__(step_len=1.55, goal_sample_rate=0.15, search_radius=8.8, iter_max=840)
        self.mode_label = "LQR-RRT*"
        self.last_rollout = []
        self.last_funnel = None

    def steer(self, x_start, x_goal, step_len=None):
        state = np.array([x_start.x, x_start.y], dtype=float)
        target = np.array([x_goal.x, x_goal.y], dtype=float)
        rollout = [tuple(state)]
        gain = 0.42
        max_step = 0.52

        for _ in range(7):
            error = target - state
            dist = float(np.linalg.norm(error))
            if dist < 0.35:
                break
            control = gain * error
            norm = float(np.linalg.norm(control))
            if norm > max_step:
                control = control / norm * max_step
            state = state + control
            rollout.append((float(state[0]), float(state[1])))

        node = Node((float(state[0]), float(state[1])))
        node.parent = x_start
        node.lqr_rollout = rollout
        self.last_rollout = rollout
        self.last_funnel = tuple(state)
        return node

    def after_add_node(self, k, node):
        super().after_add_node(k, node)
        self.extra_shapes = [
            {"kind": "polyline", "points": getattr(node, "lqr_rollout", []), "color": "#2563eb", "linewidth": 2.0, "alpha": 0.70},
            {"kind": "circle", "center": (node.x, node.y), "radius": max(0.9, 2.4 - 0.0015 * k), "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.12},
        ]

    def sample(self, k):
        if self.best_path and random.random() < 0.50:
            return self.sample_near_path(2.6)
        return super().sample(k)

    def snapshot(self, iteration, phase, final=False):
        data = super().snapshot(iteration, phase, final=final)
        funnel_nodes = self.V[-44::6]
        for node in funnel_nodes:
            data["extra_shapes"].append(
                {"kind": "circle", "center": (node.x, node.y), "radius": 1.1, "edge": "#2563eb", "face": "#93c5fd", "alpha": 0.07, "linewidth": 1.0}
            )
        return data

    def phase_text(self, k):
        return "extend with local feedback rollout"

    def solution_text(self):
        return "feedback-steered route improved"

    def title(self):
        return "059 LQR-RRT* - local feedback rollouts and funnels"


def main():
    random.seed(59)
    np.random.seed(59)
    planner = LQRRRTStar()
    path = planner.planning(save_gif=True, gif_name="059_lqr_rrt_star")
    if not path:
        raise RuntimeError("LQR-RRT* did not reach the goal")


if __name__ == "__main__":
    main()
