"""
Closed-loop RRT* (CL-RRT*) 2D path planning demo.

CL-RRT* extends toward a sampled state by rolling out a simple feedback-tracked
motion primitive instead of drawing an unconstrained straight edge. The orange
arrows show the heading produced by the closed-loop rollout while the green tree
keeps the accepted closed-loop endpoints.
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


class ClosedLoopNode(Node):
    def __init__(self, n, theta=0.0):
        super().__init__(n)
        self.theta = theta
        self.trajectory = []


class ClosedLoopRRTStar(RRTVariantDemo):
    def __init__(self):
        super().__init__(step_len=1.55, goal_sample_rate=0.14, search_radius=8.0, iter_max=820)
        self.x_start = ClosedLoopNode((18, 8), theta=0.55)
        self.x_goal = ClosedLoopNode((37, 18), theta=0.0)
        self.V = [self.x_start]
        self.mode_label = "CL-RRT*"
        self.last_rollout = []

    def steer(self, x_start, x_goal, step_len=None):
        x, y, theta = x_start.x, x_start.y, getattr(x_start, "theta", 0.0)
        target_theta = math.atan2(x_goal.y - y, x_goal.x - x)
        ds = 0.35
        rollout = [(x, y)]

        for _ in range(9):
            heading_error = math.atan2(math.sin(target_theta - theta), math.cos(target_theta - theta))
            theta += max(-0.34, min(0.34, 0.72 * heading_error))
            x += ds * math.cos(theta)
            y += ds * math.sin(theta)
            rollout.append((x, y))
            if math.hypot(x_goal.x - x, x_goal.y - y) < 0.45:
                break

        node = ClosedLoopNode((x, y), theta=theta)
        node.parent = x_start
        node.trajectory = rollout
        self.last_rollout = rollout
        return node

    def near(self, nodelist, node):
        return []

    def rewire(self, x_new, x_near):
        return

    def edge_in_collision(self, start, end):
        trajectory = getattr(end, "trajectory", None)
        if not trajectory:
            return self.utils.is_collision(start, end)
        for i in range(1, len(trajectory)):
            if self.utils.is_collision(Node(trajectory[i - 1]), Node(trajectory[i])):
                return True
        return False

    def try_update_solution(self, k, node, snapshots):
        if self.line(node, self.x_goal) > self.step_len * 2.8:
            return
        goal_rollout = self.steer(node, self.x_goal)
        if self.edge_in_collision(node, goal_rollout):
            return
        if self.line(goal_rollout, self.x_goal) > 1.6:
            return

        path = self.extract_path(goal_rollout) + [(self.x_goal.x, self.x_goal.y)]
        cost = self.path_length(path)
        self.candidate_path = path
        if cost + 0.05 < self.best_cost:
            self.best_cost = cost
            self.best_path = path
            snapshots.append(self.snapshot(k, self.solution_text(), final=False))

    def extract_path(self, node):
        chain = []
        current = node
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            current = current.parent
        chain.reverse()

        path = []
        for item in chain:
            trajectory = getattr(item, "trajectory", None)
            if trajectory:
                if path and trajectory[0] == path[-1]:
                    path.extend(trajectory[1:])
                else:
                    path.extend(trajectory)
            else:
                path.append((item.x, item.y))
        return path

    def planning(self, save_gif=False, gif_name="057_closed_loop_rrt_star"):
        return super().planning(save_gif=save_gif, gif_name=gif_name)

    def after_add_node(self, k, node):
        super().after_add_node(k, node)
        self.extra_shapes = [
            {"kind": "polyline", "points": node.trajectory, "color": "#f97316", "linewidth": 2.0, "alpha": 0.72},
            {"kind": "heading", "pose": (node.x, node.y, node.theta), "color": "#2563eb"},
        ]

    def snapshot(self, iteration, phase, final=False):
        data = super().snapshot(iteration, phase, final=final)
        headings = []
        for node in self.V[-60:]:
            if hasattr(node, "theta"):
                headings.append({"kind": "heading", "pose": (node.x, node.y, node.theta), "length": 0.8, "color": "#2563eb", "alpha": 0.38})
        data["extra_shapes"].extend(headings[-28:])
        return data

    def phase_text(self, k):
        return "roll out feedback-tracked steering primitive"

    def solution_text(self):
        return "closed-loop route improved"

    def title(self):
        return "057 CL-RRT* - closed-loop steering rollouts"


def main():
    random.seed(57)
    np.random.seed(57)
    planner = ClosedLoopRRTStar()
    path = planner.planning(save_gif=True)
    if not path:
        raise RuntimeError("CL-RRT* did not reach the goal")


if __name__ == "__main__":
    main()
