"""Shared helpers for model-based path-following demos."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from Search_2D.curve_demo_utils import GridCurveDemo, wrap_pi


class ReferencePathFollower(GridCurveDemo):
    def __init__(self, s_start=(5, 5), s_goal=(45, 25)):
        super().__init__(s_start, s_goal)
        self.dt = 0.16
        self.speed = 1.75
        self.lookahead = 2.8
        self.reference_path = []
        self.reference_headings = []
        self.tracked_path = []
        self.target_history = []
        self.control_history = []

    def build_reference(self):
        self.raw_path, self.visited = self.a_star_search()
        self.reference_path = self.densify_reference(self.raw_path)
        self.reference_headings = self.compute_reference_headings(self.reference_path)

    @staticmethod
    def densify_reference(path):
        if len(path) < 2:
            return [(float(x), float(y)) for x, y in path]

        samples = []
        for i in range(len(path) - 1):
            start = np.array(path[i], dtype=float)
            end = np.array(path[i + 1], dtype=float)
            distance = np.linalg.norm(end - start)
            count = max(2, int(distance / 0.18))
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                samples.append(tuple(start + (end - start) * t))
        samples.append(tuple(map(float, path[-1])))
        return samples

    @staticmethod
    def compute_reference_headings(path):
        headings = []
        for i, point in enumerate(path):
            if i < len(path) - 1:
                nxt = path[i + 1]
                heading = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
            elif headings:
                heading = headings[-1]
            else:
                heading = 0.0
            headings.append(heading)
        return headings

    def closest_reference_index(self, position, start_index):
        end_index = min(len(self.reference_path), start_index + 55)
        window = self.reference_path[start_index:end_index]
        if not window:
            return len(self.reference_path) - 1
        distances = [self.distance(position, point) for point in window]
        return start_index + int(np.argmin(distances))

    def lookahead_index(self, position, start_index, lookahead=None):
        lookahead = self.lookahead if lookahead is None else lookahead
        index = start_index
        while index < len(self.reference_path) - 1:
            if self.distance(position, self.reference_path[index]) >= lookahead:
                return index
            index += 1
        return len(self.reference_path) - 1

    def reference_state(self, position, theta, closest_index):
        target_index = self.lookahead_index(position, closest_index)
        target = self.reference_path[target_index]
        reference_heading = self.reference_headings[target_index]
        dx = position[0] - target[0]
        dy = position[1] - target[1]
        lateral_error = -math.sin(reference_heading) * dx + math.cos(reference_heading) * dy
        heading_error = wrap_pi(theta - reference_heading)
        return target_index, target, reference_heading, lateral_error, heading_error

    def propagate(self, state, omega, speed=None):
        x, y, theta = state
        velocity = self.speed if speed is None else speed
        theta = wrap_pi(theta + omega * self.dt)
        x += velocity * math.cos(theta) * self.dt
        y += velocity * math.sin(theta) * self.dt
        x = float(np.clip(x, 1.0, self.Env.x_range - 2.0))
        y = float(np.clip(y, 1.0, self.Env.y_range - 2.0))
        return x, y, theta

    def draw_vehicle(self, position, theta, color="gold"):
        heading = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-heading[1], heading[0]])
        center = np.array(position)
        nose = center + heading * 0.85
        left = center - heading * 0.55 + normal * 0.42
        right = center - heading * 0.55 - normal * 0.42
        polygon = np.vstack([nose, left, right, nose])
        plt.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=1.5)
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.9)

    def draw_reference_and_trace(self, history, target=None, prediction=None, trace_color="crimson"):
        self.draw_base(self.frame_title)
        plt.plot(
            [p[0] for p in self.reference_path],
            [p[1] for p in self.reference_path],
            color="tab:blue",
            alpha=0.34,
            linewidth=2.2,
            label="reference",
        )
        plt.plot(
            [p[0] for p in history],
            [p[1] for p in history],
            color=trace_color,
            linewidth=2.8,
            label="tracked",
        )
        if prediction:
            plt.plot(
                [p[0] for p in prediction],
                [p[1] for p in prediction],
                "o-",
                color="tab:purple",
                linewidth=1.8,
                markersize=3,
                alpha=0.75,
                label="prediction",
            )
        if target is not None:
            plt.plot(target[0], target[1], "o", color="tab:orange", markersize=7, label="target")
            plt.plot(
                [history[-1][0], target[0]],
                [history[-1][1], target[1]],
                "--",
                color="tab:orange",
                linewidth=1.1,
                alpha=0.75,
            )


def solve_discrete_lqr(a_matrix, b_matrix, q_matrix, r_matrix, iterations=120):
    p_matrix = q_matrix.copy()
    for _ in range(iterations):
        gain_den = r_matrix + b_matrix.T @ p_matrix @ b_matrix
        gain = np.linalg.solve(gain_den, b_matrix.T @ p_matrix @ a_matrix)
        p_next = q_matrix + a_matrix.T @ p_matrix @ (a_matrix - b_matrix @ gain)
        if np.max(np.abs(p_next - p_matrix)) < 1e-8:
            p_matrix = p_next
            break
        p_matrix = p_next
    return np.linalg.solve(r_matrix + b_matrix.T @ p_matrix @ b_matrix, b_matrix.T @ p_matrix @ a_matrix)
