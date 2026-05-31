"""Timed Elastic Band (TEB) path optimization demo."""

from metrics import install_metrics
install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo


class TimedElasticBand(GridCurveDemo):
    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.band = self.initialize_band(self.raw_path)
        self.time_deltas = [1.0 for _ in range(max(0, len(self.band) - 1))]
        self.optimization_history = [list(self.band)]
        self.optimize_band(iterations=32)
        return self.band, self.visited

    def initialize_band(self, path):
        if len(path) <= 2:
            return [(float(x), float(y)) for x, y in path]

        band = [path[0]]
        stride = 3
        for i in range(stride, len(path) - 1, stride):
            band.append(path[i])
        band.append(path[-1])
        return [(float(x), float(y)) for x, y in band]

    def optimize_band(self, iterations=32):
        obstacle_points = np.array(list(self.obs), dtype=float)
        original = np.array(self.band, dtype=float)
        points = original.copy()

        for _ in range(iterations):
            updated = points.copy()
            for i in range(1, len(points) - 1):
                smooth_force = points[i - 1] + points[i + 1] - 2.0 * points[i]
                anchor_force = original[i] - points[i]
                obstacle_force = self.obstacle_repulsion(points[i], obstacle_points)
                updated[i] += 0.24 * smooth_force + 0.10 * anchor_force + 0.55 * obstacle_force
                updated[i][0] = min(max(updated[i][0], 1.0), self.Env.x_range - 2.0)
                updated[i][1] = min(max(updated[i][1], 1.0), self.Env.y_range - 2.0)

            points = self.repair_if_needed(updated, original)
            self.optimization_history.append([tuple(p) for p in points])

        self.band = [tuple(p) for p in points]

    @staticmethod
    def obstacle_repulsion(point, obstacle_points):
        if len(obstacle_points) == 0:
            return np.array([0.0, 0.0])
        deltas = point - obstacle_points
        distances = np.linalg.norm(deltas, axis=1)
        nearest = np.argmin(distances)
        dist = max(distances[nearest], 1e-6)
        influence = 3.0
        if dist >= influence:
            return np.array([0.0, 0.0])
        direction = deltas[nearest] / dist
        strength = (influence - dist) / influence
        return direction * strength

    def repair_if_needed(self, points, original):
        repaired = points.copy()
        for i in range(1, len(points) - 1):
            rounded = (int(round(points[i][0])), int(round(points[i][1])))
            if rounded in self.obs:
                repaired[i] = original[i]

        for i in range(len(repaired) - 1):
            if self.is_collision(repaired[i], repaired[i + 1]):
                if i + 1 < len(repaired) - 1:
                    repaired[i + 1] = original[i + 1]
        return repaired

    def timed_path_length(self):
        return self.path_length(self.band)

    def run_demonstration(self):
        print("Starting Timed Elastic Band demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("075 TEB - Initial A* Path", self.visited, self.raw_path)

        for step in np.linspace(0, len(self.optimization_history) - 1, 14, dtype=int):
            band = self.optimization_history[int(step)]
            self.draw_base(f"075 TEB - Elastic Band Iteration {int(step)}")
            plt.plot(
                [p[0] for p in self.raw_path],
                [p[1] for p in self.raw_path],
                color="tab:blue",
                alpha=0.25,
                linewidth=2,
            )
            plt.plot([p[0] for p in band], [p[1] for p in band], "o-", color="crimson", linewidth=2.5)
            self.capture_frame()

        self.save_gif("080_TEB", fps=4)
        plt.close("all")
        print(f"Band poses: {len(self.band)}")
        print(f"Time intervals: {len(self.time_deltas)}")
        print(f"TEB path length: {self.timed_path_length():.3f}")


def main():
    TimedElasticBand().run_demonstration()


if __name__ == "__main__":
    main()
