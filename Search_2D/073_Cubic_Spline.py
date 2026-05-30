"""Cubic Spline path smoothing demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo


class CubicSplinePlanner(GridCurveDemo):
    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.spline_path, self.spline_anchors = self.build_collision_checked_spline()
        return self.spline_path, self.visited

    def build_collision_checked_spline(self):
        anchor_sets = [
            self.waypoints,
            self.raw_path[::4] + [self.raw_path[-1]],
            self.raw_path[::2] + [self.raw_path[-1]],
            self.raw_path,
        ]

        for anchors in anchor_sets:
            anchors = self.unique_consecutive(anchors)
            samples = self.sample_cubic_spline(anchors)
            if samples and not self.samples_collide(samples):
                return samples, anchors

        return [(float(x), float(y)) for x, y in self.raw_path], self.raw_path

    @staticmethod
    def unique_consecutive(points):
        unique = []
        for point in points:
            if not unique or point != unique[-1]:
                unique.append(point)
        return unique

    def sample_cubic_spline(self, anchors, sample_count=180):
        if len(anchors) < 2:
            return anchors

        pts = np.array(anchors, dtype=float)
        chord = np.zeros(len(pts))
        chord[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        if chord[-1] == 0:
            return [tuple(pts[0])]

        spline_x = CubicSpline(chord, pts[:, 0], bc_type="natural")
        spline_y = CubicSpline(chord, pts[:, 1], bc_type="natural")
        ts = np.linspace(0.0, chord[-1], sample_count)
        return [(float(spline_x(t)), float(spline_y(t))) for t in ts]

    def run_demonstration(self):
        print("Starting Cubic Spline demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("073 Cubic Spline - Grid Search", self.visited, self.raw_path)
        self.draw_search_frame("073 Cubic Spline - Spline Anchors", [], self.raw_path, self.spline_anchors)

        for i in range(1, 13):
            upto = max(2, int(len(self.spline_path) * i / 12))
            visible = self.spline_path[:upto]
            self.draw_base("073 Cubic Spline - Natural Cubic Trajectory")
            plt.plot(
                [p[0] for p in self.spline_anchors],
                [p[1] for p in self.spline_anchors],
                "o--",
                color="tab:orange",
                alpha=0.45,
            )
            plt.plot([p[0] for p in visible], [p[1] for p in visible], color="crimson", linewidth=3)
            self.capture_frame()

        self.save_gif("073_Cubic_Spline", fps=4)
        plt.close("all")
        print(f"Spline anchors: {len(self.spline_anchors)}")
        print(f"Spline samples: {len(self.spline_path)}")
        print(f"Spline path length: {self.path_length(self.spline_path):.3f}")


def main():
    CubicSplinePlanner().run_demonstration()


if __name__ == "__main__":
    main()
