"""B-Spline path smoothing demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo


class BSplinePlanner(GridCurveDemo):
    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.bspline_path, self.control_points = self.build_collision_checked_bspline()
        return self.bspline_path, self.visited

    def build_collision_checked_bspline(self):
        control_sets = [
            self.waypoints,
            self.raw_path[::4] + [self.raw_path[-1]],
            self.raw_path[::2] + [self.raw_path[-1]],
            self.raw_path,
        ]

        for controls in control_sets:
            controls = self.unique_consecutive(controls)
            for smoothing in (0.0, 0.5, 1.5):
                samples = self.sample_bspline(controls, smoothing=smoothing)
                if samples and not self.samples_collide(samples):
                    return samples, controls

        return [(float(x), float(y)) for x, y in self.raw_path], self.raw_path

    @staticmethod
    def unique_consecutive(points):
        unique = []
        for point in points:
            if not unique or point != unique[-1]:
                unique.append(point)
        return unique

    def sample_bspline(self, controls, smoothing=0.0, sample_count=180):
        if len(controls) < 2:
            return controls
        pts = np.array(controls, dtype=float)
        degree = min(3, len(pts) - 1)
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=degree)
        except ValueError:
            return []

        u = np.linspace(0.0, 1.0, sample_count)
        x, y = splev(u, tck)
        return [(float(px), float(py)) for px, py in zip(x, y)]

    def run_demonstration(self):
        print("Starting B-Spline demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("074 B-Spline - Grid Search", self.visited, self.raw_path)
        self.draw_search_frame("074 B-Spline - Control Polygon", [], self.raw_path, self.control_points)

        for i in range(1, 13):
            upto = max(2, int(len(self.bspline_path) * i / 12))
            visible = self.bspline_path[:upto]
            self.draw_base("074 B-Spline - Smooth Basis Curve")
            plt.plot(
                [p[0] for p in self.control_points],
                [p[1] for p in self.control_points],
                "o--",
                color="tab:purple",
                alpha=0.45,
            )
            plt.plot([p[0] for p in visible], [p[1] for p in visible], color="crimson", linewidth=3)
            self.capture_frame()

        self.save_gif("074_B_Spline", fps=4)
        plt.close("all")
        print(f"Control points: {len(self.control_points)}")
        print(f"B-Spline samples: {len(self.bspline_path)}")
        print(f"B-Spline path length: {self.path_length(self.bspline_path):.3f}")


def main():
    BSplinePlanner().run_demonstration()


if __name__ == "__main__":
    main()
