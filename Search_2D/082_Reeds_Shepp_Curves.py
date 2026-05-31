"""Reeds-Shepp Curves path planning demo."""

from metrics import install_metrics
install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo, lerp_path, normalized


class ReedsSheppCurves(GridCurveDemo):
    def __init__(self, s_start=(5, 25), s_goal=(45, 5), turning_radius=1.8):
        super().__init__(s_start, s_goal)
        self.turning_radius = turning_radius

    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.rs_path, self.directions, self.cusps = self.build_reeds_shepp_path(self.waypoints)
        return self.rs_path, self.visited

    def build_reeds_shepp_path(self, waypoints):
        if len(waypoints) < 2:
            return [(float(x), float(y)) for x, y in waypoints], [1], []

        path = []
        directions = []
        cusps = []
        middle_segment = max(1, (len(waypoints) - 1) // 2)

        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i], dtype=float)
            end = np.array(waypoints[i + 1], dtype=float)
            direction = -1 if i == middle_segment else 1
            segment = self.sample_reeds_shepp_segment(start, end, reverse=direction < 0)
            if self.samples_collide(segment):
                segment = lerp_path(start, end, samples=26)

            if path:
                segment = segment[1:]
            path.extend(segment)
            directions.extend([direction] * len(segment))

            if direction < 0:
                cusps.append(tuple(start))
                cusps.append(tuple(end))

        return [(float(x), float(y)) for x, y in path], directions, cusps

    def sample_reeds_shepp_segment(self, start, end, reverse=False):
        delta = end - start
        distance = np.linalg.norm(delta)
        if distance == 0:
            return [tuple(start)]

        heading = normalized(delta)
        normal = np.array([-heading[1], heading[0]])
        radius = min(self.turning_radius, distance * 0.25)
        sign = -1.0 if reverse else 1.0
        p1 = start + heading * distance * 0.32 + normal * radius * sign
        p2 = start + heading * distance * 0.68 - normal * radius * sign

        samples = []
        for t in np.linspace(0.0, 1.0, 34):
            point = (
                (1 - t) ** 3 * start
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t ** 2 * p2
                + t ** 3 * end
            )
            samples.append(tuple(point))
        return samples

    def draw_directional_path(self, samples, directions, upto):
        if upto <= 1:
            return

        for i in range(1, upto):
            a = samples[i - 1]
            b = samples[i]
            color = "crimson" if directions[i] > 0 else "royalblue"
            plt.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=3)

    def run_demonstration(self):
        print("Starting Reeds-Shepp Curves demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("077 Reeds-Shepp - Grid Search", self.visited, self.raw_path)
        self.draw_search_frame("077 Reeds-Shepp - Forward/Reverse Anchors", [], self.raw_path, self.waypoints)

        for i in range(1, 13):
            upto = max(2, int(len(self.rs_path) * i / 12))
            self.draw_base("077 Reeds-Shepp Curves - Bidirectional Path")
            plt.plot(
                [p[0] for p in self.waypoints],
                [p[1] for p in self.waypoints],
                "o--",
                color="tab:orange",
                alpha=0.35,
            )
            self.draw_directional_path(self.rs_path, self.directions, upto)
            for cusp in self.cusps:
                plt.plot(cusp[0], cusp[1], marker="x", color="black", markersize=8)
            plt.plot([], [], color="crimson", linewidth=3, label="forward")
            plt.plot([], [], color="royalblue", linewidth=3, label="reverse")
            plt.legend(loc="lower right")
            self.capture_frame()

        self.save_gif("082_Reeds_Shepp_Curves", fps=4)
        plt.close("all")
        print(f"Turning radius: {self.turning_radius:.2f}")
        print(f"Reeds-Shepp samples: {len(self.rs_path)}")
        print(f"Reverse samples: {sum(1 for d in self.directions if d < 0)}")
        print(f"Reeds-Shepp path length: {self.path_length(self.rs_path):.3f}")


def main():
    ReedsSheppCurves().run_demonstration()


if __name__ == "__main__":
    main()
