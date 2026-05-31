"""Dubins Curves path planning demo."""

from metrics import install_metrics
install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo, normalized


class DubinsCurves(GridCurveDemo):
    def __init__(self, s_start=(5, 5), s_goal=(45, 25), turning_radius=2.2):
        super().__init__(s_start, s_goal)
        self.turning_radius = turning_radius

    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.dubins_path, self.arc_centers = self.build_dubins_fillets(self.waypoints)
        return self.dubins_path, self.visited

    def build_dubins_fillets(self, waypoints):
        if len(waypoints) < 3:
            return [(float(x), float(y)) for x, y in waypoints], []

        curve = [tuple(map(float, waypoints[0]))]
        centers = []

        for i in range(1, len(waypoints) - 1):
            p_prev = np.array(waypoints[i - 1], dtype=float)
            p = np.array(waypoints[i], dtype=float)
            p_next = np.array(waypoints[i + 1], dtype=float)
            incoming = normalized(p - p_prev)
            outgoing = normalized(p_next - p)
            seg_in = np.linalg.norm(p - p_prev)
            seg_out = np.linalg.norm(p_next - p)
            radius = min(self.turning_radius, seg_in * 0.35, seg_out * 0.35)

            if radius <= 0.25 or np.linalg.norm(incoming + outgoing) < 1e-6:
                curve.append(tuple(p))
                continue

            arc_start = p - incoming * radius
            arc_end = p + outgoing * radius
            turn = np.cross(np.append(incoming, 0.0), np.append(outgoing, 0.0))[2]
            normal = np.array([-incoming[1], incoming[0]]) if turn > 0 else np.array([incoming[1], -incoming[0]])
            center = arc_start + normal * radius

            arc = self.sample_arc(center, arc_start, arc_end, ccw=turn > 0)
            candidate = [curve[-1]] + [tuple(arc_start)] + arc + [tuple(arc_end)]
            if self.samples_collide(candidate):
                curve.append(tuple(p))
                continue

            curve.extend(candidate[1:])
            centers.append(tuple(center))

        curve.append(tuple(map(float, waypoints[-1])))
        return curve, centers

    def sample_arc(self, center, arc_start, arc_end, ccw=True, samples=18):
        a0 = math.atan2(arc_start[1] - center[1], arc_start[0] - center[0])
        a1 = math.atan2(arc_end[1] - center[1], arc_end[0] - center[0])
        if ccw and a1 < a0:
            a1 += 2.0 * math.pi
        if not ccw and a1 > a0:
            a1 -= 2.0 * math.pi
        angles = np.linspace(a0, a1, samples)
        radius = np.linalg.norm(arc_start - center)
        return [(float(center[0] + radius * math.cos(a)), float(center[1] + radius * math.sin(a))) for a in angles]

    def run_demonstration(self):
        print("Starting Dubins Curves demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("076 Dubins Curves - Grid Search", self.visited, self.raw_path)
        self.draw_search_frame("076 Dubins Curves - Waypoint Skeleton", [], self.raw_path, self.waypoints)

        for i in range(1, 13):
            upto = max(2, int(len(self.dubins_path) * i / 12))
            visible = self.dubins_path[:upto]
            self.draw_base("076 Dubins Curves - Forward Line-Arc-Line Path")
            plt.plot(
                [p[0] for p in self.waypoints],
                [p[1] for p in self.waypoints],
                "o--",
                color="tab:orange",
                alpha=0.45,
            )
            for center in self.arc_centers:
                plt.plot(center[0], center[1], "x", color="tab:purple", alpha=0.6)
            plt.plot([p[0] for p in visible], [p[1] for p in visible], color="crimson", linewidth=3)
            self.capture_frame()

        self.save_gif("081_Dubins_Curves", fps=4)
        plt.close("all")
        print(f"Turning radius: {self.turning_radius:.2f}")
        print(f"Dubins samples: {len(self.dubins_path)}")
        print(f"Dubins path length: {self.path_length(self.dubins_path):.3f}")


def main():
    DubinsCurves().run_demonstration()


if __name__ == "__main__":
    main()
