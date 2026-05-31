"""Vehicle Routing Problem (VRP) savings-heuristic demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.remaining_algorithm_helpers import COLORS, GifRecorder, polyline_length


class VehicleRoutingProblem:
    def __init__(self):
        self.depot = (6.0, 15.0)
        self.customers = {
            1: (13.0, 24.0, 2),
            2: (18.5, 9.0, 3),
            3: (25.0, 21.0, 2),
            4: (29.0, 6.5, 4),
            5: (35.0, 16.0, 2),
            6: (42.0, 25.0, 3),
            7: (45.0, 8.0, 2),
            8: (23.0, 14.0, 1),
        }
        self.capacity = 7
        self.routes = [[cid] for cid in self.customers]
        self.merge_history = []
        self.recorder = GifRecorder()

    def planning(self, save_gif=False):
        start_ms = now_ms()
        savings = self.compute_savings()
        for saving, a, b in savings:
            merged = self.try_merge(a, b)
            if merged:
                self.merge_history.append((saving, a, b, [list(route) for route in self.routes]))
            if len(self.routes) <= 3:
                break
        elapsed = now_ms() - start_ms
        paths = [self.route_points(route) for route in self.routes]
        print_metrics_for(paths, elapsed, source="vrp")
        if save_gif:
            self.save_gif()
        return paths

    def compute_savings(self):
        savings = []
        for i in self.customers:
            for j in self.customers:
                if i >= j:
                    continue
                saving = self.distance(self.depot, self.xy(i)) + self.distance(self.depot, self.xy(j)) - self.distance(self.xy(i), self.xy(j))
                savings.append((saving, i, j))
        return sorted(savings, reverse=True)

    def try_merge(self, a, b):
        route_a = next((route for route in self.routes if a in route), None)
        route_b = next((route for route in self.routes if b in route), None)
        if route_a is None or route_b is None or route_a is route_b:
            return False
        if self.route_demand(route_a) + self.route_demand(route_b) > self.capacity:
            return False

        candidates = []
        if route_a[-1] == a and route_b[0] == b:
            candidates.append(route_a + route_b)
        if route_a[0] == a and route_b[-1] == b:
            candidates.append(route_b + route_a)
        if route_a[0] == a and route_b[0] == b:
            candidates.append(list(reversed(route_a)) + route_b)
        if route_a[-1] == a and route_b[-1] == b:
            candidates.append(route_a + list(reversed(route_b)))
        if not candidates:
            return False

        merged = min(candidates, key=lambda route: polyline_length(self.route_points(route)))
        self.routes.remove(route_a)
        self.routes.remove(route_b)
        self.routes.append(merged)
        self.routes.sort(key=lambda route: route[0])
        return True

    def route_points(self, route):
        return [self.depot] + [self.xy(cid) for cid in route] + [self.depot]

    def route_demand(self, route):
        return sum(self.customers[cid][2] for cid in route)

    def xy(self, customer_id):
        x, y, _ = self.customers[customer_id]
        return x, y

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def save_gif(self):
        histories = [(0.0, None, None, [[cid] for cid in self.customers])] + self.merge_history
        for step, (saving, a, b, routes) in enumerate(histories):
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            ax.set_title(f"084 VRP - Clarke-Wright savings step {step}")
            ax.add_patch(plt.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0))
            ax.scatter([self.depot[0]], [self.depot[1]], marker="s", s=92, c="#111827", label="depot", zorder=4)
            for cid, (x, y, demand) in self.customers.items():
                ax.scatter([x], [y], s=54 + demand * 18, c="#facc15", edgecolor="#111827", zorder=5)
                ax.text(x + 0.45, y + 0.35, f"{cid}/{demand}", fontsize=8)
            for route_index, route in enumerate(routes):
                points = self.route_points(route)
                color = COLORS[route_index % len(COLORS)]
                ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color=color, linewidth=2.3, alpha=0.8, label=f"veh {route_index + 1}")
            if a is not None:
                ax.text(
                    1.5,
                    28.0,
                    f"merge customer {a} + {b}, saving {saving:.2f}\ncapacity {self.capacity}, route demands {[self.route_demand(r) for r in routes]}",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 3},
                )
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 30)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="lower right", fontsize=7)
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("084_VRP", fps=2, hold=6)
        print(f"VRP route lengths: {[round(polyline_length(self.route_points(route)), 3) for route in self.routes]}")


def main():
    paths = VehicleRoutingProblem().planning(save_gif=True)
    if not paths:
        raise RuntimeError("VRP failed to produce routes")


if __name__ == "__main__":
    main()
