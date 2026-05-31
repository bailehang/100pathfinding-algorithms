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
            before = [list(route) for route in self.routes]
            merged = self.try_merge(a, b)
            if merged:
                self.merge_history.append(
                    {
                        "saving": saving,
                        "a": a,
                        "b": b,
                        "before": before,
                        "after": [list(route) for route in self.routes],
                    }
                )
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
        initial_routes = [[cid] for cid in self.customers]
        self.draw_routes_frame(
            "084 VRP - start: one customer per route",
            initial_routes,
            message="Depot dispatches one route per customer; each circle label is customer / demand.",
            highlight=None,
        )
        for step, merge in enumerate(self.merge_history, start=1):
            self.draw_routes_frame(
                f"084 VRP - evaluate merge {step}",
                merge["before"],
                message=(
                    f"Try merging customer {merge['a']} and {merge['b']}: "
                    f"distance saving {merge['saving']:.2f}"
                ),
                highlight=(merge["a"], merge["b"]),
            )
            self.draw_routes_frame(
                f"084 VRP - accept merge {step}",
                merge["after"],
                message=(
                    f"Accepted: combined route load stays within vehicle capacity {self.capacity}."
                ),
                highlight=(merge["a"], merge["b"]),
            )

        for progress in np.linspace(0.08, 1.0, 18):
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.draw_base(ax, "084 VRP - final vehicle tours")
            self.draw_all_routes(ax, self.routes, muted=True)
            self.draw_vehicle_progress(ax, progress)
            self.draw_route_panel(ax, self.routes, "Final plan: 3 vehicles serve 8 customers and return to depot.")
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("084_VRP", fps=3, hold=8)
        print(f"VRP route lengths: {[round(polyline_length(self.route_points(route)), 3) for route in self.routes]}")

    def draw_routes_frame(self, title, routes, message, highlight):
        fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
        self.draw_base(ax, title)
        self.draw_all_routes(ax, routes, highlight=highlight)
        self.draw_route_panel(ax, routes, message)
        self.recorder.capture()
        plt.close(fig)

    def draw_base(self, ax, title):
        ax.set_title(title)
        ax.add_patch(plt.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0))
        ax.scatter([self.depot[0]], [self.depot[1]], marker="s", s=110, c="#111827", label="Depot", zorder=6)
        ax.text(self.depot[0] + 0.55, self.depot[1] + 0.55, "Depot", fontsize=8.5, weight="bold", zorder=7)
        for cid, (x, y, demand) in self.customers.items():
            ax.scatter([x], [y], s=64 + demand * 20, c="#facc15", edgecolor="#111827", linewidth=0.9, zorder=6)
            ax.text(x, y, str(cid), ha="center", va="center", fontsize=8.5, weight="bold", zorder=7)
            ax.text(x + 0.55, y + 0.50, f"d={demand}", fontsize=7.5, color="#334155", zorder=7)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_all_routes(self, ax, routes, highlight=None, muted=False):
        highlight = set(highlight or [])
        for route_index, route in enumerate(routes):
            points = self.route_points(route)
            color = COLORS[route_index % len(COLORS)]
            contains_highlight = bool(highlight.intersection(route))
            linewidth = 3.4 if contains_highlight else 2.3
            alpha = 0.38 if muted else (0.96 if contains_highlight or not highlight else 0.28)
            linestyle = "-" if contains_highlight or not highlight else "--"
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                marker="o",
                markersize=3.8,
                linestyle=linestyle,
                zorder=4,
            )
            mid = points[min(len(points) - 2, max(1, len(points) // 2))]
            ax.text(
                mid[0],
                mid[1] - 1.0,
                f"V{route_index + 1} load {self.route_demand(route)}/{self.capacity}",
                fontsize=7.3,
                color=color,
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.78, "pad": 1.5},
                zorder=8,
            )

    def draw_route_panel(self, ax, routes, message):
        x0, y0 = 1.2, 28.6
        ax.text(
            x0,
            y0,
            message,
            fontsize=8.2,
            color="#1f2937",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.93, "pad": 3},
            zorder=10,
        )
        panel_y = 1.6
        ax.text(1.2, panel_y + 6.9, "Route loads", fontsize=8.3, weight="bold", color="#111827", zorder=10)
        for route_index, route in enumerate(routes[:8]):
            load = self.route_demand(route)
            color = COLORS[route_index % len(COLORS)]
            column = route_index // 4
            row = route_index % 4
            x = 1.2 + column * 14.7
            y = panel_y + 5.8 - row * 1.18
            ax.text(x, y, f"V{route_index + 1}: {route}", fontsize=7.1, color="#111827", zorder=10)
            ax.add_patch(plt.Rectangle((x + 6.4, y - 0.08), 4.4, 0.32, facecolor="#e5e7eb", edgecolor="#94a3b8", linewidth=0.5, zorder=9))
            ax.add_patch(plt.Rectangle((x + 6.4, y - 0.08), 4.4 * min(load / self.capacity, 1.0), 0.32, facecolor=color, edgecolor="none", alpha=0.88, zorder=10))
            ax.text(x + 11.1, y - 0.03, f"{load}/{self.capacity}", fontsize=7.0, color="#111827", zorder=10)

    def draw_vehicle_progress(self, ax, progress):
        for route_index, route in enumerate(self.routes):
            points = self.sample_route(self.route_points(route), progress)
            color = COLORS[route_index % len(COLORS)]
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=3.2, alpha=0.95, zorder=8)
            x, y = points[-1]
            ax.scatter([x], [y], marker="D", s=72, color=color, edgecolor="#111827", linewidth=0.8, zorder=9)
            ax.text(x + 0.45, y + 0.45, f"V{route_index + 1}", fontsize=8, weight="bold", color=color, zorder=10)

    def sample_route(self, points, progress):
        total = polyline_length(points)
        target = total * progress
        visible = [points[0]]
        walked = 0.0
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            segment = self.distance(a, b)
            if walked + segment <= target:
                visible.append(b)
                walked += segment
                continue
            if segment > 1e-6:
                t = (target - walked) / segment
                visible.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            break
        return visible


def main():
    paths = VehicleRoutingProblem().planning(save_gif=True)
    if not paths:
        raise RuntimeError("VRP failed to produce routes")


if __name__ == "__main__":
    main()
