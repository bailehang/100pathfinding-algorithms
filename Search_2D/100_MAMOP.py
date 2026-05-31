"""Multi-Agent Multi-Objective Planning (MAMOP) demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.convex_multiagent_helpers import ConvexMultiAgentWorld, draw_agent_routes, total_route_length
from Search_2D.remaining_algorithm_helpers import GifRecorder


class MultiAgentMultiObjectivePlanning:
    def __init__(self):
        self.world = ConvexMultiAgentWorld()
        self.starts = [(4.0, 6.0), (46.0, 24.0), (4.0, 24.0)]
        self.goals = [(46.0, 24.0), (4.0, 6.0), (46.0, 6.0)]
        self.options = []
        self.chosen = None
        self.recorder = GifRecorder()

    def planning(self, save_gif=False):
        start_ms = now_ms()
        self.options = [
            self.build_option("short", distance_weight=1.0, risk_weight=0.0, congestion_weight=0.0),
            self.build_option("safe", distance_weight=1.0, risk_weight=10.0, congestion_weight=0.0),
            self.build_option("balanced", distance_weight=1.0, risk_weight=4.0, congestion_weight=5.0),
        ]
        self.chosen = min(self.options, key=lambda option: option["score"])
        elapsed = now_ms() - start_ms
        print_metrics_for(self.chosen["routes"], elapsed, source="mamop")
        if save_gif:
            self.save_gif()
        return self.chosen["routes"]

    def build_option(self, name, distance_weight, risk_weight, congestion_weight):
        region_use = {}
        routes = []
        region_paths = []
        total_risk = 0.0
        congestion = 0.0
        for start, goal in zip(self.starts, self.goals):
            start_region = self.world.locate_region(start)
            goal_region = self.world.locate_region(goal)
            penalties = {}
            for region_id, count in region_use.items():
                penalties[region_id] = congestion_weight * count
            if risk_weight > 0.0:
                for region in self.world.regions:
                    penalties[region.id] = penalties.get(region.id, 0.0) + risk_weight * self.world.route_risk([region.center])
            region_path = self.world.shortest_region_path(start_region, goal_region, penalties)
            route = self.world.route_points(start, goal, region_path)
            routes.append(route)
            region_paths.append(region_path)
            total_risk += self.world.route_risk(route)
            for region_id in region_path:
                congestion += region_use.get(region_id, 0)
                region_use[region_id] = region_use.get(region_id, 0) + 1
        distance = total_route_length(routes)
        score = distance_weight * distance + risk_weight * total_risk + congestion_weight * congestion
        return {
            "name": name,
            "routes": routes,
            "region_paths": region_paths,
            "distance": distance,
            "risk": total_risk,
            "congestion": congestion,
            "score": score,
        }

    def save_gif(self):
        for option in self.options:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.world.draw(ax, f"100 MAMOP - candidate objective: {option['name']}")
            draw_agent_routes(ax, self.starts, self.goals, option["routes"], upto=1.0)
            ax.text(
                1.2,
                28.1,
                (
                    f"distance {option['distance']:.1f}\n"
                    f"risk {option['risk']:.2f}\n"
                    f"shared-region cost {option['congestion']:.1f}\n"
                    f"weighted score {option['score']:.1f}"
                ),
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 3},
                zorder=8,
            )
            self.recorder.capture()
            plt.close(fig)

        for alpha in [0.25, 0.45, 0.65, 0.85, 1.0]:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.world.draw(ax, f"100 MAMOP - selected Pareto compromise: {self.chosen['name']}")
            draw_agent_routes(ax, self.starts, self.goals, self.chosen["routes"], upto=alpha)
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("100_MAMOP", fps=3, hold=6)


def main():
    routes = MultiAgentMultiObjectivePlanning().planning(save_gif=True)
    if not routes:
        raise RuntimeError("MAMOP failed to produce routes")


if __name__ == "__main__":
    main()
