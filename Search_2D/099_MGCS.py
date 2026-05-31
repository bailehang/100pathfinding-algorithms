"""Multi-Agent Graph of Convex Sets (MGCS / MGCS*) demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.convex_multiagent_helpers import ConvexMultiAgentWorld, draw_agent_routes
from Search_2D.remaining_algorithm_helpers import GifRecorder


class MultiAgentGCS:
    def __init__(self):
        self.world = ConvexMultiAgentWorld()
        self.starts = [(4.0, 6.0), (46.0, 24.0), (4.0, 24.0)]
        self.goals = [(46.0, 24.0), (4.0, 6.0), (46.0, 6.0)]
        self.region_paths = []
        self.routes = []
        self.recorder = GifRecorder()
        self.snapshots = []

    def planning(self, save_gif=False):
        start_ms = now_ms()
        region_use = {}
        for agent_id, (start, goal) in enumerate(zip(self.starts, self.goals)):
            start_region = self.world.locate_region(start)
            goal_region = self.world.locate_region(goal)
            penalties = {region_id: count * 5.5 for region_id, count in region_use.items()}
            region_path = self.world.shortest_region_path(start_region, goal_region, penalties)
            for step, region_id in enumerate(region_path):
                region_use[region_id] = region_use.get(region_id, 0) + max(1, 4 - step)
            route = self.world.route_points(start, goal, region_path)
            self.region_paths.append(region_path)
            self.routes.append(route)
            self.snapshots.append((agent_id, list(self.region_paths), list(self.routes)))
        elapsed = now_ms() - start_ms
        print_metrics_for(self.routes, elapsed, source="mgcs")
        if save_gif:
            self.save_gif()
        return self.routes

    def save_gif(self):
        for agent_id, region_paths, routes in self.snapshots:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.world.draw(ax, f"099 MGCS - sequential convex-set coupling agent {agent_id}")
            draw_agent_routes(ax, self.starts, self.goals, routes, upto=1.0)
            text = "\n".join(f"a{i}: regions {path}" for i, path in enumerate(region_paths))
            ax.text(
                1.2,
                28.2,
                text,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 3},
                zorder=8,
            )
            self.recorder.capture()
            plt.close(fig)
        for alpha in [0.25, 0.45, 0.65, 0.85, 1.0]:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.world.draw(ax, "099 MGCS - optimized multi-agent convex corridors")
            draw_agent_routes(ax, self.starts, self.goals, self.routes, upto=alpha)
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("099_MGCS", fps=3, hold=5)


def main():
    routes = MultiAgentGCS().planning(save_gif=True)
    if not routes:
        raise RuntimeError("MGCS failed to produce routes")


if __name__ == "__main__":
    main()
