"""Windowed Hierarchical Cooperative A* (WHCA*) demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.remaining_algorithm_helpers import COLORS, GifRecorder, GridTools


class WindowedHCAStar:
    def __init__(self):
        self.grid = GridTools()
        self.window = 10
        self.commit = 4
        self.max_windows = 38
        self.starts = [(5, 6), (45, 10), (5, 24), (45, 20)]
        self.goals = [(45, 6), (5, 10), (45, 24), (5, 20)]
        self.positions = list(self.starts)
        self.paths = [[start] for start in self.starts]
        self.guides = [self.grid.astar(start, goal) for start, goal in zip(self.starts, self.goals)]
        self.guide_progress = [0 for _ in self.starts]
        self.snapshots = []
        self.recorder = GifRecorder()

    def planning(self, save_gif=False):
        start_ms = now_ms()
        time = 0
        for window_id in range(self.max_windows):
            reservations = {}
            planned = []
            for agent_id, position in enumerate(self.positions):
                if position == self.goals[agent_id]:
                    plan = [position] * (self.window + 1)
                else:
                    guide = self.guides[agent_id]
                    subgoal_index = min(len(guide) - 1, self.guide_progress[agent_id] + self.window)
                    subgoal = guide[subgoal_index]
                    plan = self.grid.astar(position, subgoal, reservations, start_time=time, horizon=self.window)
                    if len(plan) < self.window + 1:
                        plan = plan + [plan[-1]] * (self.window + 1 - len(plan))
                self.reserve(plan, reservations, time)
                planned.append(plan)

            for offset in range(1, self.commit + 1):
                if all(self.positions[i] == self.goals[i] for i in range(len(self.positions))):
                    break
                next_positions = [planned[i][min(offset, len(planned[i]) - 1)] for i in range(len(planned))]
                self.positions = next_positions
                for i, pos in enumerate(self.positions):
                    self.paths[i].append(pos)
                    self.guide_progress[i] = self.closest_guide_index(i, pos)
                time += 1

            self.snapshots.append(
                {
                    "window": window_id,
                    "time": time,
                    "positions": list(self.positions),
                    "plans": [plan[: self.window + 1] for plan in planned],
                    "guides": [list(guide) for guide in self.guides],
                    "paths": [list(path) for path in self.paths],
                }
            )
            if all(self.positions[i] == self.goals[i] for i in range(len(self.positions))):
                break

        elapsed = now_ms() - start_ms
        print_metrics_for(self.paths, elapsed, source="whca")
        if save_gif:
            self.save_gif()
        return self.paths

    def closest_guide_index(self, agent_id, position):
        guide = self.guides[agent_id]
        start = self.guide_progress[agent_id]
        end = min(len(guide), start + self.window + self.commit + 4)
        distances = [self.grid.heuristic(position, point) for point in guide[start:end]]
        if not distances:
            return len(guide) - 1
        return start + int(np.argmin(distances))

    @staticmethod
    def reserve(plan, reservations, start_time):
        for offset, pos in enumerate(plan):
            t = start_time + offset
            reservations.setdefault(t, set()).add(pos)
            if offset > 0:
                reservations.setdefault(("edge", t), set()).add((plan[offset - 1], pos))

    def save_gif(self):
        for snapshot in self.snapshots:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.grid.draw_grid(ax, f"096 WHCA* - rolling reservation window {snapshot['window']}")
            for agent_id, (start, goal) in enumerate(zip(self.starts, self.goals)):
                color = COLORS[agent_id]
                path = snapshot["paths"][agent_id]
                plan = snapshot["plans"][agent_id]
                guide = snapshot["guides"][agent_id]
                ax.plot([p[0] for p in guide], [p[1] for p in guide], ":", color=color, linewidth=1.0, alpha=0.28)
                ax.plot([p[0] for p in plan], [p[1] for p in plan], "--", color=color, linewidth=1.3, alpha=0.45)
                ax.plot([p[0] for p in path], [p[1] for p in path], "-", color=color, linewidth=2.4, alpha=0.9)
                ax.scatter([start[0]], [start[1]], marker="s", s=38, color=color, edgecolor="#111827", zorder=5)
                ax.scatter([goal[0]], [goal[1]], marker="*", s=85, color=color, edgecolor="#111827", zorder=5)
                pos = snapshot["positions"][agent_id]
                ax.add_patch(plt.Circle(pos, 0.55, color=color, ec="#111827", zorder=6))
                ax.text(pos[0] + 0.35, pos[1] + 0.35, str(agent_id), fontsize=8, color="#111827")
            ax.text(
                1.0,
                29.0,
                f"time {snapshot['time']}  horizon {self.window}  commit {self.commit}\nsolid = executed path, dashed = current window plan",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 3},
            )
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("096_WHCA_star", fps=3, hold=5)


def main():
    paths = WindowedHCAStar().planning(save_gif=True)
    if not paths:
        raise RuntimeError("WHCA* failed to produce paths")


if __name__ == "__main__":
    main()
