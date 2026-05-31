"""
State lattice planning with motion primitives.

This is not a search over pre-generated square cells. Each state is
``(x, y, heading)``, and edges are short motion primitives that respect a
turning radius. The GIF shows the expanding pose lattice, heading arrows, and
the final kinematically feasible route.
"""

from metrics import install_metrics
install_metrics()

import heapq
import io
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image


class StateLatticePlanner:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.start = (6, 5, 0)
        self.goal_xy = (44, 24)
        self.heading_count = 16
        self.step_len = 2.0
        self.turns = (-1, 0, 1)
        self.obs_rect = [(15, 11, 10, 3), (25, 5, 3, 13), (33, 15, 10, 3)]
        self.obs_circle = [(10, 20, 3), (39, 8, 3), (42, 23, 2)]
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path = []
        self.current_edge = None

    def search(self, save_gif=False, gif_name="041_lattice_planning"):
        start = self.discretize(self.start)
        self.open_heap = [(self.heuristic(start), 0.0, start)]
        self.open_set = {start}
        self.closed = []
        self.closed_set = set()
        self.parent = {start: start}
        self.g = {start: 0.0}
        snapshots = [self.snapshot(0, "initialize pose lattice")]

        for step in range(1, 1800):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)

            if self.reached_goal(current):
                self.path = self.extract_path(current)
                snapshots.append(self.snapshot(step, "goal pose reached", final=True))
                break

            for neighbor, cost, primitive in self.successors(current):
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.current_edge = primitive
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))

            if step < 35 or step % 18 == 0:
                snapshots.append(self.snapshot(step, "expand heading-aware motion primitives"))

        if not self.path:
            raise RuntimeError("State lattice planner did not reach the goal")
        snapshots.append(self.snapshot(len(self.closed), "final motion-primitive route", final=True))
        if save_gif:
            self.save_gif(snapshots, gif_name)
        return [(x, y) for x, y, _ in self.path]

    def successors(self, state):
        x, y, h = state
        result = []
        for turn in self.turns:
            nh = (h + turn) % self.heading_count
            theta = self.heading_angle(nh)
            nx = x + self.step_len * math.cos(theta)
            ny = y + self.step_len * math.sin(theta)
            neighbor = self.discretize((nx, ny, nh))
            primitive = self.rollout(state, neighbor)
            if not self.in_bounds(neighbor) or self.primitive_collision(primitive):
                continue
            turn_penalty = 0.22 if turn else 0.0
            result.append((neighbor, self.path_length_2d(primitive) + turn_penalty, primitive))
        return result

    def rollout(self, start, end):
        sx, sy, sh = start
        ex, ey, eh = end
        points = []
        for i in range(1, 8):
            t = i / 7.0
            theta = self.blend_angle(self.heading_angle(sh), self.heading_angle(eh), t)
            x = sx + (ex - sx) * t
            y = sy + (ey - sy) * t
            bow = math.sin(t * math.pi) * 0.28 * (eh - sh)
            points.append((x - math.sin(theta) * bow, y + math.cos(theta) * bow, theta))
        return [(sx, sy, self.heading_angle(sh))] + points

    def primitive_collision(self, primitive):
        return any(self.inside_obstacle(x, y) for x, y, _ in primitive)

    def inside_obstacle(self, x, y):
        if x <= 1 or y <= 1 or x >= 49 or y >= 29:
            return True
        for ox, oy, w, h in self.obs_rect:
            if ox - 0.35 <= x <= ox + w + 0.35 and oy - 0.35 <= y <= oy + h + 0.35:
                return True
        for ox, oy, r in self.obs_circle:
            if math.hypot(x - ox, y - oy) <= r + 0.35:
                return True
        return False

    def reached_goal(self, state):
        return math.hypot(state[0] - self.goal_xy[0], state[1] - self.goal_xy[1]) < 2.2

    def heuristic(self, state):
        dx = state[0] - self.goal_xy[0]
        dy = state[1] - self.goal_xy[1]
        heading = self.heading_angle(state[2])
        goal_heading = math.atan2(self.goal_xy[1] - state[1], self.goal_xy[0] - state[0])
        heading_error = abs(math.atan2(math.sin(goal_heading - heading), math.cos(goal_heading - heading)))
        return math.hypot(dx, dy) + 0.6 * heading_error

    def discretize(self, pose):
        x, y, h = pose
        return (round(x * 2) / 2.0, round(y * 2) / 2.0, int(round(h)) % self.heading_count)

    def in_bounds(self, state):
        return self.x_range[0] < state[0] < self.x_range[1] and self.y_range[0] < state[1] < self.y_range[1]

    def heading_angle(self, h):
        return 2.0 * math.pi * h / self.heading_count

    @staticmethod
    def blend_angle(a, b, t):
        d = math.atan2(math.sin(b - a), math.cos(b - a))
        return a + d * t

    def extract_path(self, state):
        path = []
        current = state
        seen = set()
        while current in self.parent and current not in seen:
            seen.add(current)
            path.append(current)
            parent = self.parent[current]
            if parent == current:
                break
            current = parent
        return list(reversed(path))

    def snapshot(self, step, phase, final=False):
        hint = self.extract_path(self.closed[-1]) if self.closed else []
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "open": list(self.open_set)[-260:],
            "closed": list(self.closed[-520:]),
            "hint": hint,
            "path": list(self.path),
            "current_edge": self.current_edge,
        }

    def save_gif(self, snapshots, gif_name, max_frames=48):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 4)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=380, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
        self.draw_environment(ax)
        if snapshot["closed"]:
            ax.scatter([p[0] for p in snapshot["closed"]], [p[1] for p in snapshot["closed"]], s=12, color="#94a3b8", alpha=0.62, zorder=2)
        if snapshot["open"]:
            ax.scatter([p[0] for p in snapshot["open"]], [p[1] for p in snapshot["open"]], s=10, color="#22c55e", alpha=0.72, zorder=2)
        self.draw_pose_arrows(ax, snapshot["closed"][-80:], "#64748b", 0.38)
        self.draw_state_path(ax, snapshot["hint"], "#f97316", 2.0, 0.78)
        if snapshot["current_edge"]:
            ax.plot([p[0] for p in snapshot["current_edge"]], [p[1] for p in snapshot["current_edge"]], color="#2563eb", linewidth=2.0, alpha=0.65, zorder=4)
        if snapshot["path"]:
            self.draw_state_path(ax, snapshot["path"], "#d62728", 3.0, 0.94)
        ax.scatter(self.start[0], self.start[1], marker="s", s=76, color="#2b6cb0", zorder=6)
        ax.scatter(self.goal_xy[0], self.goal_xy[1], marker="s", s=76, color="#2f855a", zorder=6)
        ax.text(
            1.4,
            28.4,
            f"State lattice  step {snapshot['step']:3d}  open {len(snapshot['open']):3d}  closed {len(snapshot['closed']):3d}\n{snapshot['phase']}  state=(x,y,heading)",
            fontsize=8.5,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.88, "pad": 3},
        )
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("041 State Lattice Planning - motion primitives")
        fig.tight_layout(pad=0.35)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax):
        for ox, oy, w, h in self.obs_rect:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="#444444", facecolor="#9da3a6"))
        for ox, oy, r in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), r, edgecolor="#444444", facecolor="#9da3a6"))
        ax.add_patch(patches.Rectangle((0, 0), 50, 30, edgecolor="black", facecolor="none", linewidth=6))

    def draw_pose_arrows(self, ax, states, color, alpha):
        for x, y, h in states[::max(1, len(states) // 45 or 1)]:
            theta = self.heading_angle(h)
            ax.arrow(x, y, math.cos(theta) * 0.65, math.sin(theta) * 0.65, head_width=0.22, head_length=0.25, color=color, alpha=alpha, linewidth=0.8, zorder=3)

    def draw_state_path(self, ax, path, color, linewidth, alpha):
        if len(path) < 2:
            return
        ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=linewidth, alpha=alpha, zorder=5)

    @staticmethod
    def path_length_2d(points):
        return sum(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points)))


def main():
    random.seed(41)
    np.random.seed(41)
    planner = StateLatticePlanner()
    path = planner.search(save_gif=True)
    if not path:
        raise RuntimeError("State lattice planner returned no path")


if __name__ == "__main__":
    main()
