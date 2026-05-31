import heapq
import io
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class JPSGridDemo:
    def __init__(self, label, start=(5, 5), goal=(45, 25), weight=1.0):
        self.label = label
        self.start = start
        self.goal = goal
        self.weight = weight
        self.x_range = 51
        self.y_range = 31
        self.obs = self.build_obstacles()
        self.motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path = []
        self.best_hint = []
        self.jump_points = []
        self.extra_points = []
        self.extra_lines = []
        self.extra_regions = []
        self.dynamic_events = []

    def build_obstacles(self):
        obs = set()
        for i in range(self.x_range):
            obs.add((i, 0))
            obs.add((i, self.y_range - 1))
        for i in range(self.y_range):
            obs.add((0, i))
            obs.add((self.x_range - 1, i))
        for i in range(10, 22):
            obs.add((i, 15))
        for i in range(1, 15):
            obs.add((20, i))
        for i in range(15, 30):
            obs.add((30, i))
        for i in range(1, 16):
            obs.add((40, i))
        return obs

    def search(self, save_gif=False, gif_name="jps_variant", max_steps=1500):
        self.open_heap = []
        self.open_set = {self.start}
        self.closed = []
        self.closed_set = set()
        self.parent = {self.start: self.start}
        self.g = {self.start: 0.0}
        self.path = []
        snapshots = [self.snapshot(0, "initialize grid")]
        heapq.heappush(self.open_heap, (self.key(self.start), 0.0, self.start))

        for step in range(1, max_steps + 1):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            self.on_expand(step, current, snapshots)

            if current == self.goal:
                self.path = self.extract_path(current)
                snapshots.append(self.snapshot(step, "goal reached", final=True))
                break

            for neighbor, cost in self.successors(current):
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.weight * self.heuristic(neighbor), new_cost, neighbor))

            if step < 35 or step % 18 == 0:
                self.best_hint = self.partial_path(current)
                snapshots.append(self.snapshot(step, self.phase_text(step, current)))

        if not self.path:
            raise RuntimeError(f"{self.label} did not reach the goal")
        snapshots.append(self.snapshot(len(self.closed), "final path", final=True))
        if save_gif:
            self.save_gif(snapshots, gif_name)
        return self.path

    def successors(self, current):
        result = []
        for dx, dy, cost in self.motions:
            jump = self.jump(current, dx, dy)
            if jump is None:
                continue
            dist = math.hypot(jump[0] - current[0], jump[1] - current[1])
            if dist <= 0:
                continue
            result.append((jump, dist if dx and dy else dist))
        return result

    def jump(self, current, dx, dy):
        x, y = current
        last = current
        limit = self.jump_limit(current, dx, dy)
        ray = []
        for _ in range(limit):
            x += dx
            y += dy
            node = (x, y)
            if not self.in_bounds(node) or node in self.obs:
                break
            ray.append(node)
            last = node
            if node == self.goal or self.is_forced(node, dx, dy):
                self.jump_points.append(node)
                self.extra_lines.append((current, node))
                return node
        if last != current:
            self.extra_lines.append((current, last))
            return last
        return None

    def jump_limit(self, current, dx, dy):
        return 9

    def is_forced(self, node, dx, dy):
        x, y = node
        if dx != 0 and dy != 0:
            return ((x - dx, y + dy) in self.obs and (x, y + dy) not in self.obs) or (
                (x + dx, y - dy) in self.obs and (x + dx, y) not in self.obs
            )
        if dx != 0:
            return ((x, y + 1) in self.obs and (x + dx, y + 1) not in self.obs) or (
                (x, y - 1) in self.obs and (x + dx, y - 1) not in self.obs
            )
        return ((x + 1, y) in self.obs and (x + 1, y + dy) not in self.obs) or (
            (x - 1, y) in self.obs and (x - 1, y + dy) not in self.obs
        )

    def on_expand(self, step, current, snapshots):
        return

    def phase_text(self, step, current):
        return "jump pruning expands symmetry-reduced nodes"

    def key(self, node):
        return self.weight * self.heuristic(node)

    def heuristic(self, node):
        dx = abs(node[0] - self.goal[0])
        dy = abs(node[1] - self.goal[1])
        return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

    def partial_path(self, node):
        return self.extract_path(node) if node in self.parent else []

    def extract_path(self, node):
        path = []
        current = node
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
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "closed": list(self.closed[-360:]),
            "open": list(self.open_set)[-260:],
            "jump_points": list(dict.fromkeys(self.jump_points[-180:])),
            "extra_points": list(self.extra_points[-180:]),
            "extra_lines": list(self.extra_lines[-220:]),
            "extra_regions": list(self.extra_regions),
            "dynamic_events": list(self.dynamic_events[-80:]),
            "hint": list(self.best_hint),
            "path": list(self.path),
            "cost": self.path_length(self.path) if self.path else None,
        }

    def save_gif(self, snapshots, gif_name, max_frames=48):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 4)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=360, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
        obs_x = [p[0] for p in self.obs]
        obs_y = [p[1] for p in self.obs]
        ax.scatter(obs_x, obs_y, marker="s", s=26, color="#111827", zorder=2)

        for region in snapshot["extra_regions"]:
            ax.add_patch(
                plt.Rectangle(
                    region["xy"],
                    region["w"],
                    region["h"],
                    edgecolor=region.get("edge", "#2563eb"),
                    facecolor=region.get("face", "#93c5fd"),
                    alpha=region.get("alpha", 0.12),
                    linewidth=1.4,
                    linestyle="--",
                    zorder=1,
                )
            )

        if snapshot["closed"]:
            ax.scatter([p[0] for p in snapshot["closed"]], [p[1] for p in snapshot["closed"]], s=18, color="#94a3b8", alpha=0.72, zorder=3)
        if snapshot["open"]:
            ax.scatter([p[0] for p in snapshot["open"]], [p[1] for p in snapshot["open"]], s=15, color="#22c55e", alpha=0.78, zorder=3)

        if snapshot["extra_lines"]:
            for a, b in snapshot["extra_lines"]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#2563eb", linewidth=0.7, alpha=0.28, zorder=3)

        if snapshot["jump_points"]:
            ax.scatter([p[0] for p in snapshot["jump_points"]], [p[1] for p in snapshot["jump_points"]], marker="D", s=30, color="#f59e0b", alpha=0.90, zorder=5)
        if snapshot["extra_points"]:
            ax.scatter([p[0] for p in snapshot["extra_points"]], [p[1] for p in snapshot["extra_points"]], marker="x", s=40, color="#a855f7", alpha=0.90, zorder=5)
        if snapshot["dynamic_events"]:
            ax.scatter([p[0] for p in snapshot["dynamic_events"]], [p[1] for p in snapshot["dynamic_events"]], marker="s", s=42, color="#ef4444", alpha=0.90, zorder=5)

        if snapshot["hint"] and not snapshot["final"]:
            ax.plot([p[0] for p in snapshot["hint"]], [p[1] for p in snapshot["hint"]], color="#f97316", linewidth=2.0, alpha=0.82, zorder=6)
        if snapshot["path"]:
            ax.plot([p[0] for p in snapshot["path"]], [p[1] for p in snapshot["path"]], color="#d62728", linewidth=3.0, alpha=0.94, zorder=7)

        ax.scatter(self.start[0], self.start[1], marker="s", s=82, color="#2b6cb0", zorder=8)
        ax.scatter(self.goal[0], self.goal[1], marker="s", s=82, color="#2f855a", zorder=8)
        cost = "searching" if snapshot["cost"] is None else f"path {snapshot['cost']:.1f}"
        ax.text(
            1.4,
            29.1,
            f"{self.label}  step {snapshot['step']:3d}  open {len(snapshot['open']):3d}  closed {len(snapshot['closed']):3d}\n{snapshot['phase']}  {cost}",
            fontsize=8.4,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.88, "pad": 3},
        )
        ax.set_xlim(0, self.x_range - 1)
        ax.set_ylim(0, self.y_range - 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.title())
        fig.tight_layout(pad=0.35)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def title(self):
        return self.label

    def in_bounds(self, node):
        x, y = node
        return 0 <= x < self.x_range and 0 <= y < self.y_range

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))


class BidirectionalGridDemo(JPSGridDemo):
    def search(self, save_gif=False, gif_name="bidirectional_jps", max_steps=900):
        f_open = [(self.heuristic(self.start), 0.0, self.start)]
        b_open = [(self.heuristic_reverse(self.goal), 0.0, self.goal)]
        f_parent = {self.start: self.start}
        b_parent = {self.goal: self.goal}
        f_g = {self.start: 0.0}
        b_g = {self.goal: 0.0}
        f_closed, b_closed = set(), set()
        snapshots = [self.snapshot_bi(0, "initialize bidirectional frontiers", f_closed, b_closed, None)]
        meet = None

        for step in range(1, max_steps + 1):
            if f_open:
                _, _, current = heapq.heappop(f_open)
                if current not in f_closed:
                    f_closed.add(current)
                    if current in b_closed:
                        meet = current
                    for nb, c in self.successors(current):
                        nc = f_g[current] + c
                        if nc < f_g.get(nb, math.inf):
                            f_g[nb] = nc
                            f_parent[nb] = current
                            heapq.heappush(f_open, (nc + self.heuristic(nb), nc, nb))
            if b_open and meet is None:
                _, _, current = heapq.heappop(b_open)
                if current not in b_closed:
                    b_closed.add(current)
                    if current in f_closed:
                        meet = current
                    for nb, c in self.successors(current):
                        nc = b_g[current] + c
                        if nc < b_g.get(nb, math.inf):
                            b_g[nb] = nc
                            b_parent[nb] = current
                            heapq.heappush(b_open, (nc + self.heuristic_reverse(nb), nc, nb))
            if step < 28 or step % 12 == 0 or meet is not None:
                snapshots.append(self.snapshot_bi(step, "two JPS+ frontiers search for meeting jump point", f_closed, b_closed, meet))
            if meet is not None:
                break

        if meet is None:
            raise RuntimeError(f"{self.label} did not meet")
        first = self.extract_from_parent(meet, f_parent)
        second = list(reversed(self.extract_from_parent(meet, b_parent)))[1:]
        self.path = first + second
        snapshots.append(self.snapshot_bi(step, "frontiers joined", f_closed, b_closed, meet, final=True))
        if save_gif:
            self.save_gif(snapshots, gif_name)
        return self.path

    def snapshot_bi(self, step, phase, f_closed, b_closed, meet, final=False):
        data = self.snapshot(step, phase, final=final)
        data["closed"] = list(f_closed)[-260:]
        data["open"] = list(b_closed)[-260:]
        data["extra_points"] = [] if meet is None else [meet]
        data["path"] = list(self.path)
        data["cost"] = self.path_length(self.path) if self.path else None
        return data

    def heuristic_reverse(self, node):
        dx = abs(node[0] - self.start[0])
        dy = abs(node[1] - self.start[1])
        return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

    @staticmethod
    def extract_from_parent(node, parent):
        path = []
        current = node
        seen = set()
        while current in parent and current not in seen:
            seen.add(current)
            path.append(current)
            p = parent[current]
            if p == current:
                break
            current = p
        return list(reversed(path))
