"""Havok AI style Corridor Map pathfinding demo.

This version focuses on the corridor-map idea: represent free space as a graph
of corridor samples where every node stores local clearance width. Search then
filters by an agent radius and prefers routes through wider passages.
"""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import heapq
import io
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


class HavokAICorridorMap:
    def __init__(self):
        self.bounds = (0.0, 50.0, 0.0, 30.0)
        self.start = (4.5, 6.0)
        self.goal = (45.5, 24.0)
        self.agent_radius = 1.05
        self.sample_spacing = 2.0
        self.link_distance = 3.05
        self.obstacles = [
            ("rect", (14.5, 2.0, 5.0, 12.8)),
            ("rect", (30.5, 15.0, 5.0, 12.8)),
            ("circle", (24.0, 8.5, 3.2)),
            ("circle", (26.2, 21.2, 2.9)),
        ]
        self.nodes = []
        self.clearance = []
        self.edges = {}
        self.start_id = None
        self.goal_id = None
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path_ids = []
        self.path = []
        self.snapshots = []

    def planning(self, save_gif=False):
        start_ms = now_ms()
        self.build_corridor_samples()
        self.build_corridor_edges()
        self.search_width_aware_corridor()
        self.path = [self.nodes[index] for index in self.path_ids]
        elapsed = now_ms() - start_ms
        print_metrics_for(self.path, elapsed, source="corridor_width_map")
        if save_gif:
            self.save_gif("045_havok_ai_corridor_map")
        return self.path

    def build_corridor_samples(self):
        self.nodes = [self.start, self.goal]
        xmin, xmax, ymin, ymax = self.bounds
        xs = np.arange(xmin + 2.0, xmax - 1.0, self.sample_spacing)
        ys = np.arange(ymin + 2.0, ymax - 1.0, self.sample_spacing)
        for x in xs:
            for y in ys:
                point = (float(x), float(y))
                width = self.clearance_at(point)
                if width >= self.agent_radius:
                    self.nodes.append(point)

        self.clearance = [self.clearance_at(point) for point in self.nodes]
        self.start_id = 0
        self.goal_id = 1
        self.snapshots.append(
            self.snapshot(
                0,
                "sample free-space circles; each node stores local corridor width",
                show_width_field=True,
            )
        )

    def build_corridor_edges(self):
        self.edges = {index: [] for index in range(len(self.nodes))}
        for i, point in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                other = self.nodes[j]
                distance = self.distance(point, other)
                if distance > self.link_distance:
                    continue
                min_width = min(self.clearance[i], self.clearance[j])
                if min_width < self.agent_radius:
                    continue
                if self.segment_clearance(point, other) < self.agent_radius:
                    continue
                cost = distance * (1.0 + 1.4 / max(min_width, 0.2))
                self.edges[i].append((j, cost))
                self.edges[j].append((i, cost))
        self.snapshots.append(self.snapshot(0, "connect samples whose whole segment is wide enough"))

    def search_width_aware_corridor(self):
        self.open_heap = [(self.heuristic(self.start_id), 0.0, self.start_id)]
        self.open_set = {self.start_id}
        self.parent = {self.start_id: self.start_id}
        self.g = {self.start_id: 0.0}
        for step in range(1, 900):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            if current == self.goal_id:
                self.path_ids = self.extract_path(current)
                self.snapshots.append(self.snapshot(step, "goal reached through a chain of wide corridor circles", final=True))
                self.snapshots.append(
                    self.snapshot(
                        step,
                        "final corridor width map: circles show local clearance along the route",
                        final=True,
                        show_final_circles=True,
                    )
                )
                return
            for neighbor, cost in self.edges[current]:
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 35 or step % 10 == 0:
                self.snapshots.append(self.snapshot(step, "A* expands the clearance-weighted corridor graph"))
        raise RuntimeError("Corridor map search did not reach the goal")

    def clearance_at(self, point):
        x, y = point
        xmin, xmax, ymin, ymax = self.bounds
        clearance = min(x - xmin, xmax - x, y - ymin, ymax - y)
        for kind, data in self.obstacles:
            if kind == "circle":
                ox, oy, radius = data
                clearance = min(clearance, math.hypot(x - ox, y - oy) - radius)
            else:
                ox, oy, w, h = data
                cx = min(max(x, ox), ox + w)
                cy = min(max(y, oy), oy + h)
                distance = math.hypot(x - cx, y - cy)
                if ox <= x <= ox + w and oy <= y <= oy + h:
                    distance = -min(x - ox, ox + w - x, y - oy, oy + h - y)
                clearance = min(clearance, distance)
        return max(0.0, float(clearance))

    def segment_clearance(self, a, b):
        samples = max(3, int(self.distance(a, b) / 0.35))
        return min(
            self.clearance_at((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            for t in np.linspace(0.0, 1.0, samples)
        )

    def heuristic(self, node_id):
        return self.distance(self.nodes[node_id], self.goal)

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def extract_path(self, node_id):
        path = []
        current = node_id
        while True:
            path.append(current)
            parent = self.parent[current]
            if parent == current:
                break
            current = parent
        return list(reversed(path))

    def snapshot(self, step, phase, final=False, show_width_field=False, show_final_circles=False):
        hint = self.extract_path(self.closed[-1]) if self.closed else []
        return {
            "step": step,
            "phase": phase,
            "open": list(self.open_set),
            "closed": list(self.closed),
            "hint": hint,
            "path": list(self.path_ids),
            "final": final,
            "show_width_field": show_width_field,
            "show_final_circles": show_final_circles,
        }

    def save_gif(self, gif_name, max_frames=48):
        frames = [self.render_snapshot(snapshot) for snapshot in self.select_snapshots(self.snapshots, max_frames)]
        frames.extend([frames[-1]] * 6)
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
        fig, ax = plt.subplots(figsize=(7.2, 4.7), dpi=110)
        self.draw_environment(ax)
        if snapshot["show_width_field"]:
            self.draw_width_field(ax)
        self.draw_edges(ax, snapshot)
        self.draw_node_set(ax, snapshot["closed"], "#94a3b8", 18, 0.70)
        self.draw_node_set(ax, snapshot["open"], "#22c55e", 20, 0.85)
        self.draw_node_path(ax, snapshot["hint"], "#f97316", 2.0, 0.85)
        self.draw_node_path(ax, snapshot["path"], "#dc2626", 3.1, 0.96)
        if snapshot["show_final_circles"]:
            self.draw_corridor_circles(ax)
        ax.scatter([self.start[0]], [self.start[1]], marker="s", s=78, color="#2563eb", zorder=8)
        ax.scatter([self.goal[0]], [self.goal[1]], marker="s", s=78, color="#16a34a", zorder=8)
        ax.text(
            1.2,
            27.4,
            (
                f"045 Havok AI Corridor Map  step {snapshot['step']:3d}\n"
                f"agent radius {self.agent_radius:.2f}; node color/size means clearance width\n"
                f"{snapshot['phase']}"
            ),
            fontsize=8.1,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 3},
            zorder=10,
        )
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Corridor Map: pathfinding over clearance-width circles")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax):
        ax.add_patch(patches.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0, zorder=0))
        for kind, data in self.obstacles:
            if kind == "circle":
                x, y, radius = data
                ax.add_patch(patches.Circle((x, y), radius, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=4))
            else:
                x, y, w, h = data
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=4))

    def draw_width_field(self, ax):
        xs = [point[0] for point in self.nodes[2:]]
        ys = [point[1] for point in self.nodes[2:]]
        widths = [self.clearance[index] for index in range(2, len(self.nodes))]
        scatter = ax.scatter(xs, ys, c=widths, s=np.clip(np.array(widths) * 18.0, 8, 80), cmap="viridis", alpha=0.78, zorder=2)
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label("clearance width", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    def draw_edges(self, ax, snapshot):
        visible = set(snapshot["open"]) | set(snapshot["closed"]) | set(snapshot["hint"]) | set(snapshot["path"])
        if snapshot["show_width_field"] and not visible:
            visible = set(range(len(self.nodes)))
        for node_id in visible:
            for neighbor, _ in self.edges.get(node_id, []):
                if neighbor not in visible or node_id > neighbor:
                    continue
                width = min(self.clearance[node_id], self.clearance[neighbor])
                ax.plot(
                    [self.nodes[node_id][0], self.nodes[neighbor][0]],
                    [self.nodes[node_id][1], self.nodes[neighbor][1]],
                    color="#64748b",
                    linewidth=max(0.4, min(width * 0.22, 1.7)),
                    alpha=0.24,
                    zorder=1,
                )

    def draw_node_set(self, ax, node_ids, color, size, alpha):
        if not node_ids:
            return
        ax.scatter(
            [self.nodes[index][0] for index in node_ids],
            [self.nodes[index][1] for index in node_ids],
            s=size,
            color=color,
            alpha=alpha,
            zorder=5,
        )

    def draw_node_path(self, ax, node_ids, color, linewidth, alpha):
        if len(node_ids) < 2:
            return
        points = [self.nodes[index] for index in node_ids]
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth, alpha=alpha, zorder=7)

    def draw_corridor_circles(self, ax):
        for node_id in self.path_ids:
            point = self.nodes[node_id]
            radius = self.clearance[node_id]
            ax.add_patch(
                patches.Circle(
                    point,
                    radius,
                    facecolor="#38bdf8",
                    edgecolor="#0369a1",
                    linewidth=1.0,
                    alpha=0.20,
                    zorder=3,
                )
            )
            ax.text(point[0] + 0.2, point[1] + 0.2, f"{radius:.1f}", fontsize=6.5, color="#075985", zorder=9)


def main():
    path = HavokAICorridorMap().planning(save_gif=True)
    if len(path) < 2:
        raise RuntimeError("Havok AI corridor map returned no path")


if __name__ == "__main__":
    main()
