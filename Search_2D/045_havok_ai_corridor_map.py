"""Havok AI style corridor-map pathfinding demo."""

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

sys.path.insert(0, os.path.dirname(__file__))
from cell_graph_helpers import PrecomputedCellGraph


class HavokAICorridorMap:
    def __init__(self):
        self.navmesh = PrecomputedCellGraph("navmesh", "Havok AI Corridor Map")
        self.start_point = (4.8, 8.6)
        self.goal_point = (43.8, 19.7)
        self.start = self.navmesh.pick_cell_containing(self.start_point)
        self.goal = self.navmesh.pick_cell_containing(self.goal_point)
        self.navmesh.start = self.start
        self.navmesh.goal = self.goal
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.corridor = []
        self.portals = []
        self.path = []
        self.snapshots = []

    def planning(self, save_gif=False):
        start_ms = now_ms()
        self.search_corridor()
        self.portals = self.extract_portals(self.corridor)
        self.path = self.string_pull(self.portals)
        elapsed = now_ms() - start_ms
        print_metrics_for(self.path, elapsed, source="corridor_map")
        if save_gif:
            self.save_gif("045_havok_ai_corridor_map")
        return self.path

    def search_corridor(self):
        self.open_heap = [(self.heuristic(self.start), 0.0, self.start)]
        self.open_set = {self.start}
        self.parent = {self.start: self.start}
        self.g = {self.start: 0.0}
        self.snapshots.append(self.snapshot(0, "build compact corridor graph from merged convex navmesh cells"))

        for step in range(1, 500):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            if current == self.goal:
                self.corridor = self.extract_cell_path(current)
                self.snapshots.append(self.snapshot(step, "A* reached goal polygon: corridor cells selected", final=True))
                return
            for neighbor, cost in self.navmesh.edges[current]:
                new_cost = self.g[current] + cost + self.corridor_width_penalty(current, neighbor)
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 24 or step % 6 == 0:
                self.snapshots.append(self.snapshot(step, "search corridor cells, preferring wider portals"))

        raise RuntimeError("Havok AI corridor map did not reach the goal")

    def corridor_width_penalty(self, a, b):
        portal = self.shared_portal(a, b)
        if not portal:
            return 4.0
        width = math.hypot(portal[1][0] - portal[0][0], portal[1][1] - portal[0][1])
        return 2.5 / max(width, 0.6)

    def heuristic(self, cell_id):
        ax, ay = self.navmesh.cells[cell_id]["center"]
        return math.hypot(ax - self.goal_point[0], ay - self.goal_point[1])

    def extract_cell_path(self, cell_id):
        path = []
        current = cell_id
        while True:
            path.append(current)
            parent = self.parent[current]
            if parent == current:
                break
            current = parent
        return list(reversed(path))

    def extract_portals(self, corridor):
        portals = [(self.start_point, self.start_point)]
        for left, right in zip(corridor, corridor[1:]):
            portal = self.shared_portal(left, right)
            if not portal:
                continue
            portals.append(self.orient_portal(portal, left, right))
        portals.append((self.goal_point, self.goal_point))
        self.snapshots.append(self.snapshot(len(self.closed), "shared-edge portals define the corridor map", portals=portals, final=True))
        return portals

    def shared_portal(self, a, b):
        a_edges = {
            self.navmesh.edge_key(edge): edge
            for edge in self.navmesh.polygon_edges(self.navmesh.cells[a]["polygon"])
        }
        for edge in self.navmesh.polygon_edges(self.navmesh.cells[b]["polygon"]):
            key = self.navmesh.edge_key(edge)
            if key in a_edges:
                return a_edges[key]
        return None

    def orient_portal(self, portal, from_cell, to_cell):
        a, b = portal
        c0 = np.array(self.navmesh.cells[from_cell]["center"], dtype=float)
        c1 = np.array(self.navmesh.cells[to_cell]["center"], dtype=float)
        direction = c1 - c0
        rel_a = np.array(a, dtype=float) - c0
        cross = direction[0] * rel_a[1] - direction[1] * rel_a[0]
        if cross >= 0.0:
            return a, b
        return b, a

    def string_pull(self, portals):
        if len(portals) <= 2:
            return [self.start_point, self.goal_point]

        path = [self.start_point]
        apex = np.array(self.start_point, dtype=float)
        left = np.array(portals[1][0], dtype=float)
        right = np.array(portals[1][1], dtype=float)
        left_index = right_index = 1
        index = 2

        while index < len(portals):
            new_left = np.array(portals[index][0], dtype=float)
            new_right = np.array(portals[index][1], dtype=float)
            if self.triarea2(apex, right, new_right) <= 0.0:
                if np.allclose(apex, right) or self.triarea2(apex, left, new_right) > 0.0:
                    right = new_right
                    right_index = index
                else:
                    path.append(tuple(left))
                    apex = left
                    index = left_index + 1
                    left = apex
                    right = apex
                    left_index = right_index = index
                    continue
            if self.triarea2(apex, left, new_left) >= 0.0:
                if np.allclose(apex, left) or self.triarea2(apex, right, new_left) < 0.0:
                    left = new_left
                    left_index = index
                else:
                    path.append(tuple(right))
                    apex = right
                    index = right_index + 1
                    left = apex
                    right = apex
                    left_index = right_index = index
                    continue
            index += 1

        path.append(self.goal_point)
        cleaned = []
        for point in path:
            if not cleaned or math.hypot(cleaned[-1][0] - point[0], cleaned[-1][1] - point[1]) > 1e-6:
                cleaned.append((float(point[0]), float(point[1])))
        self.snapshots.append(self.snapshot(len(self.closed), "funnel/string-pulling tightens the corridor path", portals=portals, pulled_path=cleaned, final=True))
        return cleaned

    @staticmethod
    def triarea2(a, b, c):
        ax, ay = b - a
        bx, by = c - a
        return bx * ay - ax * by

    def snapshot(self, step, phase, portals=None, pulled_path=None, final=False):
        hint = self.extract_cell_path(self.closed[-1]) if self.closed else []
        return {
            "step": step,
            "phase": phase,
            "open": list(self.open_set),
            "closed": list(self.closed),
            "hint": hint,
            "corridor": list(self.corridor),
            "portals": list(portals or []),
            "pulled_path": list(pulled_path or []),
            "final": final,
        }

    def save_gif(self, gif_name, max_frames=46):
        snapshots = self.select_snapshots(self.snapshots, max_frames)
        frames = [self.render_snapshot(snapshot) for snapshot in snapshots]
        frames.extend([frames[-1]] * 5)
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
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=110)
        for cell_id, cell in self.navmesh.cells.items():
            face = "#fef3c7"
            if cell_id in snapshot["closed"]:
                face = "#cbd5e1"
            if cell_id in snapshot["open"]:
                face = "#bbf7d0"
            if cell_id in snapshot["hint"]:
                face = "#fed7aa"
            if cell_id in snapshot["corridor"]:
                face = "#fecaca"
            ax.add_patch(
                patches.Polygon(
                    cell["polygon"],
                    closed=True,
                    edgecolor="#334155",
                    facecolor=face,
                    linewidth=1.0,
                    alpha=0.98,
                    zorder=1,
                )
            )
        for obstacle in self.navmesh.obstacles:
            ax.add_patch(
                patches.Circle(
                    obstacle["center"],
                    obstacle["radius"],
                    edgecolor="#111827",
                    facecolor="#1f2937",
                    linewidth=1.1,
                    zorder=3,
                )
            )
        self.draw_cell_path(ax, snapshot["hint"], "#f97316", 2.0, 0.8)
        self.draw_cell_path(ax, snapshot["corridor"], "#dc2626", 2.7, 0.95)
        self.draw_portals(ax, snapshot["portals"])
        self.draw_polyline(ax, snapshot["pulled_path"], "#7c3aed", 3.2, 0.95)
        ax.scatter([self.start_point[0]], [self.start_point[1]], marker="s", s=78, color="#2563eb", zorder=6)
        ax.scatter([self.goal_point[0]], [self.goal_point[1]], marker="s", s=78, color="#16a34a", zorder=6)
        ax.text(
            1.3,
            25.2,
            f"045 Havok AI Corridor Map  step {snapshot['step']:3d}  open {len(snapshot['open']):2d}  closed {len(snapshot['closed']):2d}\n{snapshot['phase']}",
            fontsize=8.3,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        ax.set_xlim(0.5, 48.5)
        ax.set_ylim(1.4, 26.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Havok AI style Corridor Map over NavMesh portals")
        fig.tight_layout(pad=0.35)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_cell_path(self, ax, cell_path, color, linewidth, alpha):
        if len(cell_path) < 2:
            return
        points = [self.navmesh.cells[cell_id]["center"] for cell_id in cell_path]
        self.draw_polyline(ax, points, color, linewidth, alpha)

    @staticmethod
    def draw_polyline(ax, points, color, linewidth, alpha):
        if len(points) < 2:
            return
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth, alpha=alpha, zorder=6)

    @staticmethod
    def draw_portals(ax, portals):
        for index, portal in enumerate(portals[1:-1], start=1):
            left, right = portal
            ax.plot([left[0], right[0]], [left[1], right[1]], color="#0f766e", linewidth=2.2, alpha=0.78, zorder=7)
            mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
            ax.text(mid[0], mid[1], str(index), fontsize=7, color="#0f172a", zorder=8)


def main():
    path = HavokAICorridorMap().planning(save_gif=True)
    if len(path) < 2:
        raise RuntimeError("Havok AI corridor map returned no path")


if __name__ == "__main__":
    main()
