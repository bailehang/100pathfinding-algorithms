import heapq
import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


class PrecomputedCellGraph:
    def __init__(self, mode, label):
        self.mode = mode
        self.label = label
        self.cells = {}
        self.edges = {}
        self.start = None
        self.goal = None
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path = []
        self.dynamic_blocks = []
        self.build()

    def build(self):
        if self.mode == "quad":
            self.build_quad_cells()
            self.start, self.goal = "q_0_1", "q_11_6"
        elif self.mode == "hex":
            self.build_hex_cells()
            self.start, self.goal = "h_0_2", "h_10_5"
        elif self.mode == "poly":
            self.build_polygon_cells()
            self.start, self.goal = "p_0_1", "p_9_4"
        elif self.mode == "multi":
            self.build_multilayer_cells()
            self.start, self.goal = "l0_0_1", "l1_8_5"
        elif self.mode == "dynamic":
            self.build_quad_cells()
            self.start, self.goal = "q_0_1", "q_11_6"
        self.build_edges()

    def build_quad_cells(self):
        blocked = {(3, 2), (3, 3), (4, 3), (7, 4), (8, 4), (8, 5)}
        for col in range(12):
            for row in range(8):
                center = (3.0 + col * 3.8, 3.5 + row * 3.0)
                poly = self.box(center, 3.05, 2.32)
                self.add_cell(f"q_{col}_{row}", center, poly, "quad", (col, row) in blocked, 0)

    def build_hex_cells(self):
        blocked = {(3, 2), (4, 2), (6, 4), (7, 4), (8, 3)}
        radius = 1.52
        for col in range(11):
            for row in range(7):
                center = (4.0 + col * 3.35, 4.0 + row * 3.1 + (1.55 if col % 2 else 0.0))
                poly = [
                    (
                        center[0] + radius * math.cos(math.pi / 6 + i * math.pi / 3),
                        center[1] + radius * math.sin(math.pi / 6 + i * math.pi / 3),
                    )
                    for i in range(6)
                ]
                self.add_cell(f"h_{col}_{row}", center, poly, "hex", (col, row) in blocked, 0)

    def build_polygon_cells(self):
        blocked = {(2, 1), (3, 2), (5, 3), (6, 2), (7, 4)}
        for col in range(10):
            for row in range(6):
                center = (4.0 + col * 4.2, 4.3 + row * 4.0 + (0.65 if col % 2 else 0.0))
                poly = self.irregular_polygon(center, col, row)
                self.add_cell(f"p_{col}_{row}", center, poly, "polygon", (col, row) in blocked, 0)

    def build_multilayer_cells(self):
        blocked0 = {(3, 2), (3, 3), (4, 3), (6, 1)}
        blocked1 = {(2, 3), (5, 3), (6, 3)}
        for layer in (0, 1):
            for col in range(9):
                for row in range(6):
                    center = (5.0 + col * 4.2, 5.0 + row * 3.8)
                    if layer == 1:
                        center = (center[0] + 0.55, center[1] + 0.45)
                    blocked = (col, row) in (blocked0 if layer == 0 else blocked1)
                    kind = "quad" if layer == 0 else "hex"
                    poly = self.box(center, 3.2, 2.7) if layer == 0 else self.hexagon(center, 1.65)
                    self.add_cell(f"l{layer}_{col}_{row}", center, poly, kind, blocked, layer)

    def add_cell(self, cell_id, center, polygon, kind, blocked=False, layer=0):
        self.cells[cell_id] = {"center": center, "polygon": polygon, "kind": kind, "blocked": blocked, "layer": layer}

    def build_edges(self):
        ids = list(self.cells)
        for cell_id in ids:
            self.edges[cell_id] = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ca, cb = self.cells[a], self.cells[b]
                if ca["blocked"] or cb["blocked"]:
                    continue
                dist = self.distance(a, b)
                if dist <= self.edge_threshold(ca, cb):
                    cost = dist * (1.0 + 0.24 * abs(ca["layer"] - cb["layer"]))
                    self.edges[a].append((b, cost))
                    self.edges[b].append((a, cost))

    def edge_threshold(self, a, b):
        if self.mode == "multi" and a["layer"] != b["layer"]:
            return 1.6 if self.is_elevator_pair(a, b) else 0.0
        if a["kind"] == b["kind"] == "hex":
            return 3.8
        if a["kind"] == b["kind"] == "polygon":
            return 5.15
        return 4.7

    @staticmethod
    def is_elevator_pair(a, b):
        ax, ay = a["center"]
        bx, by = b["center"]
        return math.hypot(ax - bx, ay - by) < 1.6 and (18 < ax < 32 or 34 < ax < 40)

    def search(self, save_gif=False, gif_name="cell_graph"):
        self.open_heap = [(self.heuristic(self.start), 0.0, self.start)]
        self.open_set = {self.start}
        self.closed = []
        self.closed_set = set()
        self.parent = {self.start: self.start}
        self.g = {self.start: 0.0}
        snapshots = [self.snapshot(0, "precomputed cells loaded")]

        for step in range(1, 700):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            self.maybe_dynamic_update(step, snapshots)
            if current == self.goal:
                self.path = self.extract_path(current)
                snapshots.append(self.snapshot(step, "goal cell reached", final=True))
                break
            for neighbor, cost in self.edges[current]:
                if neighbor in self.dynamic_blocks:
                    continue
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 30 or step % 8 == 0:
                snapshots.append(self.snapshot(step, self.phase_text(current)))

        if not self.path:
            raise RuntimeError(f"{self.label} did not reach the goal")
        snapshots.append(self.snapshot(len(self.closed), "final cell route", final=True))
        if save_gif:
            self.save_gif(snapshots, gif_name)
        return [self.cells[c]["center"] for c in self.path]

    def maybe_dynamic_update(self, step, snapshots):
        if self.mode != "dynamic" or self.dynamic_blocks or step < 34:
            return
        self.dynamic_blocks = ["q_5_4", "q_6_4", "q_7_4"]
        for cell_id in self.dynamic_blocks:
            self.cells[cell_id]["blocked"] = True
        self.open_heap = [(f, g, c) for f, g, c in self.open_heap if c not in self.dynamic_blocks]
        heapq.heapify(self.open_heap)
        snapshots.append(self.snapshot(step, "precomputed cells changed: repair frontier"))

    def phase_text(self, current):
        if self.mode == "quad":
            return "search over pre-generated quadrilateral cells"
        if self.mode == "hex":
            return "search over six-neighbor hexagonal cells"
        if self.mode == "poly":
            return "search over irregular polygon adjacency"
        if self.mode == "multi":
            return "search crosses layer-transition portals"
        return "dynamic cell blocks trigger local repair"

    def heuristic(self, cell_id):
        ax, ay = self.cells[cell_id]["center"]
        bx, by = self.cells[self.goal]["center"]
        return math.hypot(ax - bx, ay - by)

    def extract_path(self, cell_id):
        path = []
        current = cell_id
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
        return {"step": step, "phase": phase, "final": final, "open": list(self.open_set), "closed": list(self.closed), "hint": hint, "path": list(self.path)}

    def save_gif(self, snapshots, gif_name, max_frames=46):
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
        for cell_id, cell in self.cells.items():
            ax.add_patch(patches.Polygon(cell["polygon"], closed=True, edgecolor="#111827" if cell["blocked"] else "#64748b", facecolor=self.cell_color(cell_id, snapshot), linewidth=1.0, alpha=0.97, zorder=1))
        self.draw_edges(ax, snapshot)
        self.draw_path(ax, snapshot["hint"], "#f97316", 2.0, 0.82)
        self.draw_path(ax, snapshot["path"], "#d62728", 3.0, 0.95)
        sx, sy = self.cells[self.start]["center"]
        gx, gy = self.cells[self.goal]["center"]
        ax.scatter(sx, sy, marker="s", s=76, color="#2b6cb0", zorder=6)
        ax.scatter(gx, gy, marker="s", s=76, color="#2f855a", zorder=6)
        ax.text(1.4, 28.4, f"{self.label}  step {snapshot['step']:3d}  open {len(snapshot['open']):3d}  closed {len(snapshot['closed']):3d}\n{snapshot['phase']}", fontsize=8.5, color="#1f2933", bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.88, "pad": 3})
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 31)
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

    def cell_color(self, cell_id, snapshot):
        cell = self.cells[cell_id]
        if cell["blocked"]:
            return "#1f2937"
        if cell_id in snapshot["path"]:
            return "#fecaca"
        if cell_id in snapshot["hint"]:
            return "#fed7aa"
        if cell_id in snapshot["closed"]:
            return "#cbd5e1"
        if cell_id in snapshot["open"]:
            return "#bbf7d0"
        base = {"quad": "#dbeafe", "hex": "#dcfce7", "polygon": "#fef3c7"}[cell["kind"]]
        if self.mode == "multi" and cell["layer"] == 1:
            return "#ede9fe"
        return base

    def draw_edges(self, ax, snapshot):
        visible = set(snapshot["closed"]) | set(snapshot["open"]) | set(snapshot["hint"]) | set(snapshot["path"])
        for cell_id in visible:
            for neighbor, _ in self.edges.get(cell_id, []):
                if neighbor not in visible or cell_id > neighbor:
                    continue
                ax.plot([self.cells[cell_id]["center"][0], self.cells[neighbor]["center"][0]], [self.cells[cell_id]["center"][1], self.cells[neighbor]["center"][1]], color="#94a3b8", linewidth=0.7, alpha=0.35, zorder=2)

    def draw_path(self, ax, path, color, linewidth, alpha):
        if len(path) < 2:
            return
        pts = [self.cells[c]["center"] for c in path]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=linewidth, alpha=alpha, zorder=5)

    def title(self):
        return self.label

    def distance(self, a, b):
        ax, ay = self.cells[a]["center"]
        bx, by = self.cells[b]["center"]
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def box(center, w, h):
        cx, cy = center
        return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]

    @staticmethod
    def hexagon(center, radius):
        return [(center[0] + radius * math.cos(math.pi / 6 + i * math.pi / 3), center[1] + radius * math.sin(math.pi / 6 + i * math.pi / 3)) for i in range(6)]

    @staticmethod
    def irregular_polygon(center, col, row):
        cx, cy = center
        variants = [
            [(-1.45, -1.25), (1.25, -1.35), (1.45, 0.45), (0.20, 1.45), (-1.35, 0.90)],
            [(-1.45, -1.10), (1.40, -0.95), (1.05, 1.25), (-1.20, 1.40)],
            [(-1.35, -1.25), (1.45, -0.80), (0.75, 1.45), (-1.20, 1.10), (-1.65, 0.05)],
        ]
        return [(cx + x, cy + y) for x, y in variants[(col + row) % len(variants)]]
