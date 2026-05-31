import heapq
import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


class VoronoiVariantPlanner:
    def __init__(self, mode, label):
        self.mode = mode
        self.label = label
        self.x_range = 51
        self.y_range = 31
        self.start = (5, 5)
        self.goal = (45, 25)
        self.motions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
        ]
        self.obs = self.build_obstacles()
        self.free = self.build_free_map()
        self.clearance = distance_transform_edt(self.free)
        self.nearest_label = self.build_nearest_label_map()
        self.ridge = self.build_voronoi_ridge()
        self.field = self.build_field()
        self.closed = []
        self.open_nodes = set()
        self.path = []
        self.snapshots = []

    def build_obstacles(self):
        obs = set()
        for x in range(self.x_range):
            obs.add((x, 0))
            obs.add((x, self.y_range - 1))
        for y in range(self.y_range):
            obs.add((0, y))
            obs.add((self.x_range - 1, y))
        for x in range(10, 21):
            obs.add((x, 15))
        for y in range(1, 15):
            obs.add((20, y))
        for y in range(15, 30):
            obs.add((30, y))
        for y in range(1, 16):
            obs.add((40, y))
        return obs

    def build_free_map(self):
        free = np.ones((self.x_range, self.y_range), dtype=bool)
        for x, y in self.obs:
            free[x, y] = False
        return free

    def build_nearest_label_map(self):
        obstacle_points = sorted(self.obs)
        obstacle_array = np.array(obstacle_points, dtype=float)
        labels = np.zeros((self.x_range, self.y_range), dtype=np.int32)
        for x in range(self.x_range):
            for y in range(self.y_range):
                if (x, y) in self.obs:
                    labels[x, y] = -1
                    continue
                dist2 = np.sum((obstacle_array - np.array([x, y])) ** 2, axis=1)
                labels[x, y] = int(np.argmin(dist2))
        return labels

    def build_voronoi_ridge(self):
        ridge = np.zeros((self.x_range, self.y_range), dtype=bool)
        for x in range(2, self.x_range - 2):
            for y in range(2, self.y_range - 2):
                if not self.free[x, y] or self.clearance[x, y] < 1.6:
                    continue
                labels = set()
                for dx, dy in self.motions:
                    label = self.nearest_label[x + dx, y + dy]
                    if label >= 0:
                        labels.add(int(label))
                if len(labels) >= 2 and self.is_clearance_ridge(x, y):
                    ridge[x, y] = True
        return ridge

    def is_clearance_ridge(self, x, y):
        center = self.clearance[x, y]
        checks = [((1, 0), (-1, 0)), ((0, 1), (0, -1)), ((1, 1), (-1, -1)), ((1, -1), (-1, 1))]
        return any(
            center >= self.clearance[x + a[0], y + a[1]] and center >= self.clearance[x + b[0], y + b[1]]
            for a, b in checks
        )

    def build_field(self):
        clearance = np.maximum(self.clearance, 0.01)
        ridge_dist = distance_transform_edt(~self.ridge)
        norm_clearance = clearance / max(float(np.max(clearance)), 1.0)
        norm_ridge = ridge_dist / max(float(np.max(ridge_dist)), 1.0)

        if self.mode == "field":
            field = 0.90 / (clearance + 0.45) + 0.72 * norm_ridge
        elif self.mode == "weighted":
            weight_map = self.weight_map()
            field = weight_map / (clearance + 0.35) + 0.52 * norm_ridge
        elif self.mode == "fuzzy":
            safe = self.smoothstep(norm_clearance, 0.10, 0.55)
            center = 1.0 - self.smoothstep(norm_ridge, 0.05, 0.38)
            fuzzy_membership = 0.62 * safe + 0.38 * center
            field = 1.25 - fuzzy_membership + 0.38 / (clearance + 0.9)
        else:
            bottleneck = np.exp(-((clearance - 2.2) ** 2) / 4.2)
            adaptive_ridge = 0.28 + 0.75 * bottleneck
            field = 0.56 / (clearance + 0.35) + adaptive_ridge * norm_ridge

        field = np.where(self.free, field, np.nan)
        return field

    def weight_map(self):
        weight = np.ones((self.x_range, self.y_range), dtype=float)
        for x in range(self.x_range):
            for y in range(self.y_range):
                if not self.free[x, y]:
                    continue
                weight[x, y] = 0.75 + 0.45 * math.sin((x + 2 * y) * 0.18) ** 2
                if 20 < x < 36 and 7 < y < 23:
                    weight[x, y] += 0.55
        return weight

    @staticmethod
    def smoothstep(value, edge0, edge1):
        t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def search(self, save_gif=False, gif_name="voronoi_variant"):
        open_heap = [(self.heuristic(self.start), 0.0, self.start)]
        parent = {self.start: self.start}
        g_score = {self.start: 0.0}
        self.closed = []
        self.open_nodes = {self.start}
        self.path = []
        self.snapshots = [self.snapshot(0, self.initial_phase())]

        for step in range(1, 4000):
            if not open_heap:
                break
            _, cost, current = heapq.heappop(open_heap)
            if current in self.closed:
                continue
            self.open_nodes.discard(current)
            self.closed.append(current)
            if current == self.goal:
                self.path = self.extract_path(parent, current)
                self.snapshots.append(self.snapshot(step, "goal reached on shaped Voronoi field", final=True))
                break
            for neighbor, move_cost in self.neighbors(current):
                new_cost = cost + move_cost
                if new_cost + 1e-9 >= g_score.get(neighbor, math.inf):
                    continue
                g_score[neighbor] = new_cost
                parent[neighbor] = current
                self.open_nodes.add(neighbor)
                heapq.heappush(open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 70 or step % 45 == 0:
                self.snapshots.append(self.snapshot(step, self.phase_text()))

        if not self.path:
            raise RuntimeError(f"{self.label} did not find a path")
        self.snapshots.append(self.snapshot(len(self.closed), "final path follows the selected Voronoi variant", final=True))
        if save_gif:
            self.save_gif(gif_name)
        return self.path

    def neighbors(self, node):
        for dx, dy in self.motions:
            nxt = (node[0] + dx, node[1] + dy)
            if not self.valid(nxt) or self.diagonal_collision(node, nxt):
                continue
            step = math.hypot(dx, dy)
            field_cost = float(self.field[nxt[0], nxt[1]])
            yield nxt, step * (1.0 + field_cost)

    def valid(self, node):
        x, y = node
        return 0 <= x < self.x_range and 0 <= y < self.y_range and self.free[x, y]

    def diagonal_collision(self, a, b):
        if a[0] == b[0] or a[1] == b[1]:
            return False
        return not self.free[a[0], b[1]] or not self.free[b[0], a[1]]

    def heuristic(self, node):
        return math.hypot(node[0] - self.goal[0], node[1] - self.goal[1])

    @staticmethod
    def extract_path(parent, node):
        path = []
        current = node
        while current in parent:
            path.append(current)
            if parent[current] == current:
                break
            current = parent[current]
        return list(reversed(path))

    def initial_phase(self):
        if self.mode == "field":
            return "distance transform creates obstacle and Voronoi field"
        if self.mode == "weighted":
            return "weighted obstacle influence bends Voronoi clearance"
        if self.mode == "fuzzy":
            return "fuzzy membership blends safe clearance and ridge following"
        return "adaptive weights strengthen Voronoi guidance in narrow passages"

    def phase_text(self):
        if self.mode == "field":
            return "search trades goal progress against obstacle clearance"
        if self.mode == "weighted":
            return "search follows a weighted Voronoi diagram"
        if self.mode == "fuzzy":
            return "search follows soft fuzzy-safe cells instead of hard ridges"
        return "search adapts ridge attraction based on local clearance"

    def snapshot(self, step, phase, final=False):
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "closed": list(self.closed),
            "open": list(self.open_nodes),
            "path": list(self.path),
        }

    def save_gif(self, gif_name, max_frames=54):
        frames = [self.render_snapshot(snapshot) for snapshot in self.select_snapshots(self.snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 5)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=110)
        field = np.ma.masked_invalid(self.field.T)
        ax.imshow(field, origin="lower", extent=[0, self.x_range - 1, 0, self.y_range - 1], cmap="YlGnBu_r", alpha=0.78)
        obs_x = [p[0] for p in self.obs]
        obs_y = [p[1] for p in self.obs]
        ax.scatter(obs_x, obs_y, marker="s", s=18, color="#111827", zorder=3)
        ridge_x, ridge_y = np.where(self.ridge)
        ax.scatter(ridge_x, ridge_y, s=7, color="#06b6d4", alpha=0.78, zorder=4)
        if snapshot["closed"]:
            xs = [p[0] for p in snapshot["closed"]]
            ys = [p[1] for p in snapshot["closed"]]
            ax.scatter(xs, ys, s=8, color="#64748b", alpha=0.32, zorder=5)
        if snapshot["open"]:
            xs = [p[0] for p in snapshot["open"]]
            ys = [p[1] for p in snapshot["open"]]
            ax.scatter(xs, ys, s=11, color="#22c55e", alpha=0.55, zorder=5)
        if snapshot["path"]:
            ax.plot([p[0] for p in snapshot["path"]], [p[1] for p in snapshot["path"]], color="#dc2626", linewidth=2.9, zorder=7)
        ax.scatter(self.start[0], self.start[1], marker="s", s=76, color="#2563eb", zorder=8)
        ax.scatter(self.goal[0], self.goal[1], marker="s", s=76, color="#15803d", zorder=8)
        ax.text(
            1.2,
            28.1,
            f"{self.label}  step {snapshot['step']:4d}  closed {len(snapshot['closed']):4d}\n{snapshot['phase']}",
            fontsize=8.2,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=9,
        )
        ax.set_xlim(0, self.x_range - 1)
        ax.set_ylim(0, self.y_range - 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.label)
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame
