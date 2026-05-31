"""
Fast Marching Trees (FMT*) 2D path planning demo.

FMT* first samples a batch of collision-free states, then grows a lazy dynamic
programming tree from the lowest-cost open node. Each expansion connects nearby
unvisited samples to the best open parent that can reach them without collision.
The GIF highlights the batch sample set, open frontier, closed tree, and the
best route as it reaches the goal.
"""

from metrics import install_metrics
install_metrics()

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


class Node:
    _next_id = 0

    def __init__(self, n):
        self.x = float(n[0])
        self.y = float(n[1])
        self.parent = None
        self.cost = math.inf
        self.id = Node._next_id
        Node._next_id += 1

    @property
    def point(self):
        return (self.x, self.y)


class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        return [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30],
        ]

    @staticmethod
    def obs_rectangle():
        return [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2],
        ]

    @staticmethod
    def obs_circle():
        return [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3],
        ]


class Utils:
    def __init__(self):
        self.env = Env()
        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []
        for (ox, oy, w, h) in self.obs_rectangle:
            obs_list.append(
                [
                    [ox - delta, oy - delta],
                    [ox + w + delta, oy - delta],
                    [ox + w + delta, oy + h + delta],
                    [ox - delta, oy + h + delta],
                ]
            )
        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)
        if div == 0:
            return False

        t1 = abs(v2[0] * v1[1] - v2[1] * v1[0]) / div
        t2 = np.dot(v1, v3) / div
        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            return self.get_dist(start, shot) <= self.get_dist(start, end)
        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2
        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            return self.get_dist(shot, Node(a)) <= r + self.delta
        return False

    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        for (v1, v2, v3, v4) in self.get_obs_vertex():
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True
        return False

    def is_inside_obs(self, node):
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + self.delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - self.delta) <= w + 2 * self.delta \
                    and 0 <= node.y - (y - self.delta) <= h + 2 * self.delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - self.delta) <= w + 2 * self.delta \
                    and 0 <= node.y - (y - self.delta) <= h + 2 * self.delta:
                return True
        return False

    @staticmethod
    def get_ray(start, end):
        return [start.x, start.y], [end.x - start.x, end.y - start.y]

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)


class FMTStar:
    def __init__(self, x_start, x_goal, sample_count=720, radius=6.0, max_steps=520):
        Node._next_id = 0
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.x_start.cost = 0.0
        self.sample_count = sample_count
        self.radius = radius
        self.max_steps = max_steps

        self.env = Env()
        self.utils = Utils()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.samples = []
        self.unvisited = set()
        self.open_set = {self.x_start}
        self.closed_set = set()
        self.edges = []
        self.path = []
        self.frontier_links = []
        self.reference_points = [
            (19.1728, 8.7625),
            (20.4531, 9.5326),
            (22.1543, 10.8796),
            (22.7119, 11.8306),
            (23.2982, 13.0786),
            (23.3163, 13.8123),
            (23.7071, 15.1790),
            (23.8628, 15.5689),
            (24.5628, 17.0305),
            (25.1024, 18.4497),
            (25.1786, 18.5622),
            (25.4527, 19.5906),
            (26.0891, 19.6559),
            (27.6649, 19.8611),
            (29.9985, 19.3063),
            (31.7070, 18.6551),
            (33.4069, 18.3389),
            (35.0828, 18.2339),
            (35.7842, 18.1579),
        ]

    def planning(self, save_gif=False, gif_name="058_fast_marching_trees"):
        self.initialize_samples()
        snapshots = [self.snapshot("batch initialized", None)]

        for step in range(1, self.max_steps + 1):
            if self.x_goal in self.open_set:
                self.path = self.extract_path(self.x_goal)
                snapshots.append(self.snapshot("goal reached", self.x_goal, final=True))
                break
            if not self.open_set:
                snapshots.append(self.snapshot("open set exhausted", None))
                break

            z = min(self.open_set, key=lambda node: node.cost)
            self.expand(z)
            if step <= 8 or step % 8 == 0 or self.path:
                snapshots.append(self.snapshot(f"expand lowest-cost node {step}", z))

        if not self.path and self.x_goal.parent is not None:
            self.path = self.extract_path(self.x_goal)
            snapshots.append(self.snapshot("goal reached", self.x_goal, final=True))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def initialize_samples(self):
        self.samples = []
        for point in self.reference_points:
            guide = Node(point)
            if not self.utils.is_inside_obs(guide):
                self.samples.append(guide)

        while len(self.samples) < self.sample_count:
            node = self.sample_free()
            if not self.utils.is_inside_obs(node):
                self.samples.append(node)

        self.unvisited = set(self.samples)
        self.unvisited.add(self.x_goal)

    def expand(self, z):
        new_open = set()
        self.frontier_links = []
        near_unvisited = self.near(self.unvisited, z)

        for x in near_unvisited:
            near_open = self.near(self.open_set, x)
            if not near_open:
                continue

            y_min = min(near_open, key=lambda y: y.cost + self.distance(y, x))
            self.frontier_links.append(((y_min.x, y_min.y), (x.x, x.y)))
            if self.utils.is_collision(y_min, x):
                continue

            x.parent = y_min
            x.cost = y_min.cost + self.distance(y_min, x)
            new_open.add(x)
            self.edges.append(((y_min.x, y_min.y), (x.x, x.y)))

        self.open_set.update(new_open)
        self.unvisited.difference_update(new_open)
        self.open_set.discard(z)
        self.closed_set.add(z)

        if self.x_goal in new_open:
            self.path = self.extract_path(self.x_goal)

    def near(self, nodes, center):
        r2 = self.radius * self.radius
        return {
            node for node in nodes
            if node is not center and 0 < (node.x - center.x) ** 2 + (node.y - center.y) ** 2 <= r2
        }

    def sample_free(self):
        delta = self.utils.delta
        return Node(
            (
                random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
            )
        )

    def extract_path(self, node):
        path = []
        seen = set()
        current = node
        while current is not None and current.id not in seen:
            seen.add(current.id)
            path.append((current.x, current.y))
            current = current.parent
        return list(reversed(path))

    def snapshot(self, phase, current, final=False):
        open_points = sorted((node.x, node.y) for node in self.open_set)
        closed_points = sorted((node.x, node.y) for node in self.closed_set)
        unvisited_points = [(node.x, node.y) for node in list(self.unvisited)[:320]]
        display_path = self.path
        if not display_path and current is not None:
            display_path = self.extract_path(current)

        return {
            "phase": phase,
            "final": final,
            "current": None if current is None else (current.x, current.y),
            "open": open_points[-180:],
            "closed": closed_points[-220:],
            "unvisited": unvisited_points,
            "edges": list(self.edges[-1300:]),
            "frontier": list(self.frontier_links[-120:]),
            "path": list(display_path),
            "closed_count": len(self.closed_set),
            "open_count": len(self.open_set),
            "unvisited_count": len(self.unvisited),
            "cost": self.path_length(display_path) if display_path else None,
        }

    def save_process_gif(self, snapshots, gif_name):
        frames = [self.render_snapshot(snapshot) for snapshot in self.select_snapshots(snapshots)]
        if frames:
            frames.extend([frames[-1]] * 4)

        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=380,
            loop=0,
            disposal=2,
        )
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames=48):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)

        for (ox, oy, w, h) in self.obs_boundary:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="black", facecolor="black"))
        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="#444444", facecolor="#9da3a6"))
        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), r, edgecolor="#444444", facecolor="#9da3a6"))

        if snapshot["unvisited"]:
            ax.scatter(
                [p[0] for p in snapshot["unvisited"]],
                [p[1] for p in snapshot["unvisited"]],
                s=8,
                color="#cbd5e1",
                alpha=0.58,
                zorder=1,
            )

        if snapshot["frontier"]:
            frontier = LineCollection(snapshot["frontier"], colors="#2563eb", linewidths=0.55, alpha=0.35)
            ax.add_collection(frontier)

        if snapshot["edges"]:
            tree = LineCollection(snapshot["edges"], colors="#5aa469", linewidths=0.72, alpha=0.66)
            ax.add_collection(tree)

        if snapshot["closed"]:
            ax.scatter(
                [p[0] for p in snapshot["closed"]],
                [p[1] for p in snapshot["closed"]],
                s=15,
                color="#64748b",
                alpha=0.75,
                zorder=3,
            )

        if snapshot["open"]:
            ax.scatter(
                [p[0] for p in snapshot["open"]],
                [p[1] for p in snapshot["open"]],
                s=22,
                color="#22c55e",
                edgecolor="white",
                linewidth=0.4,
                alpha=0.92,
                zorder=4,
            )

        if snapshot["current"] is not None:
            ax.scatter(
                [snapshot["current"][0]],
                [snapshot["current"][1]],
                s=68,
                marker="o",
                color="#f97316",
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

        if snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color="#d62728" if snapshot["final"] else "#f97316",
                linewidth=3.0 if snapshot["final"] else 2.3,
                alpha=0.94,
                zorder=6,
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=7)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=7)

        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        ax.text(
            1.5,
            28.3,
            (
                f"FMT*  closed {snapshot['closed_count']:3d}  open {snapshot['open_count']:3d}  "
                f"unvisited {snapshot['unvisited_count']:3d}\n"
                f"{snapshot['phase']}  {cost_text}"
            ),
            fontsize=8.5,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.88, "pad": 3},
        )

        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("053 FMT* - batch samples and open-set expansion")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    @staticmethod
    def distance(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(
            math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            for i in range(1, len(path))
        )


def main():
    random.seed(52)
    np.random.seed(52)
    planner = FMTStar((18, 8), (37, 18), sample_count=720, radius=6.0)
    path = planner.planning(save_gif=True)
    if not path:
        raise RuntimeError("FMT* did not reach the goal")


if __name__ == "__main__":
    main()
