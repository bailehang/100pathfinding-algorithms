"""
RRT# 2D path planning demo.

RRT# keeps two cost labels for every sampled state: ``g`` (current committed
cost-to-come) and ``lmc`` (one-step lookahead cost). Newly sampled edges can make
nearby vertices locally inconsistent; a priority queue then propagates the cost
improvement, similar in spirit to LPA*/D* Lite. The GIF highlights this process:
green tree edges, blue inconsistent vertices, and the current best path to goal.
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


class Node:
    _next_id = 0

    def __init__(self, n):
        self.x = float(n[0])
        self.y = float(n[1])
        self.parent = None
        self.g = math.inf
        self.lmc = math.inf
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
        self.obs_boundary = self.env.obs_boundary
        self.obs_rectangle = self.env.obs_rectangle

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
            if self.get_dist(start, shot) <= self.get_dist(start, end):
                return True

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


class RrtSharp:
    def __init__(self, x_start, x_goal, step_len, goal_sample_rate, search_radius, iter_max):
        Node._next_id = 0
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.x_start.g = 0.0
        self.x_start.lmc = 0.0

        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.eps = 1e-7

        self.env = Env()
        self.utils = Utils()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.queue = []
        self.recent_updates = []
        self.path = []
        self.best_cost = math.inf

    def planning(self, save_gif=False, gif_name="050_rrt_sharp"):
        snapshots = [self.snapshot(0)]

        for k in range(self.iter_max):
            if k % 500 == 0:
                print(k)

            x_rand = self.sample_free()
            x_nearest = self.nearest(self.V, x_rand)
            x_new = self.steer(x_nearest, x_rand)
            if x_new is None or self.utils.is_collision(x_nearest, x_new):
                continue

            x_near = self.near(self.V, x_new)
            self.choose_parent(x_new, x_near)
            self.V.append(x_new)
            self.update_queue(x_new)
            self.reduce_inconsistency(max_pops=40)
            self.relax_neighbors(x_new, x_near)
            self.try_connect_goal(x_new)
            self.reduce_inconsistency(max_pops=60)
            self.refresh_best_path()

            if k % 90 == 0:
                snapshots.append(self.snapshot(k))

        self.refresh_best_path()
        if self.path:
            snapshots.append(self.snapshot(self.iter_max))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def sample_free(self):
        if np.random.random() < self.goal_sample_rate:
            return self.x_goal

        delta = self.utils.delta
        return Node(
            (
                np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
            )
        )

    def steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node((x_start.x + dist * math.cos(theta), x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start
        return node_new

    def near(self, nodelist, node):
        n = len(self.V) + 1
        radius = min(self.search_radius * math.sqrt(math.log(n) / n), self.step_len * 2.8)
        return [
            nd for nd in nodelist
            if self.line(nd, node) <= radius and not self.utils.is_collision(node, nd)
        ]

    def choose_parent(self, x_new, x_near):
        candidates = [(node.g + self.line(node, x_new), node) for node in x_near if math.isfinite(node.g)]
        if not candidates:
            x_new.parent = self.nearest(self.V, x_new)
            x_new.lmc = x_new.parent.g + self.line(x_new.parent, x_new)
            return

        x_new.lmc, x_new.parent = min(candidates, key=lambda item: item[0])

    def relax_neighbors(self, x_new, x_near):
        if not math.isfinite(x_new.g):
            return

        for node in x_near:
            new_lmc = x_new.g + self.line(x_new, node)
            if new_lmc + self.eps < node.lmc and not self.would_create_cycle(node, x_new):
                node.lmc = new_lmc
                node.parent = x_new
                self.update_queue(node)

    def try_connect_goal(self, node):
        if self.line(node, self.x_goal) > self.step_len:
            return
        if self.utils.is_collision(node, self.x_goal) or not math.isfinite(node.g):
            return

        candidate = node.g + self.line(node, self.x_goal)
        if candidate + self.eps < self.x_goal.lmc:
            self.x_goal.lmc = candidate
            self.x_goal.parent = node
            self.update_queue(self.x_goal)

    def reduce_inconsistency(self, max_pops=60):
        pops = 0
        while self.queue and pops < max_pops:
            key1, key2, _, node = heapq.heappop(self.queue)
            if self.is_consistent(node):
                continue
            if (key1, key2) != self.key(node):
                continue

            if node.g > node.lmc:
                node.g = node.lmc
                for successor in self.near(self.V, node):
                    new_lmc = node.g + self.line(node, successor)
                    if new_lmc + self.eps < successor.lmc and not self.would_create_cycle(successor, node):
                        successor.lmc = new_lmc
                        successor.parent = node
                        self.update_queue(successor)
                self.try_connect_goal(node)
            else:
                node.g = math.inf
                self.update_queue(node)

            pops += 1

    def update_queue(self, node):
        if not self.is_consistent(node):
            key1, key2 = self.key(node)
            heapq.heappush(self.queue, (key1, key2, node.id, node))
            self.recent_updates.append((node.x, node.y))
            self.recent_updates = self.recent_updates[-90:]

    def key(self, node):
        best = min(node.g, node.lmc)
        return best + self.line(node, self.x_goal), best

    def is_consistent(self, node):
        return abs(node.g - node.lmc) <= self.eps

    def refresh_best_path(self):
        if self.x_goal.parent is None or not math.isfinite(self.x_goal.lmc):
            return
        self.path = self.extract_path(self.x_goal.parent) + [(self.x_goal.x, self.x_goal.y)]
        self.best_cost = self.x_goal.lmc

    def extract_path(self, node):
        path = []
        seen = set()
        current = node
        while current is not None and current.id not in seen:
            seen.add(current.id)
            path.append((current.x, current.y))
            current = current.parent
        return list(reversed(path))

    def would_create_cycle(self, child, candidate_parent):
        current = candidate_parent
        while current is not None:
            if current is child:
                return True
            current = current.parent
        return False

    def snapshot(self, iteration):
        edges = [
            ((node.parent.x, node.parent.y), (node.x, node.y))
            for node in self.V
            if node.parent is not None
        ]
        inconsistent = [
            (node.x, node.y)
            for node in self.V
            if not self.is_consistent(node)
        ]
        update_points = inconsistent or list(self.recent_updates)
        return {
            "iteration": iteration,
            "nodes": len(self.V),
            "edges": edges,
            "path": list(self.path),
            "cost": self.best_cost if self.path else None,
            "queue": len(self.queue),
            "updates": update_points[:90],
        }

    def save_process_gif(self, snapshots, gif_name):
        if not snapshots:
            snapshots = [self.snapshot(0)]

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
    def select_snapshots(snapshots, max_frames=42):
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

        if snapshot["edges"]:
            tree = LineCollection(snapshot["edges"], colors="#5aa469", linewidths=0.62, alpha=0.58)
            ax.add_collection(tree)

        if snapshot["updates"]:
            ax.scatter(
                [p[0] for p in snapshot["updates"]],
                [p[1] for p in snapshot["updates"]],
                s=20,
                color="#2563eb",
                alpha=0.72,
                label="recent g/lmc updates",
                zorder=4,
            )

        if snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color="#d62728",
                linewidth=3.0,
                label="best path",
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=5)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=5)

        cost_text = "searching" if snapshot["cost"] is None else f"best cost {snapshot['cost']:.1f}"
        ax.text(
            1.6,
            28.4,
            (
                f"RRT#  iter {snapshot['iteration']:4d}  nodes {snapshot['nodes']:4d}  "
                f"queue {snapshot['queue']:3d}  {cost_text}"
            ),
            fontsize=9.3,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.86, "pad": 3},
        )

        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("050 RRT# - g/lmc consistency propagation")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    @staticmethod
    def nearest(nodelist, n):
        return nodelist[int(np.argmin([(node.x - n.x) ** 2 + (node.y - n.y) ** 2 for node in nodelist]))]

    @staticmethod
    def line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    random.seed(50)
    np.random.seed(50)
    x_start = (18, 8)
    x_goal = (37, 18)

    rrt_sharp = RrtSharp(x_start, x_goal, 4.0, 0.16, 22, 2300)
    path = rrt_sharp.planning(save_gif=True)
    if not path:
        raise RuntimeError("RRT# did not reach the goal")


if __name__ == "__main__":
    main()
