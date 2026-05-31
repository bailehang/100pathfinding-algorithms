"""
RRT*-Smart 2D path planning demo.

RRT*-Smart extends RRT* by shortcutting the first feasible route and then
sampling around beacon points near the improved path. The GIF shows the random
tree, the current optimized route, and the beacon regions that bias later
samples toward useful corridor corners.
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
    def __init__(self, n):
        self.x = float(n[0])
        self.y = float(n[1])
        self.parent = None

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


class RrtStarSmart:
    def __init__(self, x_start, x_goal, step_len, goal_sample_rate, search_radius, iter_max):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = Env()
        self.utils = Utils()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_vertices = [vertex for rect in self.utils.get_obs_vertex() for vertex in rect]

        self.V = [self.x_start]
        self.beacons = []
        self.beacons_radius = 3.0
        self.path = []
        self.best_cost = math.inf

    def planning(self, save_gif=False, gif_name="054_rrt_star_smart"):
        snapshots = []
        first_path_iter = None

        for k in range(self.iter_max):
            if k % 500 == 0:
                print(k)

            use_beacon = first_path_iter is not None and self.beacons and (k - first_path_iter) % 2 == 0
            x_rand = self.sample_beacon() if use_beacon else self.sample_free()
            x_nearest = self.nearest(self.V, x_rand)
            x_new = self.steer(x_nearest, x_rand)

            if x_new and not self.utils.is_collision(x_nearest, x_new):
                x_near = self.near(self.V, x_new)
                self.V.append(x_new)

                if x_near:
                    self.choose_parent(x_new, x_near)
                    self.rewire(x_new, x_near)

                if self.line(x_new, self.x_goal) <= self.step_len \
                        and not self.utils.is_collision(x_new, self.x_goal):
                    raw_path = self.extract_path(x_new)
                    smart_path = self.shortcut_path(raw_path)
                    smart_cost = self.path_length(smart_path)
                    if smart_cost + 0.05 < self.best_cost:
                        self.path = smart_path
                        self.best_cost = smart_cost
                        self.beacons = self.update_beacons(smart_path)
                        first_path_iter = k if first_path_iter is None else first_path_iter

                if k % 90 == 0:
                    snapshots.append(self.snapshot(k, use_beacon))

        if not self.path:
            goal_node = self.search_goal_parent()
            if goal_node is not None:
                self.path = self.shortcut_path(self.extract_path(goal_node))
                self.best_cost = self.path_length(self.path)
                self.beacons = self.update_beacons(self.path)

        if self.path:
            snapshots.append(self.snapshot(self.iter_max, bool(self.beacons)))

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

    def sample_beacon(self):
        beacon = random.choice(self.beacons)
        radius = random.uniform(0.0, self.beacons_radius)
        theta = random.uniform(0.0, 2.0 * math.pi)
        candidate = Node((beacon[0] + radius * math.cos(theta), beacon[1] + radius * math.sin(theta)))
        if self.utils.is_inside_obs(candidate):
            return self.sample_free()
        return candidate

    def steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node((x_start.x + dist * math.cos(theta), x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start
        return node_new

    def near(self, nodelist, node):
        n = len(self.V) + 1
        radius = min(self.search_radius * math.sqrt(math.log(n) / n), self.step_len * 2.5)
        return [
            nd for nd in nodelist
            if self.line(nd, node) <= radius and not self.utils.is_collision(node, nd)
        ]

    def choose_parent(self, x_new, x_near):
        costs = [self.cost(node) + self.line(node, x_new) for node in x_near]
        x_new.parent = x_near[int(np.argmin(costs))]

    def rewire(self, x_new, x_near):
        x_new_cost = self.cost(x_new)
        for node in x_near:
            if x_new_cost + self.line(x_new, node) < self.cost(node):
                node.parent = x_new

    def search_goal_parent(self):
        candidates = [
            node for node in self.V
            if self.line(node, self.x_goal) <= self.step_len
            and not self.utils.is_collision(node, self.x_goal)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda node: self.cost(node) + self.line(node, self.x_goal))

    def extract_path(self, node):
        path = [(self.x_goal.x, self.x_goal.y)]
        current = node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return list(reversed(path))

    def shortcut_path(self, path):
        if len(path) <= 2:
            return path

        optimized = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            nxt = len(path) - 1
            while nxt > anchor + 1:
                if not self.utils.is_collision(Node(path[anchor]), Node(path[nxt])):
                    break
                nxt -= 1
            optimized.append(path[nxt])
            anchor = nxt
        return optimized

    def update_beacons(self, path):
        beacons = []
        for vertex in self.obs_vertices:
            if any(self.distance_point_to_segment(vertex, path[i], path[i + 1]) <= 3.5
                   for i in range(len(path) - 1)):
                beacons.append(tuple(vertex))
        return beacons

    @staticmethod
    def distance_point_to_segment(point, start, end):
        p = np.array(point, dtype=float)
        a = np.array(start, dtype=float)
        b = np.array(end, dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom == 0:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
        return float(np.linalg.norm(p - (a + t * ab)))

    def snapshot(self, iteration, sampled_beacon):
        edges = [
            ((node.parent.x, node.parent.y), (node.x, node.y))
            for node in self.V
            if node.parent is not None
        ]
        return {
            "iteration": iteration,
            "nodes": len(self.V),
            "edges": edges,
            "path": list(self.path),
            "cost": self.best_cost if self.path else None,
            "beacons": list(self.beacons),
            "sampled_beacon": sampled_beacon,
        }

    def save_process_gif(self, snapshots, gif_name):
        if not snapshots:
            snapshots = [self.snapshot(0, False)]

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
            tree = LineCollection(snapshot["edges"], colors="#5aa469", linewidths=0.62, alpha=0.62)
            ax.add_collection(tree)

        for beacon in snapshot["beacons"]:
            ax.add_patch(
                patches.Circle(
                    beacon,
                    self.beacons_radius,
                    edgecolor="#f59e0b",
                    facecolor="none",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                )
            )
            ax.scatter(beacon[0], beacon[1], marker="x", s=42, color="#d97706", zorder=4)

        if snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color="#d62728",
                linewidth=3.0,
                label="smart path",
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=4)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=4)

        mode = "beacon sampling" if snapshot["sampled_beacon"] else "global sampling"
        cost_text = "searching" if snapshot["cost"] is None else f"smart cost {snapshot['cost']:.1f}"
        ax.text(
            1.6,
            28.4,
            f"RRT*-Smart  iter {snapshot['iteration']:4d}  nodes {snapshot['nodes']:4d}  {mode}  {cost_text}",
            fontsize=9.3,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.86, "pad": 3},
        )

        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.y_range[0], self.y_range[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("049 RRT*-Smart - shortcut path and beacon-guided sampling")
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
    def cost(node):
        total = 0.0
        current = node
        while current.parent is not None:
            total += math.hypot(current.x - current.parent.x, current.y - current.parent.y)
            current = current.parent
        return total

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                   for i in range(1, len(path)))

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    random.seed(49)
    np.random.seed(49)
    x_start = (18, 8)
    x_goal = (37, 18)

    rrt_star_smart = RrtStarSmart(x_start, x_goal, 3.5, 0.14, 20, 2600)
    path = rrt_star_smart.planning(save_gif=True)
    if not path:
        raise RuntimeError("RRT*-Smart did not reach the goal")


if __name__ == "__main__":
    main()
