"""
Informed RRT* 2D path planning demo.

Informed RRT* behaves like RRT* until the first feasible solution is found. Then
it restricts random samples to the prolate ellipse whose foci are the start and
goal and whose major axis is the current best path cost. The GIF makes that
switch explicit: green tree edges, orange candidate routes, red best path, and
the shrinking informed sampling ellipse.
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


class InformedRrtStar:
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

        self.V = [self.x_start]
        self.path = []
        self.candidate_path = []
        self.candidate_cost = math.inf
        self.best_cost = math.inf
        self.c_min = self.line(self.x_start, self.x_goal)
        self.center = np.array([(self.x_start.x + self.x_goal.x) / 2.0,
                                (self.x_start.y + self.x_goal.y) / 2.0])
        self.rotation = self.rotation_to_world()

    def planning(self, save_gif=False, gif_name="056_informed_rrt_star"):
        snapshots = [self.snapshot(0, "global sampling")]

        for k in range(self.iter_max):
            if k % 500 == 0:
                print(k)

            sample_mode = "informed ellipse" if math.isfinite(self.best_cost) else "global sampling"
            x_rand = self.sample()
            x_nearest = self.nearest(self.V, x_rand)
            x_new = self.steer(x_nearest, x_rand)
            if x_new is None or self.utils.is_collision(x_nearest, x_new):
                continue

            x_near = self.near(self.V, x_new)
            self.V.append(x_new)
            if x_near:
                self.choose_parent(x_new, x_near)
                self.rewire(x_new, x_near)

            if self.line(x_new, self.x_goal) <= self.step_len \
                    and not self.utils.is_collision(x_new, self.x_goal):
                candidate = self.extract_path(x_new) + [(self.x_goal.x, self.x_goal.y)]
                candidate_cost = self.path_length(candidate)
                self.candidate_path = candidate
                self.candidate_cost = candidate_cost
                if candidate_cost + 0.05 < self.best_cost:
                    self.path = candidate
                    self.best_cost = candidate_cost
                    snapshots.append(self.snapshot(k, "ellipse shrinks"))
                else:
                    snapshots.append(self.snapshot(k, "candidate route"))

            if k % 100 == 0:
                snapshots.append(self.snapshot(k, sample_mode))

        if self.path:
            snapshots.append(self.snapshot(self.iter_max, "final informed optimum", final=True))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def sample(self):
        if math.isfinite(self.best_cost):
            return self.sample_informed()
        return self.sample_free()

    def sample_informed(self):
        major = self.best_cost / 2.0
        minor_sq = max(self.best_cost ** 2 - self.c_min ** 2, 0.0)
        minor = math.sqrt(minor_sq) / 2.0

        for _ in range(120):
            point = self.sample_unit_ball()
            scaled = np.array([major * point[0], minor * point[1]])
            world = self.rotation @ scaled + self.center
            node = Node(world)
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

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

    @staticmethod
    def sample_unit_ball():
        while True:
            x, y = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
            if x * x + y * y <= 1.0:
                return np.array([x, y])

    def in_bounds(self, node):
        delta = self.utils.delta
        return (
            self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
            and self.y_range[0] + delta <= node.y <= self.y_range[1] - delta
        )

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
            if self.line(nd, node) <= radius and not self.utils.is_collision(nd, node)
        ]

    def choose_parent(self, x_new, x_near):
        costs = [self.cost(node) + self.line(node, x_new) for node in x_near]
        x_new.parent = x_near[int(np.argmin(costs))]

    def rewire(self, x_new, x_near):
        x_new_cost = self.cost(x_new)
        for node in x_near:
            new_cost = x_new_cost + self.line(x_new, node)
            if new_cost + 0.05 < self.cost(node) and not self.would_create_cycle(node, x_new):
                node.parent = x_new

    def extract_path(self, node):
        path = []
        seen = set()
        current = node
        while current is not None and id(current) not in seen:
            seen.add(id(current))
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

    def snapshot(self, iteration, phase, final=False):
        edges = [
            ((node.parent.x, node.parent.y), (node.x, node.y))
            for node in self.V
            if node.parent is not None
        ]
        display_path = self.path if final else (self.candidate_path or self.path)
        display_cost = self.best_cost if final else (
            self.candidate_cost if self.candidate_path else self.best_cost
        )
        return {
            "phase": phase,
            "final": final,
            "iteration": iteration,
            "nodes": len(self.V),
            "edges": edges,
            "path": list(display_path),
            "cost": display_cost if display_path else None,
            "best_path": list(self.path),
            "best_cost": self.best_cost if self.path else None,
            "ellipse": self.ellipse_parameters() if math.isfinite(self.best_cost) else None,
        }

    def ellipse_parameters(self):
        major = self.best_cost / 2.0
        minor = math.sqrt(max(self.best_cost ** 2 - self.c_min ** 2, 0.0)) / 2.0
        theta = math.atan2(self.x_goal.y - self.x_start.y, self.x_goal.x - self.x_start.x)
        return {
            "center": tuple(self.center),
            "width": 2.0 * major,
            "height": 2.0 * minor,
            "angle": math.degrees(theta),
        }

    def save_process_gif(self, snapshots, gif_name):
        if not snapshots:
            snapshots = [self.snapshot(0, "global sampling")]

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
    def select_snapshots(snapshots, max_frames=46):
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

        if snapshot["ellipse"] is not None:
            ellipse = snapshot["ellipse"]
            ax.add_patch(
                patches.Ellipse(
                    ellipse["center"],
                    ellipse["width"],
                    ellipse["height"],
                    angle=ellipse["angle"],
                    edgecolor="#f59e0b",
                    facecolor="#fbbf24",
                    alpha=0.12,
                    linewidth=2.0,
                    linestyle="--",
                    zorder=1,
                )
            )

        if snapshot["edges"]:
            tree = LineCollection(snapshot["edges"], colors="#5aa469", linewidths=0.62, alpha=0.56)
            ax.add_collection(tree)

        if snapshot["path"]:
            color = "#d62728" if snapshot["final"] else "#f97316"
            label = "best path" if snapshot["final"] else "candidate route"
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color=color,
                linewidth=3.0 if snapshot["final"] else 2.4,
                alpha=0.92,
                label=label,
                zorder=4,
            )

        if snapshot["best_path"] and snapshot["best_path"] != snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["best_path"]],
                [p[1] for p in snapshot["best_path"]],
                color="#d62728",
                linewidth=2.8,
                alpha=0.82,
                label="current best",
                zorder=4,
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=5)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=5)

        mode = "informed" if snapshot["ellipse"] is not None else "global"
        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        if snapshot["best_cost"] is not None:
            cost_text += f" best {snapshot['best_cost']:.1f}"
        ax.text(
            1.5,
            28.4,
            (
                f"Informed RRT*  iter {snapshot['iteration']:4d}  nodes {snapshot['nodes']:4d}\n"
                f"{mode} sampling  {snapshot['phase']}  {cost_text}"
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
        ax.set_title("051 Informed RRT* - ellipse-restricted sampling")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def rotation_to_world(self):
        direction = np.array([self.x_goal.x - self.x_start.x, self.x_goal.y - self.x_start.y])
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        return np.column_stack((direction, normal))

    @staticmethod
    def nearest(nodelist, n):
        return nodelist[int(np.argmin([(node.x - n.x) ** 2 + (node.y - n.y) ** 2 for node in nodelist]))]

    @staticmethod
    def line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                   for i in range(1, len(path)))

    def cost(self, node):
        return self.path_length(self.extract_path(node))

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    random.seed(51)
    np.random.seed(51)
    x_start = (18, 8)
    x_goal = (37, 18)

    rrt_star = InformedRrtStar(x_start, x_goal, 4.0, 0.14, 22, 2500)
    path = rrt_star.planning(save_gif=True)
    if not path:
        raise RuntimeError("Informed RRT* did not reach the goal")


if __name__ == "__main__":
    main()
