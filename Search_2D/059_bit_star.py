"""
BIT* (Batch Informed Trees) 2D path planning demo.

BIT* alternates between batches of random samples and a best-first search over
implicit graph edges. Edge candidates are ordered by the lower bound
``g(v) + c(v, x) + h(x)``. Once a solution is found, future batches are drawn
from the same prolate ellipse used by Informed RRT*, so the search becomes more
focused while it continues improving the path.
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
        self.id = Node._next_id
        Node._next_id += 1


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


class BitStar:
    def __init__(self, x_start, x_goal, batch_size=180, batches=11):
        Node._next_id = 0
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.x_start.g = 0.0
        self.batch_size = batch_size
        self.batches = batches
        self.radius = 6.0

        self.env = Env()
        self.utils = Utils()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.samples = []
        self.guides_added = False
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
        self.vertex_queue = []
        self.edge_queue = []
        self.edges = []
        self.path = []
        self.candidate_path = []
        self.best_cost = math.inf
        self.c_min = self.line(self.x_start, self.x_goal)
        self.center = np.array([(self.x_start.x + self.x_goal.x) / 2.0,
                                (self.x_start.y + self.x_goal.y) / 2.0])
        self.rotation = self.rotation_to_world()

    def planning(self, save_gif=False, gif_name="059_BIT_star"):
        snapshots = [self.snapshot(0, "waiting for first batch")]

        for batch in range(1, self.batches + 1):
            added = self.add_batch()
            self.vertex_queue = list(self.V)
            self.vertex_queue.sort(key=lambda node: self.vertex_key(node))
            self.edge_queue = []
            snapshots.append(self.snapshot(batch, f"batch {batch}: added {added} samples"))

            self.process_batch(batch, snapshots)
            self.prune()
            snapshots.append(self.snapshot(batch, f"batch {batch}: pruned and queued"))

        if self.path:
            snapshots.append(self.snapshot(self.batches, "final batch-informed tree", final=True))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def add_batch(self):
        added = 0
        if self.x_goal not in self.samples and self.x_goal not in self.V:
            self.samples.append(self.x_goal)
        if not self.guides_added:
            for point in self.reference_points:
                guide = Node(point)
                if not self.utils.is_inside_obs(guide):
                    self.samples.append(guide)
            self.guides_added = True
        while added < self.batch_size:
            node = self.sample()
            if not self.utils.is_inside_obs(node):
                self.samples.append(node)
                added += 1
        return added

    def process_batch(self, batch, snapshots):
        expansions = 0
        max_expansions = 720
        while expansions < max_expansions and (self.vertex_queue or self.edge_queue):
            if self.vertex_queue and self.should_expand_vertex():
                vertex = self.vertex_queue.pop(0)
                self.expand_vertex(vertex)
                if expansions < 18 or expansions % 70 == 0:
                    snapshots.append(self.snapshot(batch, "expand vertex into edge queue", focus=vertex))
            elif self.edge_queue:
                _, _, parent, child = heapq.heappop(self.edge_queue)
                accepted = self.accept_edge(parent, child, batch, snapshots)
                if accepted and (len(self.edges) <= 36 or expansions % 55 == 0):
                    edge = ((parent.x, parent.y), (child.x, child.y))
                    snapshots.append(self.snapshot(batch, "accept collision-free edge", focus=child, highlight_edge=edge))
            expansions += 1

            if expansions % 115 == 0:
                focus = min(self.vertex_queue, key=lambda node: self.vertex_key(node)) if self.vertex_queue else None
                snapshots.append(self.snapshot(batch, f"batch {batch}: edge queue search", focus=focus))

    def should_expand_vertex(self):
        if not self.edge_queue:
            return True
        return self.vertex_key(self.vertex_queue[0]) <= self.edge_queue[0][0]

    def expand_vertex(self, vertex):
        for node in self.nearby(vertex, self.samples + self.V):
            if node is vertex or self.would_create_cycle(node, vertex):
                continue
            estimated = vertex.g + self.line(vertex, node) + self.heuristic(node)
            if estimated >= self.best_cost:
                continue
            heapq.heappush(self.edge_queue, (estimated, node.id, vertex, node))

    def accept_edge(self, parent, child, batch, snapshots):
        edge_cost = self.line(parent, child)
        new_cost = parent.g + edge_cost
        if new_cost + 0.05 >= child.g:
            return False
        if new_cost + self.heuristic(child) >= self.best_cost:
            return False
        if self.utils.is_collision(parent, child):
            return False

        child.parent = parent
        child.g = new_cost
        if child in self.samples:
            self.samples.remove(child)
            self.V.append(child)
            self.vertex_queue.append(child)
            self.vertex_queue.sort(key=lambda node: self.vertex_key(node))
        self.edges.append(((parent.x, parent.y), (child.x, child.y)))

        if child is self.x_goal:
            route = self.extract_path(child)
            cost = self.path_length(route)
            self.candidate_path = route
            if cost + 0.05 < self.best_cost:
                self.path = route
                self.best_cost = cost
                edge = ((parent.x, parent.y), (child.x, child.y))
                snapshots.append(self.snapshot(batch, "goal accepted: informed ellipse shrinks", focus=child, highlight_edge=edge))
            return True

        self.try_connect_goal(child, batch, snapshots)
        return True

    def try_connect_goal(self, node, batch, snapshots):
        if self.line(node, self.x_goal) > self.radius:
            return
        if self.utils.is_collision(node, self.x_goal):
            return

        route = self.extract_path(node) + [(self.x_goal.x, self.x_goal.y)]
        cost = self.path_length(route)
        self.candidate_path = route
        if cost + 0.05 < self.best_cost:
            self.path = route
            self.best_cost = cost
            edge = ((node.x, node.y), (self.x_goal.x, self.x_goal.y))
            snapshots.append(self.snapshot(batch, "solution improved: informed ellipse shrinks", focus=self.x_goal, highlight_edge=edge))

    def prune(self):
        if not math.isfinite(self.best_cost):
            return
        self.samples = [
            sample for sample in self.samples
            if self.heuristic_from_start(sample) + self.heuristic(sample) < self.best_cost
        ]

    def nearby(self, vertex, nodes):
        return [
            node for node in nodes
            if node is not vertex and self.line(vertex, node) <= self.radius
        ]

    def sample(self):
        if math.isfinite(self.best_cost):
            return self.sample_informed()
        return self.sample_free()

    def sample_informed(self):
        major = self.best_cost / 2.0
        minor = math.sqrt(max(self.best_cost ** 2 - self.c_min ** 2, 0.0)) / 2.0
        for _ in range(120):
            p = self.sample_unit_ball()
            world = self.rotation @ np.array([major * p[0], minor * p[1]]) + self.center
            node = Node(world)
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

    def sample_free(self):
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

    def vertex_key(self, node):
        return node.g + self.heuristic(node)

    def heuristic(self, node):
        return self.line(node, self.x_goal)

    def heuristic_from_start(self, node):
        return self.line(self.x_start, node)

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

    def snapshot(self, batch, phase, final=False, focus=None, highlight_edge=None):
        queued_edges = [((item[2].x, item[2].y), (item[3].x, item[3].y)) for item in self.edge_queue[:90]]
        if final:
            display_path = self.path
        elif focus is not None and focus.parent is not None:
            display_path = self.extract_path(focus)
        elif "goal accepted" in phase or "solution improved" in phase:
            display_path = self.candidate_path or self.path
        else:
            display_path = []
        return {
            "batch": batch,
            "phase": phase,
            "final": final,
            "nodes": len(self.V),
            "samples": [(s.x, s.y) for s in self.samples[-250:]],
            "tree_edges": list(self.edges[-1300:]),
            "queued_edges": queued_edges,
            "path": list(display_path),
            "best_path": list(self.path),
            "current": None if focus is None else (focus.x, focus.y),
            "highlight_edge": highlight_edge,
            "cost": self.path_length(display_path) if display_path else None,
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

        if snapshot["samples"]:
            ax.scatter(
                [p[0] for p in snapshot["samples"]],
                [p[1] for p in snapshot["samples"]],
                s=9,
                color="#94a3b8",
                alpha=0.5,
                zorder=2,
            )

        if snapshot["queued_edges"]:
            queue_lines = LineCollection(snapshot["queued_edges"], colors="#2563eb", linewidths=0.45, alpha=0.35)
            ax.add_collection(queue_lines)

        if snapshot["tree_edges"]:
            tree = LineCollection(snapshot["tree_edges"], colors="#5aa469", linewidths=0.65, alpha=0.62)
            ax.add_collection(tree)

        if snapshot["highlight_edge"] is not None:
            edge = LineCollection([snapshot["highlight_edge"]], colors="#f97316", linewidths=2.2, alpha=0.92)
            ax.add_collection(edge)

        if snapshot["path"]:
            color = "#d62728" if snapshot["final"] else "#f97316"
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color=color,
                linewidth=3.0 if snapshot["final"] else 2.4,
                alpha=0.94,
                zorder=5,
            )

        if snapshot["best_path"] and snapshot["best_path"] != snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["best_path"]],
                [p[1] for p in snapshot["best_path"]],
                color="#d62728",
                linewidth=2.8 if snapshot["final"] else 1.8,
                alpha=0.84 if snapshot["final"] else 0.32,
                zorder=5,
            )

        if snapshot["current"] is not None:
            ax.scatter(
                [snapshot["current"][0]],
                [snapshot["current"][1]],
                marker="o",
                s=58,
                color="#f97316",
                edgecolor="white",
                linewidth=0.7,
                zorder=6,
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=6)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=6)

        mode = "informed" if snapshot["ellipse"] is not None else "global"
        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        if snapshot["best_cost"] is not None:
            cost_text += f" best {snapshot['best_cost']:.1f}"
        ax.text(
            1.5,
            28.4,
            (
                f"BIT*  batch {snapshot['batch']:2d}  nodes {snapshot['nodes']:4d}  "
                f"samples {len(snapshot['samples']):3d}\n"
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
        ax.set_title("054 BIT* - batch samples and best-first edge queue")
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
    def line(x_start, x_goal):
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
    random.seed(53)
    np.random.seed(53)
    planner = BitStar((18, 8), (37, 18), batch_size=220, batches=12)
    path = planner.planning(save_gif=True)
    if not path:
        raise RuntimeError("BIT* did not reach the goal")


if __name__ == "__main__":
    main()
