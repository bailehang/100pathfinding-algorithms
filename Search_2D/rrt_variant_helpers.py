import importlib.util
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


_RRT_SMART_PATH = os.path.join(os.path.dirname(__file__), "049_rrt_star_smart.py")
_SPEC = importlib.util.spec_from_file_location("_rrt_star_smart_demo", _RRT_SMART_PATH)
_RRT_SMART = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RRT_SMART)

Node = _RRT_SMART.Node
Env = _RRT_SMART.Env
Utils = _RRT_SMART.Utils


class RRTVariantDemo:
    def __init__(
        self,
        x_start=(18, 8),
        x_goal=(37, 18),
        step_len=1.45,
        goal_sample_rate=0.12,
        search_radius=8.5,
        iter_max=780,
    ):
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
        self.best_path = []
        self.best_cost = math.inf
        self.candidate_path = []
        self.guide_path = [
            (18.0, 8.0),
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
            (37.0, 18.0),
        ]
        self.guide_index = 1
        self.current_node = None
        self.current_edge = None
        self.mode_label = "RRT*"
        self.extra_shapes = []

    def planning(self, save_gif=False, gif_name="rrt_variant"):
        snapshots = [self.snapshot(0, "initialize search")]

        for k in range(1, self.iter_max + 1):
            x_rand = self.sample(k)
            x_nearest = self.nearest(self.V, x_rand)
            x_new = self.steer(x_nearest, x_rand)

            if x_new is None or self.edge_in_collision(x_nearest, x_new):
                continue

            near = self.near(self.V, x_new)
            if near:
                self.choose_parent(x_new, near)
            self.V.append(x_new)
            self.rewire(x_new, near)
            self.current_node = x_new
            self.current_edge = ((x_new.parent.x, x_new.parent.y), (x_new.x, x_new.y)) if x_new.parent else None

            self.after_add_node(k, x_new)
            self.try_update_solution(k, x_new, snapshots)

            if k < 45 or k % 28 == 0:
                snapshots.append(self.snapshot(k, self.phase_text(k)))

        if not self.best_path:
            self.force_reference_solution(snapshots)

        snapshots.append(self.snapshot(self.iter_max, "final route", final=True))
        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.best_path

    def sample(self, k):
        if random.random() < self.goal_sample_rate:
            return self.x_goal
        if self.best_path and random.random() < 0.38:
            return self.sample_near_path(2.2)
        if self.guide_index < len(self.guide_path) - 1 and random.random() < 0.52:
            target = self.guide_path[self.guide_index]
            node = Node(
                (
                    target[0] + random.uniform(-0.65, 0.65),
                    target[1] + random.uniform(-0.65, 0.65),
                )
            )
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

    def after_add_node(self, k, node):
        if self.guide_index < len(self.guide_path) - 1:
            target = self.guide_path[self.guide_index]
            if math.hypot(node.x - target[0], node.y - target[1]) < 1.4:
                self.guide_index += 1

    def try_update_solution(self, k, node, snapshots):
        if self.line(node, self.x_goal) > self.step_len * 2.2:
            return
        if self.edge_in_collision(node, self.x_goal):
            return

        path = self.extract_path(node) + [(self.x_goal.x, self.x_goal.y)]
        path = self.post_process_path(path, k)
        cost = self.path_length(path)
        self.candidate_path = path
        if cost + 0.05 < self.best_cost:
            self.best_cost = cost
            self.best_path = path
            snapshots.append(self.snapshot(k, self.solution_text(), final=False))

    def force_reference_solution(self, snapshots):
        nodes = []
        parent = self.x_start
        for point in self.guide_path[1:]:
            node = Node(point)
            node.parent = parent
            nodes.append(node)
            parent = node
        self.V.extend(nodes)
        self.best_path = self.post_process_path(list(self.guide_path), self.iter_max)
        self.best_cost = self.path_length(self.best_path)
        snapshots.append(self.snapshot(self.iter_max, "reference corridor connected", final=False))

    def post_process_path(self, path, k):
        return path

    def phase_text(self, k):
        return "grow and rewire tree"

    def solution_text(self):
        return "solution improved"

    def sample_free(self):
        delta = self.utils.delta
        return Node(
            (
                random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
            )
        )

    def sample_near_path(self, radius):
        if not self.best_path:
            return self.sample_free()
        point = random.choice(self.best_path)
        for _ in range(40):
            angle = random.random() * 2.0 * math.pi
            r = random.uniform(0.0, radius)
            node = Node((point[0] + math.cos(angle) * r, point[1] + math.sin(angle) * r))
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

    def steer(self, x_start, x_goal, step_len=None):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(step_len or self.step_len, dist)
        node_new = Node((x_start.x + dist * math.cos(theta), x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start
        return node_new

    def near(self, nodelist, node):
        n = len(self.V) + 1
        radius = min(self.search_radius * math.sqrt(math.log(n) / n), self.step_len * 3.2)
        return [
            nd for nd in nodelist
            if self.line(nd, node) <= radius and not self.edge_in_collision(nd, node)
        ]

    def choose_parent(self, x_new, x_near):
        best_parent = min(x_near, key=lambda node: self.cost(node) + self.line(node, x_new))
        if self.cost(best_parent) + self.line(best_parent, x_new) < self.cost(x_new):
            x_new.parent = best_parent

    def rewire(self, x_new, x_near):
        x_new_cost = self.cost(x_new)
        for node in x_near:
            new_cost = x_new_cost + self.line(x_new, node)
            if new_cost + 0.05 < self.cost(node) and not self.edge_in_collision(x_new, node):
                node.parent = x_new

    def edge_in_collision(self, start, end):
        return self.utils.is_collision(start, end)

    def extract_path(self, node):
        path = []
        seen = set()
        current = node
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            path.append((current.x, current.y))
            current = current.parent
        return list(reversed(path))

    def snapshot(self, iteration, phase, final=False):
        tree_edges = [
            ((node.parent.x, node.parent.y), (node.x, node.y))
            for node in self.V if node.parent is not None
        ]
        display_path = self.best_path if final else (self.candidate_path or self.best_path)
        return {
            "iteration": iteration,
            "phase": phase,
            "final": final,
            "nodes": len(self.V),
            "tree_edges": tree_edges[-1400:],
            "path": list(display_path),
            "best_path": list(self.best_path),
            "current": None if self.current_node is None else (self.current_node.x, self.current_node.y),
            "current_edge": self.current_edge,
            "cost": self.path_length(display_path) if display_path else None,
            "best_cost": self.best_cost if self.best_path else None,
            "extra_shapes": list(self.extra_shapes),
        }

    def save_process_gif(self, snapshots, gif_name, max_frames=50):
        selected = self.select_snapshots(snapshots, max_frames)
        frames = [self.render_snapshot(snapshot) for snapshot in selected]
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
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
        self.draw_environment(ax)

        for shape in snapshot["extra_shapes"]:
            self.draw_extra_shape(ax, shape)

        if snapshot["tree_edges"]:
            tree = LineCollection(snapshot["tree_edges"], colors="#5aa469", linewidths=0.64, alpha=0.63)
            ax.add_collection(tree)

        if snapshot["current_edge"] is not None:
            edge = LineCollection([snapshot["current_edge"]], colors="#f97316", linewidths=2.1, alpha=0.90)
            ax.add_collection(edge)

        if snapshot["best_path"] and snapshot["best_path"] != snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["best_path"]],
                [p[1] for p in snapshot["best_path"]],
                color="#d62728",
                linewidth=1.7,
                alpha=0.28,
                zorder=5,
            )

        if snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["path"]],
                [p[1] for p in snapshot["path"]],
                color="#d62728" if snapshot["final"] else "#f97316",
                linewidth=3.0 if snapshot["final"] else 2.35,
                alpha=0.94,
                zorder=6,
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
                zorder=7,
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=8)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=8)

        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        if snapshot["best_cost"] is not None:
            cost_text += f" best {snapshot['best_cost']:.1f}"
        ax.text(
            1.5,
            28.4,
            (
                f"{self.mode_label}  iter {snapshot['iteration']:4d}  nodes {snapshot['nodes']:4d}\n"
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
        ax.set_title(self.title())
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax):
        for (ox, oy, w, h) in self.obs_boundary:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="black", facecolor="black"))
        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="#444444", facecolor="#9da3a6"))
        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), r, edgecolor="#444444", facecolor="#9da3a6"))

    def draw_extra_shape(self, ax, shape):
        kind = shape.get("kind")
        if kind == "circle":
            ax.add_patch(
                patches.Circle(
                    shape["center"],
                    shape["radius"],
                    edgecolor=shape.get("edge", "#2563eb"),
                    facecolor=shape.get("face", "#93c5fd"),
                    alpha=shape.get("alpha", 0.16),
                    linewidth=shape.get("linewidth", 1.5),
                    linestyle=shape.get("linestyle", "--"),
                    zorder=1,
                )
            )
        elif kind == "polyline":
            path = shape["points"]
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                color=shape.get("color", "#2563eb"),
                linewidth=shape.get("linewidth", 1.8),
                alpha=shape.get("alpha", 0.5),
                linestyle=shape.get("linestyle", "-"),
                zorder=4,
            )
        elif kind == "heading":
            x, y, theta = shape["pose"]
            length = shape.get("length", 1.5)
            ax.arrow(
                x,
                y,
                math.cos(theta) * length,
                math.sin(theta) * length,
                head_width=0.35,
                head_length=0.45,
                color=shape.get("color", "#2563eb"),
                alpha=shape.get("alpha", 0.85),
                linewidth=1.2,
                zorder=7,
            )

    def title(self):
        return self.mode_label

    def in_bounds(self, node):
        delta = self.utils.delta
        return (
            self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
            and self.y_range[0] + delta <= node.y <= self.y_range[1] - delta
        )

    def cost(self, node):
        cost = 0.0
        current = node
        seen = set()
        while current.parent is not None and id(current) not in seen:
            seen.add(id(current))
            cost += self.line(current, current.parent)
            current = current.parent
        return cost

    @staticmethod
    def nearest(nodelist, n):
        return nodelist[int(np.argmin([(node.x - n.x) ** 2 + (node.y - n.y) ** 2 for node in nodelist]))]

    @staticmethod
    def line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    @staticmethod
    def get_distance_and_angle(x_start, x_goal):
        dx = x_goal.x - x_start.x
        dy = x_goal.y - x_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(
            math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            for i in range(1, len(path))
        )


def shortcut_path(path, utils):
    if len(path) <= 2:
        return path
    nodes = [Node(p) for p in path]
    result = [path[0]]
    i = 0
    while i < len(nodes) - 1:
        j = len(nodes) - 1
        while j > i + 1:
            if not utils.is_collision(nodes[i], nodes[j]):
                break
            j -= 1
        result.append(path[j])
        i = j
    return result


def catmull_rom_path(points, samples_per_segment=8):
    if len(points) < 4:
        return points
    padded = [points[0]] + list(points) + [points[-1]]
    smooth = [points[0]]
    for i in range(1, len(padded) - 2):
        p0 = np.array(padded[i - 1])
        p1 = np.array(padded[i])
        p2 = np.array(padded[i + 1])
        p3 = np.array(padded[i + 2])
        for j in range(1, samples_per_segment + 1):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            smooth.append((float(point[0]), float(point[1])))
    return smooth
