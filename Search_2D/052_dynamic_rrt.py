"""Dynamic RRT 2D path planning demo."""

from metrics import install_metrics
install_metrics()

import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


class Node:
    def __init__(self, point, parent=None):
        self.x = float(point[0])
        self.y = float(point[1])
        self.parent = parent
        self.valid = True

    @property
    def point(self):
        return (self.x, self.y)


class DynamicRRTPlanner:
    def __init__(self, start=(2.0, 2.0), goal=(48.0, 24.0), seed=51):
        self.start = Node(start)
        self.goal = Node(goal)
        self.rng = np.random.default_rng(seed)
        self.x_range = (0.0, 50.0)
        self.y_range = (0.0, 30.0)
        self.step_len = 1.65
        self.goal_sample_rate = 0.13
        self.nodes = [self.start]
        self.new_nodes = []
        self.invalid_edges = []
        self.path = []
        self.old_path = []
        self.snapshots = []
        self.dynamic_obstacle = None
        self.obs_rect = [
            (14, 12, 8, 2),
            (18, 22, 8, 3),
            (26, 7, 2, 12),
            (32, 14, 10, 2),
        ]
        self.obs_circle = [
            (7, 12, 3),
            (46, 20, 2),
            (15, 5, 2),
            (37, 7, 3),
            (37, 23, 3),
        ]

    def planning(self, save_gif=False, gif_name="052_dynamic_rrt"):
        self.snapshots = [self.snapshot(0, "initial RRT starts in the old map")]
        initial_goal = self.grow_until_goal(3200, phase="initial")
        if initial_goal is None:
            raise RuntimeError("Dynamic RRT failed to find the initial path")

        self.old_path = self.extract_path(initial_goal)
        self.path = list(self.old_path)
        self.snapshots.append(self.snapshot(len(self.nodes), "initial path found", final=True))

        self.dynamic_obstacle = self.obstacle_from_path(self.old_path)
        self.snapshots.append(self.snapshot(len(self.nodes), "new obstacle appears on the current path", show_old=True))
        self.invalidate_tree()
        self.trim_invalid_subtree()
        self.path = []
        self.snapshots.append(self.snapshot(len(self.nodes), "invalid tree branches are trimmed", show_old=True))

        repaired_goal = self.grow_until_goal(4200, phase="repair")
        if repaired_goal is None:
            raise RuntimeError("Dynamic RRT failed to repair the path")
        self.path = self.extract_path(repaired_goal)
        self.snapshots.append(self.snapshot(len(self.nodes), "repaired path avoids the new obstacle", final=True, show_old=True))

        if save_gif:
            self.save_gif(gif_name)
        return self.path

    def grow_until_goal(self, max_iter, phase):
        for step in range(1, max_iter + 1):
            rnd = self.sample_node(phase)
            near = self.nearest_node(rnd)
            new_node = self.steer(near, rnd)
            if not self.in_bounds(new_node.point) or self.is_collision(near.point, new_node.point):
                continue
            self.nodes.append(new_node)
            if phase == "repair":
                self.new_nodes.append(new_node)
            if step < 80 or step % 70 == 0:
                self.snapshots.append(self.snapshot(step, self.phase_text(phase), show_old=(phase == "repair")))
            if self.distance(new_node.point, self.goal.point) <= self.step_len * 1.8 and not self.is_collision(new_node.point, self.goal.point):
                goal_node = Node(self.goal.point, new_node)
                self.nodes.append(goal_node)
                if phase == "repair":
                    self.new_nodes.append(goal_node)
                return goal_node
        return None

    def sample_node(self, phase):
        if self.rng.random() < self.goal_sample_rate:
            return self.goal
        if phase == "repair" and self.old_path and self.rng.random() < 0.24:
            anchor = np.array(self.old_path[int(self.rng.integers(0, len(self.old_path)))])
            point = anchor + self.rng.normal(0.0, 3.8, size=2)
            return Node(point)
        return Node((
            self.rng.uniform(self.x_range[0] + 1.0, self.x_range[1] - 1.0),
            self.rng.uniform(self.y_range[0] + 1.0, self.y_range[1] - 1.0),
        ))

    def nearest_node(self, rnd):
        valid_nodes = [node for node in self.nodes if node.valid]
        return min(valid_nodes, key=lambda node: self.distance(node.point, rnd.point))

    def steer(self, start, target):
        dist = self.distance(start.point, target.point)
        if dist < 1e-9:
            return Node(start.point, start)
        step = min(self.step_len, dist)
        theta = math.atan2(target.y - start.y, target.x - start.x)
        return Node((start.x + step * math.cos(theta), start.y + step * math.sin(theta)), start)

    def obstacle_from_path(self, path):
        candidates = path[len(path) // 3: 2 * len(path) // 3]
        point = min(candidates, key=lambda p: abs(p[0] - 27.0) + abs(p[1] - 15.0))
        return (point[0] + 0.8, point[1] - 0.3, 2.35)

    def invalidate_tree(self):
        self.invalid_edges = []
        if self.dynamic_obstacle is None:
            return
        for node in self.nodes:
            if node.parent and self.segment_hits_dynamic(node.parent.point, node.point):
                node.valid = False
                self.invalid_edges.append((node.parent.point, node.point))

    def trim_invalid_subtree(self):
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node.parent and not node.parent.valid and node.valid:
                    node.valid = False
                    changed = True
        self.nodes = [node for node in self.nodes if node.valid]

    def phase_text(self, phase):
        if phase == "initial":
            return "building the original RRT tree"
        return "repair phase grows new branches from the trimmed valid tree"

    def extract_path(self, node):
        path = []
        current = node
        while current is not None:
            path.append(current.point)
            current = current.parent
        return list(reversed(path))

    def is_collision(self, a, b):
        if self.inside_obstacle(a) or self.inside_obstacle(b):
            return True
        for rect in self.obs_rect:
            if self.segment_intersects_rect(a, b, rect):
                return True
        for circle in self.all_circles():
            if self.segment_intersects_circle(a, b, circle):
                return True
        return False

    def segment_hits_dynamic(self, a, b):
        return self.dynamic_obstacle is not None and self.segment_intersects_circle(a, b, self.dynamic_obstacle)

    def all_circles(self):
        circles = list(self.obs_circle)
        if self.dynamic_obstacle is not None:
            circles.append(self.dynamic_obstacle)
        return circles

    def inside_obstacle(self, point):
        x, y = point
        if not self.in_bounds(point):
            return True
        for ox, oy, w, h in self.obs_rect:
            if ox - 0.45 <= x <= ox + w + 0.45 and oy - 0.45 <= y <= oy + h + 0.45:
                return True
        for ox, oy, r in self.all_circles():
            if math.hypot(x - ox, y - oy) <= r + 0.45:
                return True
        return False

    def segment_intersects_rect(self, a, b, rect):
        ox, oy, w, h = rect
        xmin, xmax = ox - 0.45, ox + w + 0.45
        ymin, ymax = oy - 0.45, oy + h + 0.45
        if xmin <= a[0] <= xmax and ymin <= a[1] <= ymax:
            return True
        if xmin <= b[0] <= xmax and ymin <= b[1] <= ymax:
            return True
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        edges = list(zip(corners, corners[1:] + corners[:1]))
        return any(self.segments_intersect(a, b, c, d) for c, d in edges)

    @staticmethod
    def segment_intersects_circle(a, b, circle):
        ox, oy, radius = circle
        return DynamicRRTPlanner.distance_point_to_segment((ox, oy), a, b) <= radius + 0.45

    @staticmethod
    def segments_intersect(a, b, c, d):
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return o1 * o2 < 0 and o3 * o4 < 0

    @staticmethod
    def distance_point_to_segment(point, a, b):
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        closest = (ax + t * dx, ay + t * dy)
        return math.hypot(px - closest[0], py - closest[1])

    def snapshot(self, step, phase, final=False, show_old=False):
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "nodes": list(self.nodes),
            "new_nodes": set(self.new_nodes),
            "path": list(self.path),
            "old_path": list(self.old_path) if show_old else [],
            "invalid_edges": list(self.invalid_edges),
            "dynamic_obstacle": self.dynamic_obstacle,
        }

    def save_gif(self, gif_name, max_frames=58):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(self.snapshots, max_frames)]
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
        self.draw_environment(ax, snapshot["dynamic_obstacle"])
        for node in snapshot["nodes"]:
            if node.parent is None:
                continue
            color = "#f97316" if node in snapshot["new_nodes"] else "#94a3b8"
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], color=color, linewidth=0.62, alpha=0.55, zorder=2)
        for a, b in snapshot["invalid_edges"]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#dc2626", linewidth=1.0, alpha=0.55, zorder=3)
        if snapshot["old_path"]:
            self.draw_path(ax, snapshot["old_path"], "#2563eb", 2.1, 0.42, "--")
        if snapshot["path"]:
            self.draw_path(ax, snapshot["path"], "#dc2626", 2.9, 0.95, "-")
        ax.scatter(self.start.x, self.start.y, marker="s", s=75, color="#2563eb", zorder=6)
        ax.scatter(self.goal.x, self.goal.y, marker="s", s=75, color="#15803d", zorder=6)
        ax.text(
            1.3,
            28.0,
            f"Dynamic RRT  step {snapshot['step']:4d}  nodes {len(snapshot['nodes']):4d}\n{snapshot['phase']}",
            fontsize=8.4,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("052 Dynamic RRT - trim and repair")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax, dynamic_obstacle):
        ax.add_patch(patches.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0, zorder=0))
        for ox, oy, w, h in self.obs_rect:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, facecolor="#374151", edgecolor="#111827", linewidth=1.0, zorder=1))
        for ox, oy, radius in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), radius, facecolor="#374151", edgecolor="#111827", linewidth=1.0, zorder=1))
        if dynamic_obstacle is not None:
            ox, oy, radius = dynamic_obstacle
            ax.add_patch(patches.Circle((ox, oy), radius, facecolor="#ef4444", edgecolor="#7f1d1d", linewidth=1.2, alpha=0.9, zorder=4))

    @staticmethod
    def draw_path(ax, path, color, linewidth, alpha, linestyle):
        ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=5)

    def in_bounds(self, point):
        return self.x_range[0] + 1.0 <= point[0] <= self.x_range[1] - 1.0 and self.y_range[0] + 1.0 <= point[1] <= self.y_range[1] - 1.0

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    planner = DynamicRRTPlanner()
    path = planner.planning(save_gif=True, gif_name="052_dynamic_rrt")
    if not path:
        raise RuntimeError("Dynamic RRT returned no path")


if __name__ == "__main__":
    main()
