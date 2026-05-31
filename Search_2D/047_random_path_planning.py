"""Random Path Planning (RPP) 2D demo."""

from metrics import install_metrics
install_metrics()

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


class Node:
    def __init__(self, point, parent=None, cost=0.0):
        self.x = float(point[0])
        self.y = float(point[1])
        self.parent = parent
        self.cost = cost

    @property
    def point(self):
        return (self.x, self.y)


class RandomPathPlanner:
    def __init__(self, start=(2.0, 2.0), goal=(48.0, 24.0), seed=47):
        self.start = Node(start)
        self.goal = Node(goal)
        self.rng = np.random.default_rng(seed)
        self.x_range = (0.0, 50.0)
        self.y_range = (0.0, 30.0)
        self.obs_boundary = [
            (0, 0, 1, 30),
            (0, 29, 50, 1),
            (0, 0, 50, 1),
            (49, 0, 1, 30),
        ]
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
        self.delta = 0.45
        self.step_len = 2.35
        self.nodes = [self.start, self.goal]
        self.graph = {0: [], 1: []}
        self.edges = []
        self.frontier = [self.start]
        self.closed = set()
        self.rejected = []
        self.samples = []
        self.snapshots = []
        self.path = []

    def planning(self, save_gif=False, gif_name="047_random_path_planning"):
        self.snapshots = [self.snapshot(0, "random free-space roadmap starts with start and goal")]

        for step in range(1, 520):
            node = self.sample_free_node(step)
            node_id = self.add_roadmap_node(node)
            new_edges = self.connect_node(node_id)

            if step > 25 and step % 4 == 0:
                path = self.search_roadmap()
                if path:
                    self.path = path
                    self.snapshots.append(self.snapshot(step, "A* found a route through random visible edges", final=True))
                    break

            if step < 70 or step % 12 == 0 or new_edges >= 5:
                self.snapshots.append(self.snapshot(step, self.phase_text(step)))

        if not self.path:
            raise RuntimeError("Random Path Planning did not find a path")

        self.snapshots.append(self.snapshot(len(self.nodes), "final random path", final=True))
        if save_gif:
            self.save_gif(gif_name)
        return self.path

    def sample_free_node(self, step):
        self.samples = []
        self.rejected = []
        for _ in range(90):
            if step % 11 == 0 and self.rng.random() < 0.35:
                base = self.goal.point if self.rng.random() < 0.55 else self.start.point
                point = (
                    base[0] + self.rng.normal(0.0, 9.5),
                    base[1] + self.rng.normal(0.0, 5.0),
                )
            else:
                point = (
                    self.rng.uniform(self.x_range[0] + 1.2, self.x_range[1] - 1.2),
                    self.rng.uniform(self.y_range[0] + 1.2, self.y_range[1] - 1.2),
                )
            self.samples.append(point)
            if not self.in_bounds(point) or self.inside_obstacle(point):
                self.rejected.append(point)
                continue
            if self.too_close_to_existing(point):
                self.rejected.append(point)
                continue
            return Node(point)
        raise RuntimeError("Random Path Planning could not sample a free node")

    def add_roadmap_node(self, node):
        node_id = len(self.nodes)
        self.nodes.append(node)
        self.graph[node_id] = []
        return node_id

    def connect_node(self, node_id):
        node = self.nodes[node_id]
        candidates = sorted(
            ((self.distance(node.point, other.point), other_id) for other_id, other in enumerate(self.nodes[:-1])),
            key=lambda item: item[0],
        )
        connected = 0
        for distance, other_id in candidates[:12]:
            if distance > 13.5 and connected >= 2:
                continue
            other = self.nodes[other_id]
            if self.is_collision(node, other):
                self.rejected.append(other.point)
                continue
            self.graph[node_id].append((other_id, distance))
            self.graph[other_id].append((node_id, distance))
            self.edges.append((node_id, other_id))
            connected += 1
            if connected >= 8:
                break
        return connected

    def search_roadmap(self):
        start_id = 0
        goal_id = 1
        open_heap = [(self.distance(self.start.point, self.goal.point), 0.0, start_id)]
        parent = {start_id: start_id}
        g_score = {start_id: 0.0}
        closed = set()
        while open_heap:
            _, cost, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            if current == goal_id:
                return self.reconstruct_roadmap_path(parent, goal_id)
            for neighbor, edge_cost in self.graph[current]:
                new_cost = cost + edge_cost
                if new_cost + 1e-9 >= g_score.get(neighbor, math.inf):
                    continue
                g_score[neighbor] = new_cost
                parent[neighbor] = current
                priority = new_cost + self.distance(self.nodes[neighbor].point, self.goal.point)
                heapq.heappush(open_heap, (priority, new_cost, neighbor))
        return []

    def reconstruct_roadmap_path(self, parent, goal_id):
        ids = []
        current = goal_id
        while current in parent:
            ids.append(current)
            if parent[current] == current:
                break
            current = parent[current]
        return [self.nodes[node_id].point for node_id in reversed(ids)]

    def pick_frontier_node(self):
        active = [node for node in self.frontier if self.grid_key(node.point) not in self.closed]
        if not active:
            return None
        scores = np.array([self.frontier_score(node) for node in active], dtype=float)
        weights = np.exp(-(scores - scores.min()) / 9.0)
        weights /= weights.sum()
        return active[int(self.rng.choice(len(active), p=weights))]

    def frontier_score(self, node):
        return self.distance(node.point, self.goal.point) + 0.18 * node.cost

    def sample_candidates(self, node, step):
        candidates = []
        self.samples = []
        goal_angle = math.atan2(self.goal.y - node.y, self.goal.x - node.x)
        for _ in range(10):
            if self.rng.random() < 0.58:
                angle = goal_angle + self.rng.normal(0.0, 0.95)
            else:
                angle = self.rng.uniform(-math.pi, math.pi)
            if step % 47 == 0:
                angle = self.rng.uniform(-math.pi, math.pi)
            length = self.rng.uniform(self.step_len * 0.55, self.step_len * 1.45)
            point = (node.x + length * math.cos(angle), node.y + length * math.sin(angle))
            if not self.in_bounds(point):
                continue
            candidate = Node(point, node, node.cost + length)
            candidates.append(candidate)
            self.samples.append(point)
        return candidates

    def accept_best_candidate(self, current, candidates):
        valid = []
        self.rejected = []
        for candidate in candidates:
            if self.grid_key(candidate.point) in self.closed:
                self.rejected.append(candidate.point)
                continue
            if self.too_close_to_existing(candidate.point):
                self.rejected.append(candidate.point)
                continue
            if self.is_collision(current, candidate):
                self.rejected.append(candidate.point)
                continue
            valid.append(candidate)
        if not valid:
            return None
        valid.sort(key=lambda node: self.distance(node.point, self.goal.point) + 0.08 * node.cost + self.rng.uniform(0, 1.4))
        return valid[0]

    def too_close_to_existing(self, point):
        close = 0
        for node in self.nodes[-260:]:
            if self.distance(point, node.point) < 0.85:
                close += 1
                if close >= 2:
                    return True
        return False

    def phase_text(self, step):
        if step < 20:
            return "random free samples seed a visibility roadmap"
        if step % 47 == 0:
            return "goal/start biased samples help close roadmap gaps"
        return "collision-free straight segments connect sampled milestones"

    def extract_path(self, node):
        path = []
        current = node
        while current is not None:
            path.append(current.point)
            current = current.parent
        return list(reversed(path))

    def is_collision(self, start, end):
        if self.inside_obstacle(start.point) or self.inside_obstacle(end.point):
            return True
        for rect in self.obs_boundary + self.obs_rect:
            if self.segment_intersects_rect(start.point, end.point, rect):
                return True
        for circle in self.obs_circle:
            if self.segment_intersects_circle(start.point, end.point, circle):
                return True
        return False

    def inside_obstacle(self, point):
        x, y = point
        for ox, oy, w, h in self.obs_boundary + self.obs_rect:
            if ox - self.delta <= x <= ox + w + self.delta and oy - self.delta <= y <= oy + h + self.delta:
                return True
        for ox, oy, r in self.obs_circle:
            if math.hypot(x - ox, y - oy) <= r + self.delta:
                return True
        return False

    def segment_intersects_rect(self, a, b, rect):
        ox, oy, w, h = rect
        xmin, xmax = ox - self.delta, ox + w + self.delta
        ymin, ymax = oy - self.delta, oy + h + self.delta
        if self.point_in_rect(a, xmin, xmax, ymin, ymax) or self.point_in_rect(b, xmin, xmax, ymin, ymax):
            return True
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        edges = list(zip(corners, corners[1:] + corners[:1]))
        return any(self.segments_intersect(a, b, edge_start, edge_end) for edge_start, edge_end in edges)

    @staticmethod
    def point_in_rect(point, xmin, xmax, ymin, ymax):
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def segment_intersects_circle(self, a, b, circle):
        ox, oy, radius = circle
        return self.distance_point_to_segment((ox, oy), a, b) <= radius + self.delta

    @staticmethod
    def segments_intersect(a, b, c, d):
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        eps = 1e-9
        return (
            abs(o1) < eps and on_segment(a, c, b)
            or abs(o2) < eps and on_segment(a, d, b)
            or abs(o3) < eps and on_segment(c, a, d)
            or abs(o4) < eps and on_segment(c, b, d)
        )

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

    def snapshot(self, step, phase, final=False):
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "samples": list(self.samples),
            "rejected": list(self.rejected),
            "path": list(self.path),
        }

    def save_gif(self, gif_name, max_frames=52):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(self.snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 4)
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
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=110)
        self.draw_environment(ax)
        for a_id, b_id in snapshot["edges"]:
            if a_id >= len(snapshot["nodes"]) or b_id >= len(snapshot["nodes"]):
                continue
            a = snapshot["nodes"][a_id]
            b = snapshot["nodes"][b_id]
            ax.plot([a.x, b.x], [a.y, b.y], color="#94a3b8", linewidth=0.55, alpha=0.4, zorder=2)
        for node in snapshot["nodes"]:
            if node.parent is not None:
                ax.plot([node.parent.x, node.x], [node.parent.y, node.y], color="#94a3b8", linewidth=0.65, alpha=0.55, zorder=2)
        if snapshot["samples"]:
            ax.scatter([p[0] for p in snapshot["samples"]], [p[1] for p in snapshot["samples"]], s=15, color="#f97316", alpha=0.85, zorder=4)
        if snapshot["rejected"]:
            ax.scatter([p[0] for p in snapshot["rejected"]], [p[1] for p in snapshot["rejected"]], s=20, marker="x", color="#6b7280", alpha=0.85, zorder=4)
        if snapshot["path"]:
            xs = [p[0] for p in snapshot["path"]]
            ys = [p[1] for p in snapshot["path"]]
            ax.plot(xs, ys, color="#dc2626", linewidth=2.8, alpha=0.95, zorder=6)
        ax.scatter(self.start.x, self.start.y, marker="s", s=75, color="#2563eb", zorder=7)
        ax.scatter(self.goal.x, self.goal.y, marker="s", s=75, color="#15803d", zorder=7)
        ax.text(
            1.4,
            28.0,
            f"Random Path Planning  step {snapshot['step']:3d}  nodes {len(snapshot['nodes']):3d}\n{snapshot['phase']}",
            fontsize=8.5,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("047 Random Path Planning - random visibility roadmap")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax):
        for ox, oy, w, h in self.obs_boundary:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="#111827", facecolor="#111827", linewidth=1.0, zorder=1))
        for ox, oy, w, h in self.obs_rect:
            ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor="#111827", facecolor="#374151", linewidth=1.0, zorder=1))
        for ox, oy, radius in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), radius, edgecolor="#111827", facecolor="#374151", linewidth=1.0, zorder=1))

    def in_bounds(self, point):
        return self.x_range[0] + 1.0 < point[0] < self.x_range[1] - 1.0 and self.y_range[0] + 1.0 < point[1] < self.y_range[1] - 1.0

    def grid_key(self, point):
        return (round(point[0] / 1.2), round(point[1] / 1.2))

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    planner = RandomPathPlanner()
    path = planner.planning(save_gif=True, gif_name="047_random_path_planning")
    if not path:
        raise RuntimeError("Random Path Planning returned no path")


if __name__ == "__main__":
    main()
