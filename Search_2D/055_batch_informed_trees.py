"""
AIT* (Adaptively Informed Trees) 2D path planning demo.

AIT* keeps the batch-informed forward search idea, but it also grows a lazy
reverse search from the goal. The reverse tree provides an adaptive cost-to-go
estimate, so forward edge ordering becomes more informed than plain Euclidean
distance after each batch. The GIF shows the reverse heuristic wave in purple,
the forward tree in green, accepted edges in orange, and the final path in red.
"""

from metrics import install_metrics
install_metrics()

import heapq
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


_BIT_STAR_PATH = os.path.join(os.path.dirname(__file__), "053_bit_star.py")
_SPEC = importlib.util.spec_from_file_location("_bit_star_demo", _BIT_STAR_PATH)
_BIT_STAR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BIT_STAR)

Node = _BIT_STAR.Node
Env = _BIT_STAR.Env
Utils = _BIT_STAR.Utils
BitStar = _BIT_STAR.BitStar


class AITStar(BitStar):
    def __init__(self, x_start, x_goal, batch_size=185, batches=12):
        super().__init__(x_start, x_goal, batch_size=batch_size, batches=batches)
        self.radius = 6.0
        self.reverse_radius = 6.8
        self.reverse_cost = {self.x_goal.id: 0.0}
        self.reverse_parent = {}
        self.reverse_edges = []
        self.reverse_open = [(0.0, self.x_goal.id, self.x_goal)]
        self.reverse_known = {self.x_goal.id: self.x_goal}
        self.last_reverse_focus = None

    def planning(self, save_gif=False, gif_name="055_AIT_star"):
        snapshots = [self.snapshot(0, "waiting for first adaptive heuristic batch")]

        for batch in range(1, self.batches + 1):
            added = self.add_batch()
            snapshots.append(self.snapshot(batch, f"batch {batch}: added {added} samples"))

            self.update_reverse_heuristic(batch, snapshots)

            self.vertex_queue = list(self.V)
            self.vertex_queue.sort(key=lambda node: self.vertex_key(node))
            self.edge_queue = []
            self.process_batch(batch, snapshots)
            self.prune()
            snapshots.append(self.snapshot(batch, f"batch {batch}: pruned with adaptive heuristic"))

        if self.path:
            snapshots.append(self.snapshot(self.batches, "final adaptively informed tree", final=True))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def update_reverse_heuristic(self, batch, snapshots):
        nodes = self.samples + self.V + [self.x_goal]
        for node in nodes:
            self.reverse_known[node.id] = node

        expansions = 0
        while self.reverse_open and expansions < 170:
            cost, _, node = heapq.heappop(self.reverse_open)
            if cost > self.reverse_cost.get(node.id, math.inf) + 1e-9:
                continue
            self.last_reverse_focus = node

            for neighbor in self.nearby_reverse(node, nodes):
                step = self.line(node, neighbor)
                new_cost = cost + step
                if new_cost + 0.05 >= self.reverse_cost.get(neighbor.id, math.inf):
                    continue
                if self.utils.is_collision(node, neighbor):
                    continue
                self.reverse_cost[neighbor.id] = new_cost
                self.reverse_parent[neighbor.id] = node.id
                self.reverse_known[neighbor.id] = neighbor
                self.reverse_edges.append(((node.x, node.y), (neighbor.x, neighbor.y)))
                heapq.heappush(self.reverse_open, (new_cost, neighbor.id, neighbor))

            expansions += 1
            if expansions in (1, 8, 28, 70, 130):
                snapshots.append(self.snapshot(batch, "reverse heuristic wave expands", focus=node))

    def process_batch(self, batch, snapshots):
        expansions = 0
        max_expansions = 740
        while expansions < max_expansions and (self.vertex_queue or self.edge_queue):
            if self.vertex_queue and self.should_expand_vertex():
                vertex = self.vertex_queue.pop(0)
                self.expand_vertex(vertex)
                if expansions < 16 or expansions % 70 == 0:
                    snapshots.append(self.snapshot(batch, "forward tree uses adaptive heuristic", focus=vertex))
            elif self.edge_queue:
                _, _, parent, child = heapq.heappop(self.edge_queue)
                accepted = self.accept_edge(parent, child, batch, snapshots)
                if accepted and (len(self.edges) <= 38 or expansions % 58 == 0):
                    edge = ((parent.x, parent.y), (child.x, child.y))
                    snapshots.append(self.snapshot(batch, "accept forward edge", focus=child, highlight_edge=edge))
            expansions += 1

            if expansions % 120 == 0:
                focus = min(self.vertex_queue, key=lambda node: self.vertex_key(node)) if self.vertex_queue else None
                snapshots.append(self.snapshot(batch, f"batch {batch}: forward edge queue search", focus=focus))

    def expand_vertex(self, vertex):
        for node in self.nearby(vertex, self.samples + self.V):
            if node is vertex or self.would_create_cycle(node, vertex):
                continue
            estimated = vertex.g + self.line(vertex, node) + self.adaptive_heuristic(node)
            if estimated >= self.best_cost:
                continue
            heapq.heappush(self.edge_queue, (estimated, node.id, vertex, node))

    def accept_edge(self, parent, child, batch, snapshots):
        edge_cost = self.line(parent, child)
        new_cost = parent.g + edge_cost
        if new_cost + 0.05 >= child.g:
            return False
        if new_cost + self.adaptive_heuristic(child) >= self.best_cost:
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
            self.update_best_path(child, batch, snapshots, "goal accepted through adaptive heuristic")
            return True

        self.try_connect_goal(child, batch, snapshots)
        return True

    def try_connect_goal(self, node, batch, snapshots):
        if self.line(node, self.x_goal) > self.radius:
            return
        if self.utils.is_collision(node, self.x_goal):
            return
        self.x_goal.parent = node
        self.x_goal.g = node.g + self.line(node, self.x_goal)
        self.update_best_path(self.x_goal, batch, snapshots, "solution improved: reverse heuristic catches up")

    def update_best_path(self, node, batch, snapshots, phase):
        route = self.extract_path(node)
        cost = self.path_length(route)
        self.candidate_path = route
        if cost + 0.05 < self.best_cost:
            self.path = route
            self.best_cost = cost
            highlight = None
            if len(route) >= 2:
                highlight = (route[-2], route[-1])
            snapshots.append(self.snapshot(batch, phase, focus=node, highlight_edge=highlight))

    def prune(self):
        if not math.isfinite(self.best_cost):
            return
        self.samples = [
            sample for sample in self.samples
            if self.heuristic_from_start(sample) + self.adaptive_heuristic(sample) < self.best_cost
        ]

    def sample(self):
        if math.isfinite(self.best_cost):
            return self.sample_informed()
        if len(self.reverse_cost) > 24 and random.random() < 0.45:
            return self.sample_near_reverse_tree()
        return self.sample_free()

    def sample_near_reverse_tree(self):
        anchors = [
            node for node_id, node in self.reverse_known.items()
            if node_id in self.reverse_cost and node is not self.x_goal
        ]
        if not anchors:
            return self.sample_free()
        anchor = random.choice(anchors)
        for _ in range(40):
            angle = random.random() * 2.0 * math.pi
            radius = random.uniform(0.2, 3.8)
            node = Node((anchor.x + math.cos(angle) * radius, anchor.y + math.sin(angle) * radius))
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

    def vertex_key(self, node):
        return node.g + self.adaptive_heuristic(node)

    def adaptive_heuristic(self, node):
        reverse = self.reverse_cost.get(node.id, math.inf)
        euclidean = self.heuristic(node)
        if math.isfinite(reverse):
            return max(euclidean, 0.82 * reverse)
        return euclidean

    def nearby_reverse(self, vertex, nodes):
        return [
            node for node in nodes
            if node is not vertex and self.line(vertex, node) <= self.reverse_radius
        ]

    def snapshot(self, batch, phase, final=False, focus=None, highlight_edge=None):
        data = super().snapshot(batch, phase, final=final, focus=focus, highlight_edge=highlight_edge)
        data["reverse_edges"] = list(self.reverse_edges[-1000:])
        data["reverse_known"] = [
            (node.x, node.y) for node_id, node in self.reverse_known.items()
            if node_id in self.reverse_cost
        ][-220:]
        data["reverse_focus"] = (
            None if self.last_reverse_focus is None
            else (self.last_reverse_focus.x, self.last_reverse_focus.y)
        )
        data["adaptive_count"] = len(self.reverse_cost)
        return data

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
    def select_snapshots(snapshots, max_frames=50):
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
                s=8,
                color="#94a3b8",
                alpha=0.48,
                zorder=2,
            )

        if snapshot["reverse_edges"]:
            reverse = LineCollection(snapshot["reverse_edges"], colors="#8b5cf6", linewidths=0.58, alpha=0.44)
            ax.add_collection(reverse)

        if snapshot["reverse_known"]:
            ax.scatter(
                [p[0] for p in snapshot["reverse_known"]],
                [p[1] for p in snapshot["reverse_known"]],
                s=13,
                color="#a855f7",
                alpha=0.58,
                zorder=3,
            )

        if snapshot["queued_edges"]:
            queue_lines = LineCollection(snapshot["queued_edges"], colors="#2563eb", linewidths=0.45, alpha=0.34)
            ax.add_collection(queue_lines)

        if snapshot["tree_edges"]:
            tree = LineCollection(snapshot["tree_edges"], colors="#5aa469", linewidths=0.65, alpha=0.64)
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
                linewidth=3.0 if snapshot["final"] else 2.3,
                alpha=0.94,
                zorder=6,
            )

        if snapshot["best_path"] and snapshot["best_path"] != snapshot["path"]:
            ax.plot(
                [p[0] for p in snapshot["best_path"]],
                [p[1] for p in snapshot["best_path"]],
                color="#d62728",
                linewidth=2.8 if snapshot["final"] else 1.8,
                alpha=0.84 if snapshot["final"] else 0.30,
                zorder=6,
            )

        if snapshot["reverse_focus"] is not None:
            ax.scatter(
                [snapshot["reverse_focus"][0]],
                [snapshot["reverse_focus"][1]],
                marker="D",
                s=48,
                color="#7c3aed",
                edgecolor="white",
                linewidth=0.6,
                zorder=7,
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
                zorder=8,
            )

        ax.scatter(self.x_start.x, self.x_start.y, marker="s", s=72, color="#2b6cb0", zorder=9)
        ax.scatter(self.x_goal.x, self.x_goal.y, marker="s", s=72, color="#2f855a", zorder=9)

        mode = "adaptive informed" if snapshot["ellipse"] is not None else "global"
        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        if snapshot["best_cost"] is not None:
            cost_text += f" best {snapshot['best_cost']:.1f}"
        ax.text(
            1.5,
            28.4,
            (
                f"AIT* batch {snapshot['batch']:2d}  nodes {snapshot['nodes']:4d}  "
                f"adaptive h {snapshot['adaptive_count']:3d}\n"
                f"{mode}  {snapshot['phase']}  {cost_text}"
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
        ax.set_title("055 AIT* - reverse adaptive heuristic and forward search")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame


def main():
    random.seed(55)
    np.random.seed(55)
    planner = AITStar((18, 8), (37, 18), batch_size=185, batches=12)
    path = planner.planning(save_gif=True)
    if not path:
        raise RuntimeError("AIT* did not reach the goal")


if __name__ == "__main__":
    main()
