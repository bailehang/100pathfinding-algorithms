"""
ABIT* (Advanced Batch Informed Trees) 2D path planning demo.

ABIT* keeps the BIT* batch/edge-queue structure, then adds an adaptive search
schedule. Early batches use an inflated heuristic and a wider truncation bound
to find a feasible route quickly; later batches reduce both values so the search
becomes less greedy and more focused around the best known solution.
"""

from metrics import install_metrics
install_metrics()

import importlib.util
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


_BIT_STAR_PATH = os.path.join(os.path.dirname(__file__), "058_bit_star.py")
_SPEC = importlib.util.spec_from_file_location("_bit_star_demo", _BIT_STAR_PATH)
_BIT_STAR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BIT_STAR)

Node = _BIT_STAR.Node
Env = _BIT_STAR.Env
Utils = _BIT_STAR.Utils
BitStar = _BIT_STAR.BitStar


class ABITStar(BitStar):
    def __init__(self, x_start, x_goal, batch_size=190, batches=13):
        super().__init__(x_start, x_goal, batch_size=batch_size, batches=batches)
        self.initial_inflation = 2.4
        self.final_inflation = 1.0
        self.initial_truncation = 1.65
        self.final_truncation = 1.03
        self.current_inflation = self.initial_inflation
        self.current_truncation = self.initial_truncation

    def planning(self, save_gif=False, gif_name="059_ABIT_star"):
        snapshots = [self.snapshot(0, "waiting for first adaptive batch")]

        for batch in range(1, self.batches + 1):
            self.update_adaptive_parameters(batch)
            added = self.add_batch()
            self.vertex_queue = list(self.V)
            self.vertex_queue.sort(key=lambda node: self.vertex_key(node))
            self.edge_queue = []
            snapshots.append(self.snapshot(batch, f"batch {batch}: added {added} samples"))

            self.process_batch(batch, snapshots)
            self.prune()
            snapshots.append(self.snapshot(batch, f"batch {batch}: tightened queue bounds"))

        if self.path:
            self.update_adaptive_parameters(self.batches)
            snapshots.append(self.snapshot(self.batches, "final advanced batch-informed tree", final=True))

        if save_gif:
            self.save_process_gif(snapshots, gif_name)
        return self.path

    def update_adaptive_parameters(self, batch):
        progress = min(max((batch - 1) / max(self.batches - 1, 1), 0.0), 1.0)
        self.current_inflation = self.initial_inflation - (
            self.initial_inflation - self.final_inflation
        ) * progress
        self.current_truncation = self.initial_truncation - (
            self.initial_truncation - self.final_truncation
        ) * progress

    def process_batch(self, batch, snapshots):
        expansions = 0
        max_expansions = 760
        while expansions < max_expansions and (self.vertex_queue or self.edge_queue):
            if self.vertex_queue and self.should_expand_vertex():
                vertex = self.vertex_queue.pop(0)
                self.expand_vertex(vertex)
                if expansions < 18 or expansions % 70 == 0:
                    snapshots.append(self.snapshot(batch, "expand vertex with inflated key", focus=vertex))
            elif self.edge_queue:
                _, _, parent, child = heapq.heappop(self.edge_queue)
                accepted = self.accept_edge(parent, child, batch, snapshots)
                if accepted and (len(self.edges) <= 36 or expansions % 55 == 0):
                    edge = ((parent.x, parent.y), (child.x, child.y))
                    snapshots.append(self.snapshot(batch, "accept adaptive edge", focus=child, highlight_edge=edge))
            expansions += 1

            if expansions % 125 == 0:
                focus = min(self.vertex_queue, key=lambda node: self.vertex_key(node)) if self.vertex_queue else None
                snapshots.append(self.snapshot(batch, f"batch {batch}: inflated edge ordering", focus=focus))

    def expand_vertex(self, vertex):
        for node in self.nearby(vertex, self.samples + self.V):
            if node is vertex or self.would_create_cycle(node, vertex):
                continue
            true_bound = vertex.g + self.line(vertex, node) + self.heuristic(node)
            if true_bound >= self.truncation_bound():
                continue
            estimated = vertex.g + self.line(vertex, node) + self.current_inflation * self.heuristic(node)
            heapq.heappush(self.edge_queue, (estimated, node.id, vertex, node))

    def accept_edge(self, parent, child, batch, snapshots):
        edge_cost = self.line(parent, child)
        new_cost = parent.g + edge_cost
        if new_cost + 0.05 >= child.g:
            return False
        if new_cost + self.heuristic(child) >= self.truncation_bound():
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
            self.update_best_path(child, batch, snapshots, "goal accepted: inflation schedule tightens")
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
        self.update_best_path(self.x_goal, batch, snapshots, "solution improved: truncation narrows")

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
        bound = self.truncation_bound()
        self.samples = [
            sample for sample in self.samples
            if self.heuristic_from_start(sample) + self.heuristic(sample) < bound
        ]

    def sample_informed(self):
        sample_bound = self.truncation_bound()
        major = sample_bound / 2.0
        minor = math.sqrt(max(sample_bound ** 2 - self.c_min ** 2, 0.0)) / 2.0
        for _ in range(120):
            p = self.sample_unit_ball()
            world = self.rotation @ np.array([major * p[0], minor * p[1]]) + self.center
            node = Node(world)
            if self.in_bounds(node) and not self.utils.is_inside_obs(node):
                return node
        return self.sample_free()

    def truncation_bound(self):
        if not math.isfinite(self.best_cost):
            return math.inf
        return self.best_cost * self.current_truncation

    def vertex_key(self, node):
        return node.g + self.current_inflation * self.heuristic(node)

    def snapshot(self, batch, phase, final=False, focus=None, highlight_edge=None):
        data = super().snapshot(batch, phase, final=final, focus=focus, highlight_edge=highlight_edge)
        data["inflation"] = self.current_inflation
        data["truncation"] = self.current_truncation
        data["truncation_bound"] = self.truncation_bound() if math.isfinite(self.best_cost) else None
        data["ellipse"] = self.ellipse_parameters() if math.isfinite(self.best_cost) else None
        return data

    def ellipse_parameters(self):
        bound = self.truncation_bound()
        major = bound / 2.0
        minor = math.sqrt(max(bound ** 2 - self.c_min ** 2, 0.0)) / 2.0
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

        mode = "adaptive informed" if snapshot["ellipse"] is not None else "global"
        cost_text = "searching" if snapshot["cost"] is None else f"route {snapshot['cost']:.1f}"
        if snapshot["best_cost"] is not None:
            cost_text += f" best {snapshot['best_cost']:.1f}"
        bound_text = ""
        if snapshot["truncation_bound"] is not None:
            bound_text = f"  bound {snapshot['truncation_bound']:.1f}"
        ax.text(
            1.5,
            28.4,
            (
                f"ABIT* batch {snapshot['batch']:2d}  eps {snapshot['inflation']:.2f}  "
                f"trunc {snapshot['truncation']:.2f}{bound_text}\n"
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
        ax.set_title("054 ABIT* - adaptive inflated batch-informed search")
        fig.tight_layout(pad=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame


def main():
    random.seed(54)
    np.random.seed(54)
    planner = ABITStar((18, 8), (37, 18), batch_size=190, batches=13)
    path = planner.planning(save_gif=True)
    if not path:
        raise RuntimeError("ABIT* did not reach the goal")


if __name__ == "__main__":
    main()
