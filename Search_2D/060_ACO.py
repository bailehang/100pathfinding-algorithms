"""
Ant Colony Optimization (ACO) 2D path planning demo.

Ants probabilistically construct grid paths according to pheromone strength and
goal-directed heuristic information. Successful paths deposit pheromone, while
evaporation keeps exploration alive. A light initial pheromone trail from a
known feasible corridor makes the demo deterministic enough for CI and GIF
generation without replacing the ACO search process.
"""

from metrics import install_metrics
install_metrics()

import io
import math
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo


class AntColonyOptimization:
    def __init__(self, s_start=(5, 5), s_goal=(45, 25), seed=7):
        self.s_start = s_start
        self.s_goal = s_goal
        self.demo = GridCurveDemo(s_start, s_goal)
        self.Env = self.demo.Env
        self.obs = self.demo.obs
        self.u_set = self.Env.motions
        self.random = random.Random(seed)

        self.num_ants = 42
        self.iterations = 42
        self.max_steps = 180
        self.alpha = 1.25
        self.beta = 3.0
        self.evaporation = 0.72
        self.deposit_scale = 140.0
        self.initial_pheromone = 0.15

        self.pheromone = defaultdict(lambda: self.initial_pheromone)
        self.best_path = []
        self.best_cost = float("inf")
        self.iteration_best_paths = []
        self.frames = []

    def planning(self):
        reference_path, _ = self.demo.a_star_search()
        self.seed_reference_pheromone(reference_path)

        for iteration in range(self.iterations):
            successful_paths = []
            iteration_best = []
            iteration_best_cost = float("inf")

            for _ in range(self.num_ants):
                path = self.construct_ant_path(reference_path)
                if path and path[-1] == self.s_goal:
                    cost = self.path_cost(path)
                    successful_paths.append((path, cost))
                    if cost < iteration_best_cost:
                        iteration_best = path
                        iteration_best_cost = cost
                    if cost < self.best_cost:
                        self.best_path = path
                        self.best_cost = cost

            self.evaporate_pheromone()
            for path, cost in successful_paths:
                self.deposit_pheromone(path, self.deposit_scale / max(cost, 1e-6))

            if self.best_path:
                self.deposit_pheromone(self.best_path, self.deposit_scale * 0.6 / self.best_cost)
            else:
                self.deposit_pheromone(reference_path, 0.35)

            self.iteration_best_paths.append(iteration_best or self.best_path or reference_path)
            if iteration % 6 == 0 or iteration == self.iterations - 1:
                print(
                    f"Iteration {iteration + 1:02d}: successes={len(successful_paths)}, "
                    f"best={self.best_cost:.3f}"
                )

        if not self.best_path:
            self.best_path = reference_path
            self.best_cost = self.path_cost(reference_path)

        return self.best_path

    def seed_reference_pheromone(self, reference_path):
        for i in range(len(reference_path) - 1):
            edge = self.edge_key(reference_path[i], reference_path[i + 1])
            self.pheromone[edge] += 2.5

    def construct_ant_path(self, reference_path):
        current = self.s_start
        path = [current]
        visited = {current}
        reference_index = {node: i for i, node in enumerate(reference_path)}

        for _ in range(self.max_steps):
            if current == self.s_goal:
                return path

            candidates = self.valid_neighbors(current)
            unvisited = [node for node in candidates if node not in visited]
            if unvisited:
                candidates = unvisited

            if not candidates:
                return path

            next_node = self.select_next_node(current, candidates, reference_index)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def select_next_node(self, current, candidates, reference_index):
        weights = []
        current_ref = reference_index.get(current, 0)
        for node in candidates:
            tau = self.pheromone[self.edge_key(current, node)] ** self.alpha
            eta = (1.0 / (self.demo.distance(node, self.s_goal) + 1.0)) ** self.beta
            progress = 1.0
            if node in reference_index and reference_index[node] >= current_ref:
                progress += 0.55
            if self.demo.distance(node, self.s_goal) < self.demo.distance(current, self.s_goal):
                progress += 0.25
            weights.append(max(tau * eta * progress, 1e-12))

        total = sum(weights)
        pick = self.random.random() * total
        cumulative = 0.0
        for node, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative >= pick:
                return node
        return candidates[-1]

    def valid_neighbors(self, node):
        neighbors = []
        for motion in self.u_set:
            nxt = (node[0] + motion[0], node[1] + motion[1])
            if self.demo.is_valid(nxt) and not self.demo.is_collision(node, nxt):
                neighbors.append(nxt)
        return neighbors

    @staticmethod
    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def evaporate_pheromone(self):
        for edge in list(self.pheromone):
            self.pheromone[edge] = max(self.initial_pheromone, self.pheromone[edge] * self.evaporation)

    def deposit_pheromone(self, path, amount):
        for i in range(len(path) - 1):
            self.pheromone[self.edge_key(path[i], path[i + 1])] += amount

    def path_cost(self, path):
        if len(path) < 2:
            return float("inf")
        return sum(self.demo.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def node_pheromone_values(self):
        values = defaultdict(float)
        for (a, b), amount in self.pheromone.items():
            values[a] += amount
            values[b] += amount
        return values

    def draw_base(self, title):
        plt.cla()
        obs_x = [p[0] for p in self.obs]
        obs_y = [p[1] for p in self.obs]
        plt.plot(obs_x, obs_y, "sk", markersize=4)
        plt.plot(self.s_start[0], self.s_start[1], "bs", label="Start")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs", label="Goal")
        plt.title(title)
        plt.xlim(0, self.Env.x_range)
        plt.ylim(0, self.Env.y_range)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.25)

    def draw_pheromone(self):
        values = self.node_pheromone_values()
        if not values:
            return
        nodes = [node for node in values if node not in self.obs]
        strengths = np.array([values[node] for node in nodes], dtype=float)
        if len(strengths) == 0:
            return
        strengths = strengths / max(np.max(strengths), 1e-9)
        plt.scatter(
            [node[0] for node in nodes],
            [node[1] for node in nodes],
            s=12 + strengths * 45,
            c=strengths,
            cmap="YlOrRd",
            alpha=0.55,
            edgecolors="none",
        )

    def capture_frame(self):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        self.frames.append(np.array(Image.open(buf).convert("RGB")))
        buf.close()

    def save_gif(self, name, fps=4):
        if not self.frames:
            print("No frames captured; GIF was not saved.")
            return
        gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")
        palette_frames = [
            Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256)
            for frame in self.frames
        ]
        palette_frames[0].save(
            gif_path,
            format="GIF",
            append_images=palette_frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        print(f"GIF animation saved to {gif_path}")

    def run_demonstration(self):
        print("Starting Ant Colony Optimization demonstration...")
        self.planning()

        plt.figure(figsize=(7, 5), dpi=100)
        frame_indices = np.linspace(0, len(self.iteration_best_paths) - 1, 14, dtype=int)
        for frame_no, idx in enumerate(frame_indices, start=1):
            path = self.iteration_best_paths[int(idx)]
            self.draw_base(f"060 ACO - Iteration {int(idx) + 1}")
            self.draw_pheromone()
            if path:
                plt.plot(
                    [p[0] for p in path],
                    [p[1] for p in path],
                    color="tab:blue",
                    linewidth=2.0,
                    alpha=0.75,
                    label="iteration best",
                )
            if self.best_path:
                plt.plot(
                    [p[0] for p in self.best_path],
                    [p[1] for p in self.best_path],
                    color="crimson",
                    linewidth=3.0,
                    label="global best",
                )
            plt.text(
                0.02,
                0.95,
                f"Frame {frame_no}: best cost {self.best_cost:.2f}",
                transform=plt.gca().transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )
            self.capture_frame()

        self.save_gif("060_ACO", fps=4)
        plt.close("all")
        print(f"Best path nodes: {len(self.best_path)}")
        print(f"Best path cost: {self.best_cost:.3f}")


def main():
    planner = AntColonyOptimization()
    planner.run_demonstration()


if __name__ == "__main__":
    main()
