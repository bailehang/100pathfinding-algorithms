"""
Genetic Algorithm (GA) 2D path planning demo.

Each individual is a sequence of waypoint genes. The population evolves through
tournament selection, crossover, mutation, and elitism. Fitness combines path
length, obstacle clearance, and collision penalties so the best chromosomes
converge toward a short collision-free path.
"""

from metrics import install_metrics
install_metrics()

import io
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo


class GeneticAlgorithmPlanner:
    def __init__(self, s_start=(5, 5), s_goal=(45, 25), seed=11):
        self.s_start = s_start
        self.s_goal = s_goal
        self.demo = GridCurveDemo(s_start, s_goal)
        self.Env = self.demo.Env
        self.obs = self.demo.obs
        self.random = random.Random(seed)

        self.population_size = 58
        self.generations = 54
        self.gene_count = 7
        self.elite_count = 5
        self.mutation_rate = 0.32
        self.crossover_rate = 0.82

        self.free_cells = [
            (x, y)
            for x in range(1, self.Env.x_range - 1)
            for y in range(1, self.Env.y_range - 1)
            if (x, y) not in self.obs
        ]
        self.reference_path, _ = self.demo.a_star_search()
        self.reference_waypoints = self.demo.extract_safe_waypoints(self.reference_path)

        self.population = []
        self.best_path = []
        self.best_fitness = -float("inf")
        self.best_cost = float("inf")
        self.history = []
        self.frames = []

    def planning(self):
        self.population = self.initial_population()

        for generation in range(self.generations):
            scored = sorted(
                [(self.fitness(individual), individual) for individual in self.population],
                key=lambda item: item[0],
                reverse=True,
            )
            best_fitness, best_individual = scored[0]
            best_path = self.decode(best_individual)
            best_cost = self.path_cost(best_path)

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_path = best_path
                self.best_cost = best_cost

            if generation % 4 == 0 or generation == self.generations - 1:
                self.history.append(
                    {
                        "generation": generation + 1,
                        "population": [self.decode(ind) for _, ind in scored[:14]],
                        "best_path": list(self.best_path),
                        "best_cost": self.best_cost,
                        "best_fitness": self.best_fitness,
                    }
                )

            if generation % 8 == 0 or generation == self.generations - 1:
                print(
                    f"Generation {generation + 1:02d}: "
                    f"best_cost={self.best_cost:.3f}, fitness={self.best_fitness:.4f}"
                )

            next_population = [individual for _, individual in scored[: self.elite_count]]
            while len(next_population) < self.population_size:
                parent_a = self.tournament(scored)
                parent_b = self.tournament(scored)
                if self.random.random() < self.crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = list(parent_a)
                child = self.mutate(child)
                next_population.append(child)

            self.population = next_population

        return self.best_path

    def initial_population(self):
        population = []
        reference_genes = self.path_to_genes(self.reference_path)
        population.append(reference_genes)

        for _ in range(self.population_size - 1):
            genes = []
            for i in range(self.gene_count):
                if self.random.random() < 0.72 and self.reference_path:
                    index = int((i + 1) * len(self.reference_path) / (self.gene_count + 1))
                    anchor = self.reference_path[min(index, len(self.reference_path) - 1)]
                    genes.append(self.jitter_cell(anchor, radius=4))
                else:
                    genes.append(self.random.choice(self.free_cells))
            population.append(genes)
        return population

    def path_to_genes(self, path):
        genes = []
        for i in range(self.gene_count):
            index = int((i + 1) * len(path) / (self.gene_count + 1))
            genes.append(path[min(index, len(path) - 1)])
        return genes

    def decode(self, individual):
        raw_path = [self.s_start] + list(individual) + [self.s_goal]
        clean = [raw_path[0]]
        for point in raw_path[1:]:
            if point != clean[-1]:
                clean.append(point)
        return clean

    def fitness(self, individual):
        path = self.decode(individual)
        length = self.path_cost(path)
        collisions = self.collision_count(path)
        clearance = self.minimum_clearance(path)
        turn_penalty = self.turning_cost(path)
        return 1.0 / (length + 60.0 * collisions + 4.0 * turn_penalty + 8.0 / max(clearance, 0.25))

    def path_cost(self, path):
        return sum(self.demo.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def collision_count(self, path):
        count = 0
        for i in range(len(path) - 1):
            if self.segment_collides(path[i], path[i + 1]):
                count += 1
        return count

    def segment_collides(self, start, end):
        samples = max(2, int(self.demo.distance(start, end) * 2))
        for t in np.linspace(0.0, 1.0, samples):
            x = int(round(start[0] + (end[0] - start[0]) * t))
            y = int(round(start[1] + (end[1] - start[1]) * t))
            if (x, y) in self.obs or not (0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range):
                return True
        return False

    def minimum_clearance(self, path):
        min_clearance = float("inf")
        obs_array = np.array(list(self.obs), dtype=float)
        for i in range(len(path) - 1):
            samples = max(2, int(self.demo.distance(path[i], path[i + 1])))
            for t in np.linspace(0.0, 1.0, samples):
                point = np.array(
                    [
                        path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                        path[i][1] + (path[i + 1][1] - path[i][1]) * t,
                    ]
                )
                min_clearance = min(min_clearance, float(np.min(np.linalg.norm(obs_array - point, axis=1))))
        return min_clearance

    @staticmethod
    def turning_cost(path):
        if len(path) < 3:
            return 0.0
        total = 0.0
        for i in range(1, len(path) - 1):
            a = np.array(path[i]) - np.array(path[i - 1])
            b = np.array(path[i + 1]) - np.array(path[i])
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                continue
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_angle = min(1.0, max(-1.0, float(cos_angle)))
            total += abs(math.acos(cos_angle))
        return total

    def tournament(self, scored, k=4):
        candidates = self.random.sample(scored, k)
        return list(max(candidates, key=lambda item: item[0])[1])

    def crossover(self, parent_a, parent_b):
        if len(parent_a) <= 1:
            return list(parent_a)
        point = self.random.randint(1, len(parent_a) - 1)
        return list(parent_a[:point]) + list(parent_b[point:])

    def mutate(self, individual):
        mutated = list(individual)
        for i, gene in enumerate(mutated):
            if self.random.random() < self.mutation_rate:
                if self.random.random() < 0.75:
                    mutated[i] = self.jitter_cell(gene, radius=5)
                else:
                    mutated[i] = self.random.choice(self.free_cells)
        return mutated

    def jitter_cell(self, cell, radius=4):
        for _ in range(20):
            x = cell[0] + self.random.randint(-radius, radius)
            y = cell[1] + self.random.randint(-radius, radius)
            x = min(max(x, 1), self.Env.x_range - 2)
            y = min(max(y, 1), self.Env.y_range - 2)
            if (x, y) not in self.obs:
                return (x, y)
        return cell if cell not in self.obs else self.random.choice(self.free_cells)

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

    def save_gif(self, name, fps=3):
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

    def draw_population(self, paths, fraction=1.0):
        for path in paths:
            if len(path) < 2:
                continue
            upto = max(2, int(len(path) * fraction))
            visible = path[:upto]
            color = "tab:green" if self.collision_count(path) == 0 else "tab:blue"
            alpha = 0.22 if color == "tab:blue" else 0.32
            plt.plot(
                [p[0] for p in visible],
                [p[1] for p in visible],
                color=color,
                linewidth=1.2,
                alpha=alpha,
            )

    def run_demonstration(self):
        print("Starting Genetic Algorithm demonstration...")
        self.planning()

        plt.figure(figsize=(7, 5), dpi=100)
        for item in self.history:
            for fraction, stage in ((0.45, "population sampling"), (1.0, "selection result")):
                self.draw_base(f"061 GA - Generation {item['generation']} ({stage})")
                self.draw_population(item["population"], fraction=fraction)
                if item["best_path"]:
                    plt.plot(
                        [p[0] for p in item["best_path"]],
                        [p[1] for p in item["best_path"]],
                        color="crimson",
                        linewidth=3.0,
                        label="best chromosome",
                    )
                plt.text(
                    0.02,
                    0.95,
                    f"best cost {item['best_cost']:.2f} | fitness {item['best_fitness']:.4f}",
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
                )
                self.capture_frame()

        self.save_gif("066_GA", fps=3)
        plt.close("all")
        print(f"Best path genes: {len(self.best_path)}")
        print(f"Best path cost: {self.best_cost:.3f}")
        print(f"Best path collisions: {self.collision_count(self.best_path)}")


def main():
    planner = GeneticAlgorithmPlanner()
    planner.run_demonstration()


if __name__ == "__main__":
    main()
