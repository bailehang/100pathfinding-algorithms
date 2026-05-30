"""
Particle Swarm Optimization (PSO) 2D path planning demo.

Each particle encodes a small set of intermediate waypoints. The swarm updates
its waypoint positions using inertia, personal best, and global best terms.
Fitness rewards short, smooth, collision-free paths with obstacle clearance, so
the demonstration shows candidate paths converging toward a feasible route.
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


class ParticleSwarmPlanner:
    def __init__(self, s_start=(5, 5), s_goal=(45, 25), seed=62):
        self.s_start = s_start
        self.s_goal = s_goal
        self.demo = GridCurveDemo(s_start, s_goal)
        self.Env = self.demo.Env
        self.obs = self.demo.obs
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)

        self.swarm_size = 48
        self.iterations = 58
        self.waypoint_count = 6
        self.inertia = 0.62
        self.cognitive = 1.45
        self.social = 1.55
        self.velocity_limit = 3.5

        self.reference_path, _ = self.demo.a_star_search()
        self.reference_waypoints = self.path_to_waypoints(self.reference_path)
        self.free_cells = np.array(
            [
                (x, y)
                for x in range(1, self.Env.x_range - 1)
                for y in range(1, self.Env.y_range - 1)
                if (x, y) not in self.obs
            ],
            dtype=float,
        )
        self.obs_array = np.array(list(self.obs), dtype=float)

        self.positions = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_score = None
        self.global_best = None
        self.global_best_score = float("inf")
        self.best_path = []
        self.best_cost = float("inf")
        self.history = []
        self.frames = []

    def planning(self):
        self.positions = self.initial_positions()
        self.velocities = self.rng.uniform(
            -0.8,
            0.8,
            size=(self.swarm_size, self.waypoint_count, 2),
        )
        self.personal_best = self.positions.copy()
        self.personal_best_score = np.array([self.fitness(position) for position in self.positions])
        best_index = int(np.argmin(self.personal_best_score))
        self.global_best = self.personal_best[best_index].copy()
        self.global_best_score = float(self.personal_best_score[best_index])
        self.best_path = self.decode(self.global_best)
        self.best_cost = self.path_cost(self.best_path)

        for iteration in range(self.iterations):
            scores = np.array([self.fitness(position) for position in self.positions])
            improved = scores < self.personal_best_score
            self.personal_best[improved] = self.positions[improved]
            self.personal_best_score[improved] = scores[improved]

            best_index = int(np.argmin(self.personal_best_score))
            if self.personal_best_score[best_index] < self.global_best_score:
                self.global_best = self.personal_best[best_index].copy()
                self.global_best_score = float(self.personal_best_score[best_index])
                self.best_path = self.decode(self.global_best)
                self.best_cost = self.path_cost(self.best_path)

            if iteration % 4 == 0 or iteration == self.iterations - 1:
                sample_indices = np.argsort(scores)[:14]
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "particles": [self.decode(self.positions[i]) for i in sample_indices],
                        "waypoints": [self.positions[i].copy() for i in sample_indices[:8]],
                        "best_path": list(self.best_path),
                        "best_cost": self.best_cost,
                        "best_score": self.global_best_score,
                        "collisions": self.collision_count(self.best_path),
                    }
                )

            if iteration % 8 == 0 or iteration == self.iterations - 1:
                print(
                    f"Iteration {iteration + 1:02d}: "
                    f"best_cost={self.best_cost:.3f}, score={self.global_best_score:.3f}, "
                    f"collisions={self.collision_count(self.best_path)}"
                )

            r1 = self.rng.random(size=self.positions.shape)
            r2 = self.rng.random(size=self.positions.shape)
            self.velocities = (
                self.inertia * self.velocities
                + self.cognitive * r1 * (self.personal_best - self.positions)
                + self.social * r2 * (self.global_best - self.positions)
            )
            self.velocities = np.clip(self.velocities, -self.velocity_limit, self.velocity_limit)
            self.positions = self.repair_positions(self.positions + self.velocities)

        return self.best_path

    def initial_positions(self):
        positions = [np.array(self.reference_waypoints, dtype=float)]
        for _ in range(self.swarm_size - 1):
            particle = []
            for waypoint in self.reference_waypoints:
                if self.random.random() < 0.76:
                    particle.append(self.jitter_point(waypoint, radius=6.0))
                else:
                    particle.append(self.free_cells[self.random.randrange(len(self.free_cells))])
            positions.append(np.array(particle, dtype=float))
        return np.array(positions, dtype=float)

    def path_to_waypoints(self, path):
        waypoints = []
        for i in range(self.waypoint_count):
            index = int((i + 1) * len(path) / (self.waypoint_count + 1))
            waypoints.append(path[min(index, len(path) - 1)])
        return waypoints

    def jitter_point(self, point, radius=5.0):
        for _ in range(30):
            candidate = np.array(
                [
                    point[0] + self.rng.uniform(-radius, radius),
                    point[1] + self.rng.uniform(-radius, radius),
                ],
                dtype=float,
            )
            candidate = self.clamp_point(candidate)
            if self.demo.is_valid(candidate):
                return candidate
        return np.array(point, dtype=float)

    def repair_positions(self, positions):
        repaired = positions.copy()
        repaired[:, :, 0] = np.clip(repaired[:, :, 0], 1, self.Env.x_range - 2)
        repaired[:, :, 1] = np.clip(repaired[:, :, 1], 1, self.Env.y_range - 2)
        for particle_index in range(repaired.shape[0]):
            for waypoint_index in range(repaired.shape[1]):
                if not self.demo.is_valid(repaired[particle_index, waypoint_index]):
                    repaired[particle_index, waypoint_index] = self.nearest_free_point(
                        repaired[particle_index, waypoint_index]
                    )
        return repaired

    def nearest_free_point(self, point):
        distances = np.linalg.norm(self.free_cells - point, axis=1)
        return self.free_cells[int(np.argmin(distances))].copy()

    def clamp_point(self, point):
        return np.array(
            [
                min(max(point[0], 1), self.Env.x_range - 2),
                min(max(point[1], 1), self.Env.y_range - 2),
            ],
            dtype=float,
        )

    def decode(self, particle):
        raw_path = [self.s_start]
        for waypoint in particle:
            rounded = (float(waypoint[0]), float(waypoint[1]))
            if self.demo.distance(raw_path[-1], rounded) > 0.3:
                raw_path.append(rounded)
        raw_path.append(self.s_goal)
        return raw_path

    def fitness(self, particle):
        path = self.decode(particle)
        length = self.path_cost(path)
        collisions = self.collision_count(path)
        clearance = self.minimum_clearance(path)
        turns = self.turning_cost(path)
        obstacle_penalty = 90.0 * collisions
        clearance_penalty = 12.0 / max(clearance, 0.25)
        return length + obstacle_penalty + 5.0 * turns + clearance_penalty

    def path_cost(self, path):
        return sum(self.demo.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def collision_count(self, path):
        count = 0
        for i in range(len(path) - 1):
            if self.segment_collides(path[i], path[i + 1]):
                count += 1
        return count

    def segment_collides(self, start, end):
        samples = max(2, int(self.demo.distance(start, end) * 2.5))
        for t in np.linspace(0.0, 1.0, samples):
            point = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
            if not self.demo.is_valid(point):
                return True
        return False

    def minimum_clearance(self, path):
        min_clearance = float("inf")
        for i in range(len(path) - 1):
            samples = max(2, int(self.demo.distance(path[i], path[i + 1]) * 1.5))
            for t in np.linspace(0.0, 1.0, samples):
                point = np.array(
                    [
                        path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                        path[i][1] + (path[i + 1][1] - path[i][1]) * t,
                    ],
                    dtype=float,
                )
                min_clearance = min(
                    min_clearance,
                    float(np.min(np.linalg.norm(self.obs_array - point, axis=1))),
                )
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

    def draw_particles(self, paths, waypoint_sets, fraction=1.0):
        for path in paths:
            if len(path) < 2:
                continue
            upto = max(2, int(len(path) * fraction))
            visible = path[:upto]
            collision_free = self.collision_count(path) == 0
            color = "tab:green" if collision_free else "tab:blue"
            alpha = 0.26 if collision_free else 0.16
            plt.plot(
                [p[0] for p in visible],
                [p[1] for p in visible],
                color=color,
                linewidth=1.1,
                alpha=alpha,
            )

        for waypoints in waypoint_sets:
            plt.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                s=14,
                color="tab:orange",
                alpha=0.42,
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

    def run_demonstration(self):
        print("Starting Particle Swarm Optimization demonstration...")
        self.planning()

        plt.figure(figsize=(7, 5), dpi=100)
        for item in self.history:
            for fraction, stage in ((0.45, "velocity update"), (1.0, "swarm evaluation")):
                self.draw_base(f"062 PSO - Iteration {item['iteration']} ({stage})")
                self.draw_particles(item["particles"], item["waypoints"], fraction=fraction)
                if item["best_path"]:
                    plt.plot(
                        [p[0] for p in item["best_path"]],
                        [p[1] for p in item["best_path"]],
                        color="crimson",
                        linewidth=3.0,
                        label="global best",
                    )
                plt.text(
                    0.02,
                    0.95,
                    (
                        f"best cost {item['best_cost']:.2f} | "
                        f"score {item['best_score']:.2f} | collisions {item['collisions']}"
                    ),
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
                )
                self.capture_frame()

        self.save_gif("062_PSO", fps=3)
        plt.close("all")
        print(f"Best path waypoints: {len(self.best_path)}")
        print(f"Best path cost: {self.best_cost:.3f}")
        print(f"Best path collisions: {self.collision_count(self.best_path)}")


def main():
    planner = ParticleSwarmPlanner()
    planner.run_demonstration()


if __name__ == "__main__":
    main()
