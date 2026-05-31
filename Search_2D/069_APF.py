"""
Artificial Potential Field (APF) 2D path planning demo.

The planner treats the goal as an attractive potential and obstacles as
repulsive potentials. A small tangential component around the closest obstacle
helps the demo avoid the common APF local-minimum trap on the shared grid map.
"""

from metrics import install_metrics
install_metrics()

import io
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D import env
from Search_2D.curve_demo_utils import GridCurveDemo


class ArtificialPotentialField:
    def __init__(self, s_start=(5.0, 5.0), s_goal=(45.0, 25.0)):
        self.s_start = np.array(s_start, dtype=float)
        self.s_goal = np.array(s_goal, dtype=float)

        self.Env = env.Env()
        self.obs = self.Env.obs
        self.obstacles = np.array(sorted(self.obs), dtype=float)

        self.k_att = 2.5
        self.k_rep = 8.0
        self.influence_radius = 2.5
        self.step_size = 0.45
        self.goal_tolerance = 0.65
        self.max_iterations = 450

        self.path = []
        self.guide_waypoints = []
        self.force_history = []
        self.frames = []

    def planning(self):
        guide = GridCurveDemo(tuple(self.s_start), tuple(self.s_goal))
        raw_path, _ = guide.a_star_search()
        self.guide_waypoints = raw_path

        position = self.s_start.copy()
        self.path = [tuple(position)]
        stagnant_steps = 0
        target_index = 1 if len(self.guide_waypoints) > 1 else 0
        recent_cells = []

        for _ in range(self.max_iterations):
            target = np.array(self.guide_waypoints[target_index], dtype=float)
            current_cell = (int(round(position[0])), int(round(position[1])))
            recent_cells.append(current_cell)
            recent_cells = recent_cells[-16:]
            if recent_cells.count(current_cell) >= 4 and target_index < len(self.guide_waypoints) - 1:
                target_index += 1
                recent_cells = []
                target = np.array(self.guide_waypoints[target_index], dtype=float)

            if np.linalg.norm(position - target) <= 1.45 and target_index < len(self.guide_waypoints) - 1:
                target_index += 1
                target = np.array(self.guide_waypoints[target_index], dtype=float)

            force, closest_obstacle = self.total_force(position, stagnant_steps, target)
            norm = np.linalg.norm(force)
            if norm < 1e-9:
                break

            direction = force / norm
            next_position = self.safe_step(position, direction)

            movement = np.linalg.norm(next_position - position)
            stagnant_steps = stagnant_steps + 1 if movement < 0.05 else 0

            position = next_position
            self.path.append(tuple(position))
            self.force_history.append((tuple(position), tuple(direction), closest_obstacle))

            if np.linalg.norm(position - self.s_goal) <= self.goal_tolerance:
                self.path.append(tuple(self.s_goal))
                break

        return self.path

    def total_force(self, position, stagnant_steps, target=None):
        if target is None:
            target = self.s_goal
        attractive = self.k_att * (target - position)
        repulsive, closest_obstacle, nearest_distance = self.repulsive_force(position)
        tangential = np.array([0.0, 0.0])

        if closest_obstacle is not None and nearest_distance < self.influence_radius * 1.15:
            radial = position - closest_obstacle
            if np.linalg.norm(radial) > 1e-9:
                radial = radial / np.linalg.norm(radial)
                tangential = np.array([-radial[1], radial[0]])
                turn_sign = np.sign(np.cross(np.append(radial, 0.0), np.append(target - position, 0.0))[2])
                if turn_sign == 0:
                    turn_sign = 1.0
                tangential *= turn_sign * (0.55 + 0.15 * min(stagnant_steps, 8))

        return attractive + repulsive + tangential, closest_obstacle

    def repulsive_force(self, position):
        deltas = position - self.obstacles
        distances = np.linalg.norm(deltas, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = max(float(distances[nearest_index]), 1e-6)
        closest_obstacle = self.obstacles[nearest_index]

        mask = distances < self.influence_radius
        force = np.array([0.0, 0.0])
        for delta, distance in zip(deltas[mask], distances[mask]):
            distance = max(float(distance), 1e-6)
            direction = delta / distance
            strength = self.k_rep * (1.0 / distance - 1.0 / self.influence_radius) / (distance ** 2)
            force += strength * direction

        return force, closest_obstacle, nearest_distance

    def safe_step(self, position, direction):
        candidate_directions = [direction]
        for angle_deg in (20, -20, 40, -40, 70, -70, 100, -100):
            angle = math.radians(angle_deg)
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            candidate_directions.append(rot @ direction)

        for candidate in candidate_directions:
            next_position = position + candidate * self.step_size
            next_position[0] = min(max(next_position[0], 1.0), self.Env.x_range - 2.0)
            next_position[1] = min(max(next_position[1], 1.0), self.Env.y_range - 2.0)
            if self.segment_is_safe(position, next_position):
                return next_position

        return position

    def segment_is_safe(self, start, end):
        samples = max(2, int(np.linalg.norm(end - start) / 0.1))
        for t in np.linspace(0.0, 1.0, samples):
            point = start + (end - start) * t
            rounded = (int(round(point[0])), int(round(point[1])))
            if rounded in self.obs:
                return False
        return True

    def path_length(self):
        if len(self.path) < 2:
            return 0.0
        return sum(
            math.hypot(self.path[i + 1][0] - self.path[i][0], self.path[i + 1][1] - self.path[i][1])
            for i in range(len(self.path) - 1)
        )

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

    def draw_vector_field(self):
        xs = np.arange(4, self.Env.x_range - 3, 5)
        ys = np.arange(4, self.Env.y_range - 3, 5)
        u_values, v_values, x_values, y_values = [], [], [], []

        for x in xs:
            for y in ys:
                if (int(x), int(y)) in self.obs:
                    continue
                force, _ = self.total_force(np.array([x, y], dtype=float), stagnant_steps=0)
                norm = np.linalg.norm(force)
                if norm <= 1e-9:
                    continue
                direction = force / norm
                x_values.append(x)
                y_values.append(y)
                u_values.append(direction[0])
                v_values.append(direction[1])

        plt.quiver(x_values, y_values, u_values, v_values, color="tab:cyan", alpha=0.55, width=0.004)

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

    def save_gif(self, name, fps=5):
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
        print("Starting Artificial Potential Field demonstration...")
        self.planning()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_base("064 APF - Potential Field")
        self.draw_vector_field()
        self.capture_frame()

        frame_count = 16
        for i in range(1, frame_count + 1):
            upto = max(2, int(len(self.path) * i / frame_count))
            visible_path = self.path[:upto]
            self.draw_base("064 APF - Gradient Descent Path")
            self.draw_vector_field()
            if self.guide_waypoints:
                plt.plot(
                    [p[0] for p in self.guide_waypoints],
                    [p[1] for p in self.guide_waypoints],
                    "o--",
                    color="tab:blue",
                    alpha=0.35,
                    linewidth=1.8,
                    label="local goals",
                )
            plt.plot(
                [p[0] for p in visible_path],
                [p[1] for p in visible_path],
                color="crimson",
                linewidth=2.8,
                label="APF path",
            )
            current = visible_path[-1]
            plt.plot(current[0], current[1], "o", color="tab:orange", markersize=8)
            self.capture_frame()

        self.save_gif("069_APF", fps=5)
        plt.close("all")

        reached = np.linalg.norm(np.array(self.path[-1]) - self.s_goal) <= self.goal_tolerance
        print(f"Reached goal: {reached}")
        print(f"APF path samples: {len(self.path)}")
        print(f"APF path length: {self.path_length():.3f}")


def main():
    planner = ArtificialPotentialField()
    planner.run_demonstration()


if __name__ == "__main__":
    main()
