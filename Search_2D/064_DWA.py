"""
Dynamic Window Approach (DWA) 2D local planning demo.

DWA samples feasible linear/angular velocities from the robot's dynamic window,
rolls out short unicycle trajectories, scores them by goal heading, obstacle
clearance, speed, and progress, then applies the best command.
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

from Search_2D.curve_demo_utils import GridCurveDemo, wrap_pi


class DynamicWindowApproach:
    def __init__(self, s_start=(5.0, 5.0), s_goal=(45.0, 25.0)):
        self.s_start = np.array(s_start, dtype=float)
        self.s_goal = np.array(s_goal, dtype=float)
        self.demo = GridCurveDemo(tuple(self.s_start), tuple(self.s_goal))
        self.Env = self.demo.Env
        self.obs = self.demo.obs
        self.obstacles = np.array(sorted(self.obs), dtype=float)

        self.max_speed = 1.45
        self.min_speed = 0.0
        self.max_yaw_rate = math.radians(95.0)
        self.max_accel = 1.15
        self.max_delta_yaw_rate = math.radians(115.0)
        self.v_resolution = 0.12
        self.yaw_rate_resolution = math.radians(10.0)
        self.dt = 0.18
        self.predict_time = 2.0
        self.robot_radius = 0.45
        self.goal_tolerance = 0.9
        self.max_steps = 340

        self.heading_weight = 1.4
        self.clearance_weight = 0.45
        self.speed_weight = 0.25
        self.progress_weight = 1.4

        self.state = np.array([self.s_start[0], self.s_start[1], math.atan2(1.0, 1.0), 0.0, 0.0])
        self.path = [tuple(self.s_start)]
        self.frames = []
        self.guide_waypoints = []
        self.candidate_trajectories = []
        self.best_trajectory = []

    def planning(self):
        raw_path, _ = self.demo.a_star_search()
        self.guide_waypoints = raw_path[::2]
        if raw_path[-1] != self.guide_waypoints[-1]:
            self.guide_waypoints.append(raw_path[-1])

        target_index = 1 if len(self.guide_waypoints) > 1 else 0
        stagnant_steps = 0
        for _ in range(self.max_steps):
            local_goal = np.array(self.guide_waypoints[target_index], dtype=float)
            if (
                np.linalg.norm(self.state[:2] - local_goal) < 1.8
                or stagnant_steps > 10
            ) and target_index < len(self.guide_waypoints) - 1:
                target_index += 1
                stagnant_steps = 0
                local_goal = np.array(self.guide_waypoints[target_index], dtype=float)

            previous_position = self.state[:2].copy()
            control, best_traj, candidates = self.select_control(local_goal)
            self.candidate_trajectories = candidates
            self.best_trajectory = best_traj
            self.state = self.motion(self.state, control, self.dt)
            self.path.append((float(self.state[0]), float(self.state[1])))
            if np.linalg.norm(self.state[:2] - previous_position) < 0.03:
                stagnant_steps += 1
            else:
                stagnant_steps = 0

            if np.linalg.norm(self.state[:2] - self.s_goal) <= self.goal_tolerance:
                self.path.append(tuple(self.s_goal))
                break

        return self.path

    def dynamic_window(self):
        vs = [
            self.min_speed,
            self.max_speed,
            self.state[3] - self.max_accel * self.dt,
            self.state[3] + self.max_accel * self.dt,
        ]
        ws = [
            -self.max_yaw_rate,
            self.max_yaw_rate,
            self.state[4] - self.max_delta_yaw_rate * self.dt,
            self.state[4] + self.max_delta_yaw_rate * self.dt,
        ]
        return [
            max(vs[0], vs[2]),
            min(vs[1], vs[3]),
            max(ws[0], ws[2]),
            min(ws[1], ws[3]),
        ]

    def select_control(self, local_goal):
        dw = self.dynamic_window()
        best_score = -float("inf")
        best_control = np.array([0.0, 0.0])
        best_traj = []
        candidates = []

        v_values = np.arange(dw[0], dw[1] + self.v_resolution, self.v_resolution)
        w_values = np.arange(dw[2], dw[3] + self.yaw_rate_resolution, self.yaw_rate_resolution)
        if len(v_values) == 0:
            v_values = np.array([dw[0]])
        if len(w_values) == 0:
            w_values = np.array([dw[2]])

        for v in v_values:
            for w in w_values:
                control = np.array([float(v), float(w)])
                traj = self.predict_trajectory(self.state, control)
                if self.trajectory_collides(traj):
                    continue

                score = self.score_trajectory(traj, control, local_goal)
                candidates.append(traj)
                if score > best_score:
                    best_score = score
                    best_control = control
                    best_traj = traj

        if not best_traj:
            recovery_turn = self.max_delta_yaw_rate * self.dt
            best_control = np.array([0.0, recovery_turn])
            best_traj = self.predict_trajectory(self.state, best_control)

        return best_control, best_traj, candidates

    def predict_trajectory(self, state, control):
        traj = [state.copy()]
        rollout = state.copy()
        elapsed = 0.0
        while elapsed <= self.predict_time:
            rollout = self.motion(rollout, control, self.dt)
            traj.append(rollout.copy())
            elapsed += self.dt
        return traj

    @staticmethod
    def motion(state, control, dt):
        x, y, yaw, _, _ = state
        v, w = control
        yaw = wrap_pi(yaw + w * dt)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        return np.array([x, y, yaw, v, w], dtype=float)

    def trajectory_collides(self, traj):
        for state in traj:
            if not (1.0 <= state[0] <= self.Env.x_range - 2.0 and 1.0 <= state[1] <= self.Env.y_range - 2.0):
                return True
            nearest = self.nearest_obstacle_distance(state[:2])
            if nearest <= self.robot_radius:
                return True
        return False

    def nearest_obstacle_distance(self, position):
        distances = np.linalg.norm(self.obstacles - position, axis=1)
        return float(np.min(distances))

    def score_trajectory(self, traj, control, local_goal):
        final = traj[-1]
        to_goal = local_goal - final[:2]
        goal_heading = math.atan2(to_goal[1], to_goal[0])
        heading_score = math.pi - abs(wrap_pi(goal_heading - final[2]))
        clearance = min(self.nearest_obstacle_distance(state[:2]) for state in traj)
        clearance_score = min(clearance, 6.0) / 6.0
        speed_score = control[0] / self.max_speed if self.max_speed > 0 else 0.0
        progress_score = -np.linalg.norm(final[:2] - local_goal)
        return (
            self.heading_weight * heading_score
            + self.clearance_weight * clearance_score
            + self.speed_weight * speed_score
            + self.progress_weight * progress_score
        )

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

    def draw_robot(self):
        x, y, yaw, _, _ = self.state
        circle = plt.Circle((x, y), self.robot_radius, color="tab:orange", alpha=0.85)
        plt.gca().add_patch(circle)
        plt.arrow(
            x,
            y,
            math.cos(yaw) * 1.0,
            math.sin(yaw) * 1.0,
            head_width=0.35,
            head_length=0.45,
            color="black",
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
        print("Starting Dynamic Window Approach demonstration...")
        self.planning()

        plt.figure(figsize=(7, 5), dpi=100)
        frame_count = 18
        replay_indices = np.linspace(1, len(self.path) - 1, frame_count, dtype=int)
        for index in replay_indices:
            self.draw_base("064 DWA - Dynamic Window Local Planner")
            if self.guide_waypoints:
                plt.plot(
                    [p[0] for p in self.guide_waypoints],
                    [p[1] for p in self.guide_waypoints],
                    "o--",
                    color="tab:blue",
                    alpha=0.35,
                    linewidth=1.8,
                    label="reference waypoints",
                )
            partial_path = self.path[:index + 1]
            plt.plot([p[0] for p in partial_path], [p[1] for p in partial_path], color="crimson", linewidth=2.8)
            # Reconstruct a visible robot state for replay frames.
            p0 = np.array(self.path[max(0, index - 1)])
            p1 = np.array(self.path[index])
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if np.linalg.norm(p1 - p0) > 1e-9 else self.state[2]
            self.state = np.array([p1[0], p1[1], yaw, self.state[3], self.state[4]])
            self.draw_robot()
            self.capture_frame()

        self.save_gif("064_DWA", fps=5)
        plt.close("all")
        reached = np.linalg.norm(np.array(self.path[-1]) - self.s_goal) <= self.goal_tolerance
        print(f"Reached goal: {reached}")
        print(f"DWA path samples: {len(self.path)}")
        print(f"DWA path length: {self.path_length():.3f}")


def main():
    planner = DynamicWindowApproach()
    planner.run_demonstration()


if __name__ == "__main__":
    main()
