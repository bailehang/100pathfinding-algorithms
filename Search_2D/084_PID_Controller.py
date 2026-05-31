"""PID controller path following demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.curve_demo_utils import GridCurveDemo, wrap_pi


class PIDControllerPathFollower(GridCurveDemo):
    def __init__(self, s_start=(5, 5), s_goal=(45, 25)):
        super().__init__(s_start, s_goal)
        self.dt = 0.16
        self.speed = 1.7
        self.lookahead = 2.8
        self.kp = 2.4
        self.ki = 0.10
        self.kd = 0.78
        self.reference_path = []
        self.tracked_path = []
        self.control_history = []
        self.target_history = []

    def plan(self):
        start_ms = now_ms()
        self.raw_path, self.visited = self.a_star_search()
        self.reference_path = self.densify_reference(self.raw_path)
        self.follow_reference()
        elapsed = now_ms() - start_ms
        print_metrics_for(self.tracked_path, elapsed, source="pid")
        return self.tracked_path, self.visited

    def densify_reference(self, path):
        if len(path) < 2:
            return [(float(x), float(y)) for x, y in path]

        samples = []
        for i in range(len(path) - 1):
            start = np.array(path[i], dtype=float)
            end = np.array(path[i + 1], dtype=float)
            distance = np.linalg.norm(end - start)
            count = max(2, int(distance / 0.18))
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                samples.append(tuple(start + (end - start) * t))
        samples.append(tuple(map(float, path[-1])))
        return samples

    def follow_reference(self, max_steps=460):
        x, y = map(float, self.s_start)
        theta = math.atan2(
            self.reference_path[1][1] - self.reference_path[0][1],
            self.reference_path[1][0] - self.reference_path[0][0],
        )
        integral = 0.0
        previous_error = 0.0
        closest_index = 0

        self.tracked_path = [(x, y)]
        self.control_history = []
        self.target_history = []

        for _ in range(max_steps):
            closest_index = self.closest_reference_index((x, y), closest_index)
            target_index = self.lookahead_index((x, y), closest_index)
            target = self.reference_path[target_index]
            goal_distance = self.distance((x, y), self.s_goal)

            desired_heading = math.atan2(target[1] - y, target[0] - x)
            error = wrap_pi(desired_heading - theta)
            integral = float(np.clip(integral + error * self.dt, -1.8, 1.8))
            derivative = (error - previous_error) / self.dt
            p_term = self.kp * error
            i_term = self.ki * integral
            d_term = self.kd * derivative
            omega = float(np.clip(p_term + i_term + d_term, -2.4, 2.4))
            velocity = min(self.speed, max(0.45, goal_distance * 0.55))

            theta = wrap_pi(theta + omega * self.dt)
            x += velocity * math.cos(theta) * self.dt
            y += velocity * math.sin(theta) * self.dt
            x = float(np.clip(x, 1.0, self.Env.x_range - 2.0))
            y = float(np.clip(y, 1.0, self.Env.y_range - 2.0))

            self.tracked_path.append((x, y))
            self.target_history.append(target)
            self.control_history.append(
                {
                    "error": error,
                    "p": p_term,
                    "i": i_term,
                    "d": d_term,
                    "omega": omega,
                    "theta": theta,
                }
            )

            previous_error = error
            if goal_distance < 0.55 and target_index >= len(self.reference_path) - 5:
                break

    def closest_reference_index(self, position, start_index):
        end_index = min(len(self.reference_path), start_index + 45)
        window = self.reference_path[start_index:end_index]
        if not window:
            return len(self.reference_path) - 1
        distances = [self.distance(position, point) for point in window]
        return start_index + int(np.argmin(distances))

    def lookahead_index(self, position, start_index):
        index = start_index
        while index < len(self.reference_path) - 1:
            if self.distance(position, self.reference_path[index]) >= self.lookahead:
                return index
            index += 1
        return len(self.reference_path) - 1

    def draw_pid_terms(self, upto):
        if not self.control_history:
            return
        terms = self.control_history[:upto]
        xs = np.arange(len(terms))
        p_values = [item["p"] for item in terms]
        i_values = [item["i"] for item in terms]
        d_values = [item["d"] for item in terms]
        ax = plt.gca().inset_axes([0.61, 0.07, 0.34, 0.25])
        ax.plot(xs, p_values, color="tab:red", linewidth=1.4, label="P")
        ax.plot(xs, i_values, color="tab:green", linewidth=1.4, label="I")
        ax.plot(xs, d_values, color="tab:purple", linewidth=1.4, label="D")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title("PID terms", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    def draw_vehicle(self, position, theta):
        heading = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-heading[1], heading[0]])
        center = np.array(position)
        nose = center + heading * 0.85
        left = center - heading * 0.55 + normal * 0.42
        right = center - heading * 0.55 - normal * 0.42
        polygon = np.vstack([nose, left, right, nose])
        plt.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=1.7)
        plt.fill(polygon[:, 0], polygon[:, 1], color="gold", alpha=0.9)

    def run_demonstration(self):
        print("Starting PID Controller path following demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("084 PID Controller - Reference Path", self.visited, self.raw_path)

        frame_indices = np.linspace(4, len(self.tracked_path) - 1, 26, dtype=int)
        for frame_index in frame_indices:
            history = self.tracked_path[: frame_index + 1]
            control_index = min(frame_index - 1, len(self.control_history) - 1)
            target = self.target_history[control_index]
            theta = self.control_history[control_index]["theta"]

            self.draw_base("084 PID Controller - Path Following")
            plt.plot(
                [p[0] for p in self.reference_path],
                [p[1] for p in self.reference_path],
                color="tab:blue",
                alpha=0.35,
                linewidth=2.2,
                label="reference",
            )
            plt.plot(
                [p[0] for p in history],
                [p[1] for p in history],
                color="crimson",
                linewidth=2.8,
                label="tracked",
            )
            plt.plot(target[0], target[1], "o", color="tab:orange", markersize=7, label="lookahead")
            plt.plot(
                [history[-1][0], target[0]],
                [history[-1][1], target[1]],
                "--",
                color="tab:orange",
                linewidth=1.2,
                alpha=0.75,
            )
            self.draw_vehicle(history[-1], theta)
            self.draw_pid_terms(control_index + 1)
            plt.legend(loc="upper left")
            self.capture_frame()

        self.save_gif("084_PID_Controller", fps=5)
        plt.close("all")
        print(f"Reference samples: {len(self.reference_path)}")
        print(f"Tracked samples: {len(self.tracked_path)}")
        print(f"Final tracking error: {self.distance(self.tracked_path[-1], self.s_goal):.3f}")


def main():
    PIDControllerPathFollower().run_demonstration()


if __name__ == "__main__":
    main()
