"""Linear Quadratic Regulator (LQR) path following demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.model_control_helpers import ReferencePathFollower, solve_discrete_lqr


class LQRPathFollower(ReferencePathFollower):
    def __init__(self, s_start=(5, 5), s_goal=(45, 25)):
        super().__init__(s_start, s_goal)
        self.frame_title = "085 LQR - Optimal Feedback Path Following"
        a_matrix = np.array([[1.0, self.speed * self.dt], [0.0, 1.0]])
        b_matrix = np.array([[0.0], [self.dt]])
        q_matrix = np.diag([4.8, 2.2])
        r_matrix = np.array([[0.62]])
        self.gain = solve_discrete_lqr(a_matrix, b_matrix, q_matrix, r_matrix)

    def plan(self):
        start_ms = now_ms()
        self.build_reference()
        self.follow_reference()
        elapsed = now_ms() - start_ms
        print_metrics_for(self.tracked_path, elapsed, source="lqr")
        return self.tracked_path, self.visited

    def follow_reference(self, max_steps=430):
        state = (
            float(self.s_start[0]),
            float(self.s_start[1]),
            self.reference_headings[1],
        )
        closest_index = 0
        self.tracked_path = [(state[0], state[1])]
        self.target_history = []
        self.control_history = []

        for _ in range(max_steps):
            position = (state[0], state[1])
            closest_index = self.closest_reference_index(position, closest_index)
            target_index, target, _, lateral_error, heading_error = self.reference_state(
                position, state[2], closest_index
            )
            error_state = np.array([[lateral_error], [heading_error]])
            omega = float(np.clip(-(self.gain @ error_state)[0, 0], -2.25, 2.25))
            goal_distance = self.distance(position, self.s_goal)
            speed = min(self.speed, max(0.45, goal_distance * 0.58))
            state = self.propagate(state, omega, speed=speed)

            self.tracked_path.append((state[0], state[1]))
            self.target_history.append(target)
            self.control_history.append(
                {
                    "lateral_error": lateral_error,
                    "heading_error": heading_error,
                    "omega": omega,
                    "target_index": target_index,
                    "theta": state[2],
                }
            )

            if goal_distance < 0.55 and target_index >= len(self.reference_path) - 5:
                break

    def draw_error_feedback(self, upto):
        terms = self.control_history[:upto]
        if not terms:
            return
        xs = np.arange(len(terms))
        lateral = [item["lateral_error"] for item in terms]
        heading = [item["heading_error"] for item in terms]
        omega = [item["omega"] for item in terms]
        ax = plt.gca().inset_axes([0.60, 0.07, 0.35, 0.26])
        ax.plot(xs, lateral, color="tab:red", linewidth=1.35, label="lat err")
        ax.plot(xs, heading, color="tab:green", linewidth=1.35, label="head err")
        ax.plot(xs, omega, color="tab:purple", linewidth=1.35, label="omega")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"LQR K=[{self.gain[0, 0]:.2f}, {self.gain[0, 1]:.2f}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    def run_demonstration(self):
        print("Starting LQR path following demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("085 LQR - Reference Path", self.visited, self.raw_path)

        frame_indices = np.linspace(4, len(self.tracked_path) - 1, 27, dtype=int)
        for frame_index in frame_indices:
            history = self.tracked_path[: frame_index + 1]
            control_index = min(frame_index - 1, len(self.control_history) - 1)
            target = self.target_history[control_index]
            theta = self.control_history[control_index]["theta"]
            self.draw_reference_and_trace(history, target=target, trace_color="crimson")
            self.draw_vehicle(history[-1], theta, color="gold")
            self.draw_error_feedback(control_index + 1)
            plt.legend(loc="upper left")
            self.capture_frame()

        self.save_gif("085_LQR", fps=5)
        plt.close("all")
        print(f"LQR gain: [{self.gain[0, 0]:.3f}, {self.gain[0, 1]:.3f}]")
        print(f"Tracked samples: {len(self.tracked_path)}")
        print(f"Final tracking error: {self.distance(self.tracked_path[-1], self.s_goal):.3f}")


def main():
    LQRPathFollower().run_demonstration()


if __name__ == "__main__":
    main()
