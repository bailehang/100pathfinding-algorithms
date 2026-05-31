"""Model Predictive Control (MPC) path following demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.model_control_helpers import ReferencePathFollower


class MPCPathFollower(ReferencePathFollower):
    def __init__(self, s_start=(5, 5), s_goal=(45, 25)):
        super().__init__(s_start, s_goal)
        self.frame_title = "086 MPC - Receding Horizon Path Following"
        self.dt = 0.18
        self.speed = 1.65
        self.lookahead = 2.5
        self.horizon = 12
        self.turn_rates = np.linspace(-1.65, 1.65, 9)
        self.prediction_history = []

    def plan(self):
        start_ms = now_ms()
        self.build_reference()
        self.follow_reference()
        elapsed = now_ms() - start_ms
        print_metrics_for(self.tracked_path, elapsed, source="mpc")
        return self.tracked_path, self.visited

    def follow_reference(self, max_steps=420):
        state = (
            float(self.s_start[0]),
            float(self.s_start[1]),
            self.reference_headings[1],
        )
        closest_index = 0
        previous_omega = 0.0
        self.tracked_path = [(state[0], state[1])]
        self.target_history = []
        self.control_history = []
        self.prediction_history = []

        for _ in range(max_steps):
            position = (state[0], state[1])
            closest_index = self.closest_reference_index(position, closest_index)
            omega, prediction, best_cost, target = self.choose_control(state, closest_index, previous_omega)
            goal_distance = self.distance(position, self.s_goal)
            speed = min(self.speed, max(0.40, goal_distance * 0.56))
            state = self.propagate(state, omega, speed=speed)

            self.tracked_path.append((state[0], state[1]))
            self.target_history.append(target)
            self.prediction_history.append(prediction)
            self.control_history.append(
                {
                    "omega": omega,
                    "cost": best_cost,
                    "theta": state[2],
                    "closest_index": closest_index,
                }
            )
            previous_omega = omega

            target_index = self.closest_reference_index((state[0], state[1]), closest_index)
            if goal_distance < 0.55 and target_index >= len(self.reference_path) - 5:
                break

    def choose_control(self, state, closest_index, previous_omega):
        best_cost = float("inf")
        best_omega = 0.0
        best_prediction = []
        best_target = self.reference_path[closest_index]

        for first_omega in self.turn_rates:
            for second_omega in self.turn_rates[::2]:
                controls = [first_omega] * (self.horizon // 2)
                controls.extend([second_omega] * (self.horizon - len(controls)))
                prediction, cost, target = self.rollout_cost(state, closest_index, controls, previous_omega)
                if cost < best_cost:
                    best_cost = cost
                    best_omega = float(first_omega)
                    best_prediction = prediction
                    best_target = target

        return best_omega, best_prediction, best_cost, best_target

    def rollout_cost(self, state, closest_index, controls, previous_omega):
        sim_state = state
        sim_index = closest_index
        prediction = [(state[0], state[1])]
        cost = 0.0
        last_omega = previous_omega
        target = self.reference_path[closest_index]

        for step, omega in enumerate(controls, start=1):
            sim_state = self.propagate(sim_state, omega)
            position = (sim_state[0], sim_state[1])
            sim_index = self.closest_reference_index(position, sim_index)
            target_index, target, _, lateral_error, heading_error = self.reference_state(
                position, sim_state[2], sim_index
            )
            progress_reward = 0.035 * target_index
            cost += 4.0 * lateral_error**2
            cost += 1.4 * heading_error**2
            cost += 0.12 * omega**2
            cost += 0.16 * (omega - last_omega) ** 2
            cost -= progress_reward
            if self.samples_collide([prediction[-1], position]):
                cost += 500.0 + 30.0 * step
            prediction.append(position)
            last_omega = omega

        terminal_distance = self.distance(prediction[-1], self.s_goal)
        cost += 0.38 * terminal_distance
        return prediction, cost, target

    def draw_mpc_costs(self, upto):
        terms = self.control_history[:upto]
        if not terms:
            return
        xs = np.arange(len(terms))
        omegas = [item["omega"] for item in terms]
        costs = [min(item["cost"], 80.0) / 18.0 for item in terms]
        ax = plt.gca().inset_axes([0.60, 0.07, 0.35, 0.26])
        ax.plot(xs, omegas, color="tab:purple", linewidth=1.35, label="chosen omega")
        ax.plot(xs, costs, color="tab:green", linewidth=1.35, label="cost / 18")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"MPC horizon={self.horizon}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    def run_demonstration(self):
        print("Starting MPC path following demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)
        self.draw_search_frame("086 MPC - Reference Path", self.visited, self.raw_path)

        frame_indices = np.linspace(4, len(self.tracked_path) - 1, 27, dtype=int)
        for frame_index in frame_indices:
            history = self.tracked_path[: frame_index + 1]
            control_index = min(frame_index - 1, len(self.control_history) - 1)
            target = self.target_history[control_index]
            theta = self.control_history[control_index]["theta"]
            prediction = self.prediction_history[control_index]
            self.draw_reference_and_trace(
                history,
                target=target,
                prediction=prediction,
                trace_color="crimson",
            )
            self.draw_vehicle(history[-1], theta, color="lightskyblue")
            self.draw_mpc_costs(control_index + 1)
            plt.legend(loc="upper left")
            self.capture_frame()

        self.save_gif("086_MPC", fps=5)
        plt.close("all")
        print(f"MPC horizon: {self.horizon}")
        print(f"Tracked samples: {len(self.tracked_path)}")
        print(f"Final tracking error: {self.distance(self.tracked_path[-1], self.s_goal):.3f}")


def main():
    MPCPathFollower().run_demonstration()


if __name__ == "__main__":
    main()
