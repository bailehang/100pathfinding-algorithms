"""UE5-style MassAI avoidance demo."""

from metrics import install_metrics

install_metrics()

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class UE5AIAvoidanceDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("ue5", "097 UE5 AI Avoidance - MassAI style steering")
        self.dt = 0.34
        self.pref_speed = 1.10
        self.max_steps = 210

    def score_velocity(self, agent_index, velocity, positions, velocities, desired):
        agent = self.agents[agent_index]
        predicted = positions[agent_index] + velocity * self.dt
        score = 1.35 * np.linalg.norm(velocity - desired)
        score += 0.030 * np.linalg.norm(agent.goal - predicted)
        score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.48, horizon=3.2, ellipse=False) * 0.92
        score += self.obstacle_penalty(agent, predicted) * 1.15
        score += self.abrupt_change_penalty(agent, velocity, 0.32)
        score += self.side_lane_penalty(agent, velocity, strength=0.28)
        score += self.boundary_penalty(predicted)
        return float(score)

    def desired_velocity(self, agent, step):
        desired = super().desired_velocity(agent, step)
        separation = np.zeros(2, dtype=float)
        for other in self.agents:
            if other.id == agent.id:
                continue
            delta = agent.pos - other.pos
            dist = np.linalg.norm(delta)
            if 1e-6 < dist < self.radius * 4.2:
                separation += delta / dist * ((self.radius * 4.2 - dist) / (self.radius * 4.2))
        if np.linalg.norm(separation) > 1e-6:
            desired = desired + separation * 0.55
        speed = np.linalg.norm(desired)
        if speed < 1e-6:
            return desired
        return desired / speed * min(speed, self.pref_speed * 1.05)

    def phase_text(self, step):
        if step < 45:
            return "UE5 MassAI: desired velocity blends separation and obstacle steering"
        if step < 100:
            return "UE5 MassAI: smooth velocity changes reduce jitter in the crossing flow"
        return "UE5 MassAI: local avoidance keeps agents separated while they swap sides"


def main():
    demo = UE5AIAvoidanceDemo()
    paths = demo.planning(save_gif=True, gif_name="097_UE5_AI_Avoidance")
    if not paths:
        raise RuntimeError("UE5 AI Avoidance demo returned no paths")


if __name__ == "__main__":
    main()
