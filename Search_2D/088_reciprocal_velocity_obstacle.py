"""Reciprocal Velocity Obstacle (RVO) multi-agent avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class ReciprocalVelocityObstacleDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("rvo", "088 Reciprocal Velocity Obstacle - shared avoidance")


def main():
    demo = ReciprocalVelocityObstacleDemo()
    paths = demo.planning(save_gif=True, gif_name="088_reciprocal_velocity_obstacle")
    if not paths:
        raise RuntimeError("RVO demo returned no paths")


if __name__ == "__main__":
    main()
