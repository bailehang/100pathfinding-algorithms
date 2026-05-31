"""Velocity Obstacle (VO) multi-agent avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class VelocityObstacleDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("vo", "087 Velocity Obstacle - head-on avoidance")


def main():
    demo = VelocityObstacleDemo()
    paths = demo.planning(save_gif=True, gif_name="087_velocity_obstacle")
    if not paths:
        raise RuntimeError("Velocity Obstacle demo returned no paths")


if __name__ == "__main__":
    main()
