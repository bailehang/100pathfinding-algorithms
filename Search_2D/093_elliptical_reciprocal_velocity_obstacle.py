"""Elliptical Reciprocal Velocity Obstacle (ERVO/EORCA) avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class EllipticalReciprocalVelocityObstacleDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("ervo", "093 ERVO - elliptical reciprocal avoidance")


def main():
    demo = EllipticalReciprocalVelocityObstacleDemo()
    paths = demo.planning(save_gif=True, gif_name="093_elliptical_reciprocal_velocity_obstacle")
    if not paths:
        raise RuntimeError("ERVO demo returned no paths")


if __name__ == "__main__":
    main()
