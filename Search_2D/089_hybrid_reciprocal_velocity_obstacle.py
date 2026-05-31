"""Hybrid Reciprocal Velocity Obstacle (HRVO) multi-agent avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class HybridReciprocalVelocityObstacleDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("hrvo", "089 HRVO - hybrid reciprocal side choice")


def main():
    demo = HybridReciprocalVelocityObstacleDemo()
    paths = demo.planning(save_gif=True, gif_name="089_hybrid_reciprocal_velocity_obstacle")
    if not paths:
        raise RuntimeError("HRVO demo returned no paths")


if __name__ == "__main__":
    main()
