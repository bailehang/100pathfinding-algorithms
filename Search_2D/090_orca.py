"""Optimal Reciprocal Collision Avoidance (ORCA) multi-agent avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class OrcaDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("orca", "090 ORCA - reciprocal constraint avoidance")


def main():
    demo = OrcaDemo()
    paths = demo.planning(save_gif=True, gif_name="090_orca")
    if not paths:
        raise RuntimeError("ORCA demo returned no paths")


if __name__ == "__main__":
    main()
