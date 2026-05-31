"""Pedestrian ORCA (PORCA) multi-agent avoidance demo."""

from metrics import install_metrics
install_metrics()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_avoidance_helpers import VelocityObstacleComparisonDemo


class PedestrianOrcaDemo(VelocityObstacleComparisonDemo):
    def __init__(self):
        super().__init__("porca", "091 Pedestrian ORCA - lane-like passing")


def main():
    demo = PedestrianOrcaDemo()
    paths = demo.planning(save_gif=True, gif_name="091_porca")
    if not paths:
        raise RuntimeError("PORCA demo returned no paths")


if __name__ == "__main__":
    main()
