"""
A_star 2D  --  reference implementation for the shared/reusable layout.

@author: huiming zhou
@author: clark bai

This demo is the worked example for doc/CODE_REVIEW_reusability.md:

* the 51x31 world comes from ``common.env.Env`` (no inline copy);
* visualisation + GIF capture come from ``common.plotting.GifPlotter``
  (no inline ``Plotting`` / ``capture_frame`` / ``save_animation_as_gif``);
* timing/length use the explicit ``benchmarks.metrics.measure`` API instead of
  the global monkeypatch bootstrap.

It remains a standalone runnable script: ``python 008_Astar.py``.
"""

import os
import sys

# Make the shared packages importable when run as a standalone script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import heapq
import math

from common.env import Env
from common.plotting import GifPlotter
from benchmarks.metrics import measure


class AStar:
    """A* search: priority = g (cost-to-come) + h (heuristic)."""

    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.env = Env()
        self.u_set = self.env.motions      # feasible moves
        self.obs = self.env.obs            # obstacle cells

        self.OPEN = []                     # priority queue
        self.CLOSED = []                   # visited order (for plotting)
        self.PARENT = dict()
        self.g = dict()

    def searching(self):
        """Run A* and return (path, visited_order)."""
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        closed_set = set()  # O(1) membership: skip already-expanded nodes
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            if s in closed_set:            # lazy deletion: ignore stale entries
                continue
            closed_set.add(s)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                if s_n in closed_set:
                    continue
                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def get_neighbor(self, s):
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        if self.is_collision(s_start, s_goal):
            return math.inf
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        path = [self.s_goal]
        s = self.s_goal
        while s != self.s_start:
            s = PARENT[s]
            path.append(s)
        return list(path)

    def heuristic(self, s):
        goal = self.s_goal
        if self.heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    astar = AStar(s_start, s_goal, "euclidean")
    plot = GifPlotter(s_start, s_goal)

    with measure() as m:
        path, visited = astar.searching()
    m.record(path, expanded=len(visited))
    print(f"[Metrics] {m.line()}")

    plot.animation(path, visited, "008_Astar", save_gif=True)


if __name__ == "__main__":
    main()
