"""Shared demos for the final pathfinding algorithms."""

from __future__ import annotations

import heapq
import io
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from Search_2D import env


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#f97316", "#9333ea", "#0891b2"]


class GifRecorder:
    def __init__(self, figsize=(7.0, 4.7), dpi=110):
        self.figsize = figsize
        self.dpi = dpi
        self.frames = []

    def capture(self):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("RGB"))
        buf.close()

    def save(self, name, fps=5, hold=4):
        if not self.frames:
            raise RuntimeError(f"No frames captured for {name}")
        if hold > 0:
            self.frames.extend([self.frames[-1]] * hold)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")
        self.frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        print(f"GIF animation saved to {gif_path}")


class GridTools:
    def __init__(self):
        self.Env = env.Env()
        self.obs = set(self.Env.obs)
        self.motions = self.Env.motions

    def is_valid(self, node):
        x, y = node
        return 0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range and node not in self.obs

    def is_collision(self, a, b):
        if a in self.obs or b in self.obs:
            return True
        if a[0] != b[0] and a[1] != b[1]:
            if b[0] - a[0] == a[1] - b[1]:
                s1 = (min(a[0], b[0]), min(a[1], b[1]))
                s2 = (max(a[0], b[0]), max(a[1], b[1]))
            else:
                s1 = (min(a[0], b[0]), max(a[1], b[1]))
                s2 = (max(a[0], b[0]), min(a[1], b[1]))
            return s1 in self.obs or s2 in self.obs
        return False

    def astar(self, start, goal, reservations=None, start_time=0, horizon=None):
        reservations = reservations or {}
        open_set = [(self.heuristic(start, goal), 0.0, start, start_time)]
        parent = {(start, start_time): None}
        cost = {(start, start_time): 0.0}
        best_key = (start, start_time)
        best_score = self.heuristic(start, goal)

        while open_set:
            _, g_score, current, time = heapq.heappop(open_set)
            key = (current, time)
            if g_score > cost.get(key, float("inf")):
                continue
            current_score = self.heuristic(current, goal)
            if current_score < best_score:
                best_score = current_score
                best_key = key
            if current == goal:
                return self.reconstruct_time_path(parent, key)
            if horizon is not None and time - start_time >= horizon:
                continue

            for dx, dy in self.motions + [(0, 0)]:
                nxt = (current[0] + dx, current[1] + dy)
                next_time = time + 1
                if not self.is_valid(nxt) or self.is_collision(current, nxt):
                    continue
                if nxt in reservations.get(next_time, set()):
                    continue
                if (nxt, current) in reservations.get(("edge", next_time), set()):
                    continue
                next_key = (nxt, next_time)
                step_cost = 1.55 if (dx, dy) == (0, 0) else math.hypot(dx, dy)
                next_cost = g_score + step_cost
                if next_cost < cost.get(next_key, float("inf")):
                    cost[next_key] = next_cost
                    parent[next_key] = key
                    priority = next_cost + self.heuristic(nxt, goal)
                    heapq.heappush(open_set, (priority, next_cost, nxt, next_time))

        return self.reconstruct_time_path(parent, best_key)

    @staticmethod
    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def reconstruct_time_path(parent, key):
        path = []
        while key is not None:
            path.append(key[0])
            key = parent[key]
        return list(reversed(path))

    def draw_grid(self, ax, title):
        obs_x = [p[0] for p in self.obs]
        obs_y = [p[1] for p in self.obs]
        ax.scatter(obs_x, obs_y, s=9, c="#111827", marker="s", linewidths=0)
        ax.set_xlim(0, self.Env.x_range)
        ax.set_ylim(0, self.Env.y_range)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_title(title)


def polyline_length(path):
    if len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


@dataclass
class SimpleAgent:
    start: tuple[float, float]
    goal: tuple[float, float]
    color: str
    radius: float = 0.55
