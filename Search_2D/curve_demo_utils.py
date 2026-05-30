"""Shared helpers for curve and kinematic planning demos."""

from __future__ import annotations

import heapq
import io
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Search_2D import env


class GridCurveDemo:
    def __init__(self, s_start=(5, 5), s_goal=(45, 25)):
        self.s_start = s_start
        self.s_goal = s_goal
        self.Env = env.Env()
        self.obs = self.Env.obs
        self.u_set = self.Env.motions
        self.frames = []

    def a_star_search(self):
        open_set = [(0.0, self.s_start)]
        parent = {self.s_start: self.s_start}
        g = {self.s_start: 0.0}
        closed = set()
        visited = []

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
            visited.append(current)

            if current == self.s_goal:
                return self.extract_path(parent), visited

            for motion in self.u_set:
                neighbor = (current[0] + motion[0], current[1] + motion[1])
                if not self.is_valid(neighbor) or self.is_collision(current, neighbor):
                    continue

                new_cost = g[current] + self.distance(current, neighbor)
                if neighbor not in g or new_cost < g[neighbor]:
                    g[neighbor] = new_cost
                    parent[neighbor] = current
                    priority = new_cost + self.distance(neighbor, self.s_goal)
                    heapq.heappush(open_set, (priority, neighbor))

        return [self.s_start], visited

    def extract_path(self, parent):
        path = [self.s_goal]
        current = self.s_goal
        while current != self.s_start:
            current = parent[current]
            path.append(current)
        return list(reversed(path))

    def extract_safe_waypoints(self, path):
        if len(path) <= 2:
            return path

        waypoints = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            nxt = len(path) - 1
            while nxt > anchor + 1:
                if self.line_of_sight(path[anchor], path[nxt]):
                    break
                nxt -= 1
            waypoints.append(path[nxt])
            anchor = nxt
        return waypoints

    def is_valid(self, node):
        x, y = int(round(node[0])), int(round(node[1]))
        return 0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range and (x, y) not in self.obs

    def is_collision(self, s_start, s_end):
        a = (int(round(s_start[0])), int(round(s_start[1])))
        b = (int(round(s_end[0])), int(round(s_end[1])))
        if a in self.obs or b in self.obs:
            return True

        if a[0] != b[0] and a[1] != b[1]:
            if b[0] - a[0] == a[1] - b[1]:
                s1 = (min(a[0], b[0]), min(a[1], b[1]))
                s2 = (max(a[0], b[0]), max(a[1], b[1]))
            else:
                s1 = (min(a[0], b[0]), max(a[1], b[1]))
                s2 = (max(a[0], b[0]), min(a[1], b[1]))
            if s1 in self.obs or s2 in self.obs:
                return True
        return False

    def line_of_sight(self, start, end):
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(end[0])), int(round(end[1]))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        current = (x0, y0)

        while True:
            if current in self.obs:
                return False
            if current == (x1, y1):
                return True

            e2 = 2 * err
            next_x, next_y = current
            if e2 > -dy:
                err -= dy
                next_x += sx
            if e2 < dx:
                err += dx
                next_y += sy
            nxt = (next_x, next_y)
            if self.is_collision(current, nxt):
                return False
            current = nxt

    def samples_collide(self, samples):
        for point in samples:
            if not self.is_valid(point):
                return True
        for i in range(len(samples) - 1):
            if self.is_collision(samples[i], samples[i + 1]):
                return True
        return False

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def path_length(path):
        if len(path) < 2:
            return 0.0
        return sum(GridCurveDemo.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def draw_base(self, title):
        plt.cla()
        obs_x = [p[0] for p in self.obs]
        obs_y = [p[1] for p in self.obs]
        plt.plot(obs_x, obs_y, "sk", markersize=4)
        plt.plot(self.s_start[0], self.s_start[1], "bs", label="Start")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs", label="Goal")
        plt.title(title)
        plt.xlim(0, self.Env.x_range)
        plt.ylim(0, self.Env.y_range)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.25)

    def capture_frame(self):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        self.frames.append(np.array(Image.open(buf).convert("RGB")))
        buf.close()

    def save_gif(self, name, fps=4):
        if not self.frames:
            print("No frames captured; GIF was not saved.")
            return

        gif_dir = Path(__file__).resolve().parent / "gif"
        gif_dir.mkdir(exist_ok=True)
        gif_path = gif_dir / f"{name}.gif"
        palette_frames = [
            Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256)
            for frame in self.frames
        ]
        palette_frames[0].save(
            gif_path,
            format="GIF",
            append_images=palette_frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        print(f"GIF animation saved to {gif_path}")

    def draw_search_frame(self, title, visited, raw_path, waypoints=None):
        self.draw_base(title)
        if visited:
            plt.scatter(
                [p[0] for p in visited],
                [p[1] for p in visited],
                s=8,
                c="lightgray",
                alpha=0.8,
                label="A* visited",
            )
        if raw_path:
            plt.plot(
                [p[0] for p in raw_path],
                [p[1] for p in raw_path],
                color="tab:blue",
                linewidth=2,
                alpha=0.35,
                label="A* path",
            )
        if waypoints:
            plt.plot(
                [p[0] for p in waypoints],
                [p[1] for p in waypoints],
                "o--",
                color="tab:orange",
                linewidth=2,
                label="safe waypoints",
            )
        self.capture_frame()


def lerp_path(a, b, samples=24):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return [tuple(a + (b - a) * t) for t in np.linspace(0.0, 1.0, samples)]


def normalized(vector):
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.0, 0.0])
    return vector / norm


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mod2pi(angle):
    return angle % (2.0 * math.pi)
