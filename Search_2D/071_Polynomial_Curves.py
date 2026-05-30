"""
Polynomial Curves 2D path planning demo.

The demo first finds a collision-free waypoint path on the shared grid map,
then converts the waypoints into a piecewise polynomial trajectory. Cubic
Hermite segments are used where they remain collision-free; otherwise the
segment falls back to a cubic smoothstep line between safe waypoints.
"""

from metrics import install_metrics
install_metrics()

import heapq
import io
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D import env


class PolynomialCurves:
    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.Env = env.Env()
        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.frames = []

        self.raw_path = []
        self.visited = []
        self.waypoints = []
        self.polynomial_path = []

    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.polynomial_path = self.build_polynomial_path(self.waypoints)
        return self.polynomial_path, self.visited

    def a_star_search(self):
        open_set = []
        heapq.heappush(open_set, (0.0, self.s_start))
        parent = {self.s_start: self.s_start}
        g = {self.s_start: 0.0}
        visited = []

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
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
        anchor_index = 0

        while anchor_index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > anchor_index + 1:
                if self.line_of_sight(path[anchor_index], path[next_index]):
                    break
                next_index -= 1

            waypoints.append(path[next_index])
            anchor_index = next_index

        return waypoints

    def build_polynomial_path(self, waypoints):
        if len(waypoints) < 2:
            return waypoints

        curve = []
        for i in range(len(waypoints) - 1):
            p0 = np.array(waypoints[i], dtype=float)
            p1 = np.array(waypoints[i + 1], dtype=float)

            prev_point = np.array(waypoints[max(i - 1, 0)], dtype=float)
            next_point = np.array(waypoints[min(i + 2, len(waypoints) - 1)], dtype=float)
            tangent0 = 0.35 * (p1 - prev_point)
            tangent1 = 0.35 * (next_point - p0)

            segment = self.sample_hermite_segment(p0, p1, tangent0, tangent1)
            if self.segment_samples_collide(segment):
                segment = self.sample_smoothstep_segment(p0, p1)

            if curve:
                segment = segment[1:]
            curve.extend(segment)

        return [(float(x), float(y)) for x, y in curve]

    def sample_hermite_segment(self, p0, p1, tangent0, tangent1, samples=24):
        points = []
        for t in np.linspace(0.0, 1.0, samples):
            h00 = 2 * t ** 3 - 3 * t ** 2 + 1
            h10 = t ** 3 - 2 * t ** 2 + t
            h01 = -2 * t ** 3 + 3 * t ** 2
            h11 = t ** 3 - t ** 2
            point = h00 * p0 + h10 * tangent0 + h01 * p1 + h11 * tangent1
            points.append(tuple(point))
        return points

    def sample_smoothstep_segment(self, p0, p1, samples=24):
        points = []
        for t in np.linspace(0.0, 1.0, samples):
            s = 3 * t ** 2 - 2 * t ** 3
            point = p0 + (p1 - p0) * s
            points.append(tuple(point))
        return points

    def segment_samples_collide(self, samples):
        for point in samples:
            rounded = (int(round(point[0])), int(round(point[1])))
            if not self.is_valid(rounded):
                return True

        for i in range(len(samples) - 1):
            a = (int(round(samples[i][0])), int(round(samples[i][1])))
            b = (int(round(samples[i + 1][0])), int(round(samples[i + 1][1])))
            if self.is_collision(a, b):
                return True
        return False

    def is_valid(self, node):
        return (
            0 <= node[0] < self.Env.x_range
            and 0 <= node[1] < self.Env.y_range
            and node not in self.obs
        )

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

    def line_of_sight(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        current = (x0, y0)

        while True:
            if current in self.obs:
                return False
            if current == end:
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

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def polynomial_path_length(self):
        if len(self.polynomial_path) < 2:
            return 0.0
        return sum(
            self.distance(self.polynomial_path[i], self.polynomial_path[i + 1])
            for i in range(len(self.polynomial_path) - 1)
        )

    def draw_base(self, title):
        plt.cla()
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]
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
        frame = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        self.frames.append(frame)

    def save_gif(self, name, fps=4):
        if not self.frames:
            print("No frames captured; GIF was not saved.")
            return

        gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

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

    def run_demonstration(self):
        print("Starting Polynomial Curves demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)

        self.draw_base("071 Polynomial Curves - Grid Search")
        if self.visited:
            plt.scatter(
                [p[0] for p in self.visited],
                [p[1] for p in self.visited],
                s=8,
                c="lightgray",
                alpha=0.8,
                label="A* visited",
            )
        self.capture_frame()

        self.draw_base("071 Polynomial Curves - A* Waypoint Path")
        if self.raw_path:
            plt.plot(
                [p[0] for p in self.raw_path],
                [p[1] for p in self.raw_path],
                color="tab:blue",
                linewidth=2,
                label="Discrete A* path",
            )
        self.capture_frame()

        self.draw_base("071 Polynomial Curves - Safe Waypoints")
        if self.raw_path:
            plt.plot(
                [p[0] for p in self.raw_path],
                [p[1] for p in self.raw_path],
                color="tab:blue",
                alpha=0.35,
                linewidth=2,
            )
        plt.plot(
            [p[0] for p in self.waypoints],
            [p[1] for p in self.waypoints],
            "o--",
            color="tab:orange",
            linewidth=2,
            label="Polynomial control waypoints",
        )
        self.capture_frame()

        frame_count = 12
        for i in range(1, frame_count + 1):
            upto = max(2, int(len(self.polynomial_path) * i / frame_count))
            visible_curve = self.polynomial_path[:upto]
            self.draw_base("071 Polynomial Curves - Cubic Trajectory")
            plt.plot(
                [p[0] for p in self.waypoints],
                [p[1] for p in self.waypoints],
                "o--",
                color="tab:orange",
                alpha=0.45,
                linewidth=1.5,
            )
            plt.plot(
                [p[0] for p in visible_curve],
                [p[1] for p in visible_curve],
                color="crimson",
                linewidth=3,
                label="Polynomial curve",
            )
            self.capture_frame()

        self.save_gif("071_Polynomial_Curves", fps=4)
        plt.close("all")

        print(f"Raw A* nodes: {len(self.raw_path)}")
        print(f"Control waypoints: {len(self.waypoints)}")
        print(f"Polynomial samples: {len(self.polynomial_path)}")
        print(f"Polynomial path length: {self.polynomial_path_length():.3f}")


def main():
    planner = PolynomialCurves((5, 5), (45, 25))
    planner.run_demonstration()


if __name__ == "__main__":
    main()
