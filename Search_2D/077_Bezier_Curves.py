"""
Bezier Curves 2D path planning demo.

The demo finds a collision-free waypoint path on the shared grid map, simplifies
it with line-of-sight checks, then fits cubic Bezier segments through the safe
waypoints. Curved handles are shortened automatically if a segment would cross
an obstacle.
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


class BezierCurves:
    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.Env = env.Env()
        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        self.raw_path = []
        self.visited = []
        self.waypoints = []
        self.bezier_path = []
        self.control_polygons = []
        self.frames = []

    def plan(self):
        self.raw_path, self.visited = self.a_star_search()
        self.waypoints = self.extract_safe_waypoints(self.raw_path)
        self.bezier_path, self.control_polygons = self.build_bezier_path(self.waypoints)
        return self.bezier_path, self.visited

    def a_star_search(self):
        open_set = []
        heapq.heappush(open_set, (0.0, self.s_start))
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

    def build_bezier_path(self, waypoints):
        if len(waypoints) < 2:
            return waypoints, []

        curve = []
        control_polygons = []
        points = [np.array(p, dtype=float) for p in waypoints]

        for i in range(len(points) - 1):
            p0 = points[i]
            p3 = points[i + 1]
            previous_point = points[max(i - 1, 0)]
            next_point = points[min(i + 2, len(points) - 1)]

            tangent0 = self.normalized(p3 - previous_point)
            tangent1 = self.normalized(next_point - p0)
            segment_length = self.distance(p0, p3)

            best_segment = None
            best_controls = None
            for handle_scale in (0.35, 0.22, 0.12):
                handle_length = segment_length * handle_scale
                p1 = p0 + tangent0 * handle_length
                p2 = p3 - tangent1 * handle_length
                candidate = self.sample_bezier(p0, p1, p2, p3)
                if not self.segment_samples_collide(candidate):
                    best_segment = candidate
                    best_controls = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]
                    break

            if best_segment is None:
                p1 = p0 + (p3 - p0) / 3.0
                p2 = p0 + 2.0 * (p3 - p0) / 3.0
                best_segment = self.sample_bezier(p0, p1, p2, p3)
                best_controls = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

            if curve:
                best_segment = best_segment[1:]
            curve.extend(best_segment)
            control_polygons.append(best_controls)

        return [(float(x), float(y)) for x, y in curve], control_polygons

    def sample_bezier(self, p0, p1, p2, p3, samples=28):
        points = []
        for t in np.linspace(0.0, 1.0, samples):
            point = (
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * t ** 2 * p2
                + t ** 3 * p3
            )
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

    @staticmethod
    def normalized(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.array([0.0, 0.0])
        return vector / norm

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

    def bezier_path_length(self):
        if len(self.bezier_path) < 2:
            return 0.0
        return sum(
            self.distance(self.bezier_path[i], self.bezier_path[i + 1])
            for i in range(len(self.bezier_path) - 1)
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
        print("Starting Bezier Curves demonstration...")
        self.plan()

        plt.figure(figsize=(7, 5), dpi=100)

        self.draw_base("072 Bezier Curves - Grid Search")
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

        self.draw_base("072 Bezier Curves - Safe Waypoints")
        plt.plot(
            [p[0] for p in self.raw_path],
            [p[1] for p in self.raw_path],
            color="tab:blue",
            alpha=0.35,
            linewidth=2,
            label="A* path",
        )
        plt.plot(
            [p[0] for p in self.waypoints],
            [p[1] for p in self.waypoints],
            "o--",
            color="tab:orange",
            linewidth=2,
            label="Bezier anchors",
        )
        self.capture_frame()

        self.draw_base("072 Bezier Curves - Control Polygons")
        for controls in self.control_polygons:
            plt.plot(
                [p[0] for p in controls],
                [p[1] for p in controls],
                "o--",
                color="tab:purple",
                alpha=0.55,
                linewidth=1.5,
            )
        self.capture_frame()

        frame_count = 12
        for i in range(1, frame_count + 1):
            upto = max(2, int(len(self.bezier_path) * i / frame_count))
            visible_curve = self.bezier_path[:upto]
            self.draw_base("072 Bezier Curves - Cubic Bezier Trajectory")
            for controls in self.control_polygons:
                plt.plot(
                    [p[0] for p in controls],
                    [p[1] for p in controls],
                    "--",
                    color="tab:purple",
                    alpha=0.25,
                    linewidth=1,
                )
            plt.plot(
                [p[0] for p in visible_curve],
                [p[1] for p in visible_curve],
                color="crimson",
                linewidth=3,
                label="Bezier curve",
            )
            self.capture_frame()

        self.save_gif("077_Bezier_Curves", fps=4)
        plt.close("all")

        print(f"Raw A* nodes: {len(self.raw_path)}")
        print(f"Bezier anchors: {len(self.waypoints)}")
        print(f"Bezier samples: {len(self.bezier_path)}")
        print(f"Bezier path length: {self.bezier_path_length():.3f}")


def main():
    planner = BezierCurves((5, 5), (45, 25))
    planner.run_demonstration()


if __name__ == "__main__":
    main()
