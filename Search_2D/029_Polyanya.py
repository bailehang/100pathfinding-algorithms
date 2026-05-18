"""
Polyanya Any-Angle Pathfinding Demo
@author: clark bai
@author: assistant

This demo uses the repository's grid map as a simple square navigation mesh.
Each free grid cell is a convex polygon and each shared edge is a portal. The
search state follows Polyanya's core idea: a root point plus an interval on a
mesh edge that is visible from that root.
"""

import heapq
import io
import math
import os
from dataclasses import dataclass, field


EPS = 1e-9


class Env:
    """Environment class for 2D grid world."""

    def __init__(self):
        self.x_range = 51
        self.y_range = 31
        self.motions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """Initialize obstacles' positions."""

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs


@dataclass
class PolyanyaState:
    """A Polyanya interval state on a square-cell navigation mesh."""

    root: tuple[float, float]
    interval: tuple[tuple[float, float], tuple[float, float]]
    cell: tuple[int, int]
    g_root: float
    path_roots: list[tuple[float, float]] = field(default_factory=list)


class Polyanya:
    """Polyanya-style interval search on a grid-derived navigation mesh."""

    def __init__(self, s_start, s_goal):
        self.s_start = (float(s_start[0]), float(s_start[1]))
        self.s_goal = (float(s_goal[0]), float(s_goal[1]))
        self.Env = Env()
        self.obs = self.Env.obs
        self.OPEN = []
        self.CLOSED = []
        self.visited_intervals = []
        self.generated_roots = []
        self.counter = 0
        self.best_cost = math.inf
        self.best_path = []
        self.los_cache = {}

    def searching(self):
        """Run Polyanya interval search and return the path plus visited states."""

        start_cell = self.point_to_cell(self.s_start)
        if start_cell in self.obs:
            raise ValueError("Start point lies inside an obstacle.")

        start_state = PolyanyaState(
            root=self.s_start,
            interval=self.cell_bounds(start_cell)[0],
            cell=start_cell,
            g_root=0.0,
            path_roots=[self.s_start],
        )

        best = {}
        self.push_state(start_state, best)

        while self.OPEN:
            priority, _, state = heapq.heappop(self.OPEN)
            if priority >= self.best_cost:
                break

            state_key = self.state_key(state)
            if best.get(state_key, math.inf) + EPS < state.g_root:
                continue

            self.CLOSED.append(state)
            self.visited_intervals.append(state.interval)
            self.generated_roots.append(state.root)

            if self.line_of_sight(state.root, self.s_goal):
                cost = state.g_root + self.distance(state.root, self.s_goal)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = self.compact_path(state.path_roots + [self.s_goal])

            if state.g_root >= self.best_cost:
                continue

            for nxt in self.expand_state(state):
                if nxt.g_root + self.heuristic_interval(nxt) >= self.best_cost:
                    continue
                self.push_state(nxt, best)

        return self.best_path, self.CLOSED

    def push_state(self, state, best):
        key = self.state_key(state)
        if state.g_root + EPS >= best.get(key, math.inf):
            return

        best[key] = state.g_root
        self.counter += 1
        priority = state.g_root + self.heuristic_interval(state)
        heapq.heappush(self.OPEN, (priority, self.counter, state))

    def expand_state(self, state):
        successors = []
        cx, cy = state.cell

        for dx, dy in self.Env.motions:
            next_cell = (cx + dx, cy + dy)
            if not self.is_free(next_cell):
                continue

            exit_interval = self.portal_between(state.cell, next_cell)
            if self.point_in_cell(state.root, state.cell):
                clipped = exit_interval
            else:
                clipped = self.project_interval(state.root, state.interval, exit_interval)

            if clipped is not None and self.interval_length(clipped) > 1e-6:
                successors.append(
                    PolyanyaState(
                        root=state.root,
                        interval=clipped,
                        cell=next_cell,
                        g_root=state.g_root,
                        path_roots=state.path_roots,
                    )
                )
                continue

            for corner in state.interval:
                if not self.valid_turn_point(corner):
                    continue
                if not self.segment_inside_cell(corner, self.interval_midpoint(exit_interval), state.cell):
                    continue

                new_roots = self.compact_path(state.path_roots + [corner])
                successors.append(
                    PolyanyaState(
                        root=corner,
                        interval=exit_interval,
                        cell=next_cell,
                        g_root=state.g_root + self.distance(state.root, corner),
                        path_roots=new_roots,
                    )
                )

        return successors

    def project_interval(self, root, entry_interval, exit_interval):
        """Project the visible cone through entry_interval onto exit_interval."""

        if self.point_in_cell(root, self.point_to_cell(root)):
            if self.point_to_cell(root) == self.point_to_cell(self.interval_midpoint(exit_interval)):
                return exit_interval

        a, b = entry_interval
        e0, e1 = exit_interval

        if self.point_in_cell(root, self.point_to_cell(self.interval_midpoint(exit_interval))):
            return exit_interval

        ts = []

        for t, point in ((0.0, e0), (1.0, e1)):
            if self.in_cone(root, a, b, point):
                ts.append(t)

        for ray_point in (a, b):
            hit = self.line_segment_intersection_parameter(root, ray_point, e0, e1)
            if hit is not None:
                ts.append(hit)

        if len(ts) < 2:
            return None

        lo = max(0.0, min(ts))
        hi = min(1.0, max(ts))
        if hi - lo <= 1e-6:
            return None

        return (self.lerp(e0, e1, lo), self.lerp(e0, e1, hi))

    def in_cone(self, root, a, b, point):
        ax, ay = a[0] - root[0], a[1] - root[1]
        bx, by = b[0] - root[0], b[1] - root[1]
        px, py = point[0] - root[0], point[1] - root[1]

        cross_ab = self.cross((ax, ay), (bx, by))
        cross_ap = self.cross((ax, ay), (px, py))
        cross_pb = self.cross((px, py), (bx, by))

        if abs(cross_ab) < EPS:
            return abs(cross_ap) < 1e-7 and self.dot((ax, ay), (px, py)) >= -EPS

        if cross_ab > 0:
            return cross_ap >= -1e-7 and cross_pb >= -1e-7

        return cross_ap <= 1e-7 and cross_pb <= 1e-7

    def line_segment_intersection_parameter(self, ray_start, ray_through, seg_start, seg_end):
        """Return segment parameter t where a forward ray hits a segment."""

        rx = ray_through[0] - ray_start[0]
        ry = ray_through[1] - ray_start[1]
        sx = seg_end[0] - seg_start[0]
        sy = seg_end[1] - seg_start[1]
        denom = self.cross((rx, ry), (sx, sy))

        if abs(denom) < EPS:
            return None

        qpx = seg_start[0] - ray_start[0]
        qpy = seg_start[1] - ray_start[1]
        u = self.cross((qpx, qpy), (sx, sy)) / denom
        t = self.cross((qpx, qpy), (rx, ry)) / denom

        if u >= -1e-7 and -1e-7 <= t <= 1.0 + 1e-7:
            return min(1.0, max(0.0, t))

        return None

    def line_of_sight(self, start, goal):
        key = self.segment_key(start, goal)
        if key in self.los_cache:
            return self.los_cache[key]

        if not self.point_inside_world(start) or not self.point_inside_world(goal):
            self.los_cache[key] = False
            return False

        for obs in self.obs:
            if self.segment_intersects_cell_interior(start, goal, obs):
                self.los_cache[key] = False
                return False

        self.los_cache[key] = True
        return True

    def segment_intersects_cell_interior(self, start, goal, cell):
        xmin, xmax = cell[0] - 0.5 + 1e-7, cell[0] + 0.5 - 1e-7
        ymin, ymax = cell[1] - 0.5 + 1e-7, cell[1] + 0.5 - 1e-7

        if xmin >= xmax or ymin >= ymax:
            return False

        x0, y0 = start
        x1, y1 = goal
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0

        for p, q in (
            (-dx, x0 - xmin),
            (dx, xmax - x0),
            (-dy, y0 - ymin),
            (dy, ymax - y0),
        ):
            if abs(p) < EPS:
                if q < 0:
                    return False
                continue

            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)

        return t0 < t1 and t1 > EPS and t0 < 1.0 - EPS

    def segment_inside_cell(self, start, goal, cell):
        xmin, xmax = cell[0] - 0.5 - EPS, cell[0] + 0.5 + EPS
        ymin, ymax = cell[1] - 0.5 - EPS, cell[1] + 0.5 + EPS
        for point in (start, goal):
            if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                return False
        return True

    def valid_turn_point(self, point):
        if not self.is_mesh_vertex(point):
            return False

        cells = {
            (math.floor(point[0] + sx), math.floor(point[1] + sy))
            for sx in (-0.1, 0.1)
            for sy in (-0.1, 0.1)
        }
        return any(self.is_free(cell) for cell in cells)

    @staticmethod
    def is_mesh_vertex(point):
        return abs(point[0] * 2 - round(point[0] * 2)) < 1e-7 and abs(point[1] * 2 - round(point[1] * 2)) < 1e-7

    def portal_between(self, cell, next_cell):
        cx, cy = cell
        nx, ny = next_cell

        if nx == cx + 1:
            return ((cx + 0.5, cy - 0.5), (cx + 0.5, cy + 0.5))
        if nx == cx - 1:
            return ((cx - 0.5, cy + 0.5), (cx - 0.5, cy - 0.5))
        if ny == cy + 1:
            return ((cx + 0.5, cy + 0.5), (cx - 0.5, cy + 0.5))
        if ny == cy - 1:
            return ((cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5))

        raise ValueError("Cells are not adjacent.")

    def cell_bounds(self, cell):
        x, y = cell
        return [
            ((x - 0.5, y - 0.5), (x + 0.5, y - 0.5)),
            ((x + 0.5, y - 0.5), (x + 0.5, y + 0.5)),
            ((x + 0.5, y + 0.5), (x - 0.5, y + 0.5)),
            ((x - 0.5, y + 0.5), (x - 0.5, y - 0.5)),
        ]

    def point_to_cell(self, point):
        return (int(math.floor(point[0] + 0.5)), int(math.floor(point[1] + 0.5)))

    def point_inside_world(self, point):
        return 0 <= point[0] < self.Env.x_range and 0 <= point[1] < self.Env.y_range

    def point_in_cell(self, point, cell):
        return (
            cell[0] - 0.5 - EPS <= point[0] <= cell[0] + 0.5 + EPS
            and cell[1] - 0.5 - EPS <= point[1] <= cell[1] + 0.5 + EPS
        )

    def is_free(self, cell):
        x, y = cell
        return 0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range and cell not in self.obs

    def heuristic_interval(self, state):
        p = self.closest_point_on_interval(self.s_goal, state.interval)
        return self.distance(state.root, p) + self.distance(p, self.s_goal)

    def closest_point_on_interval(self, point, interval):
        a, b = interval
        ab = (b[0] - a[0], b[1] - a[1])
        denom = self.dot(ab, ab)
        if denom < EPS:
            return a

        ap = (point[0] - a[0], point[1] - a[1])
        t = max(0.0, min(1.0, self.dot(ap, ab) / denom))
        return self.lerp(a, b, t)

    def compact_path(self, path):
        compact = []
        for point in path:
            rounded = (round(point[0], 6), round(point[1], 6))
            if compact and self.distance(compact[-1], rounded) < 1e-6:
                continue
            compact.append(rounded)

        changed = True
        while changed and len(compact) > 2:
            changed = False
            reduced = [compact[0]]
            for i in range(1, len(compact) - 1):
                if self.line_of_sight(reduced[-1], compact[i + 1]):
                    changed = True
                    continue
                reduced.append(compact[i])
            reduced.append(compact[-1])
            compact = reduced

        return compact

    def state_key(self, state):
        return (
            state.cell,
            round(state.root[0], 4),
            round(state.root[1], 4),
        )

    @staticmethod
    def segment_key(start, goal):
        a = (round(start[0], 6), round(start[1], 6))
        b = (round(goal[0], 6), round(goal[1], 6))
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def interval_midpoint(interval):
        a, b = interval
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    @staticmethod
    def interval_length(interval):
        a, b = interval
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class PlottingPolyanya:
    """Visualization and GIF generation for Polyanya interval search."""

    def __init__(self, xI, xG, planner):
        import numpy as np
        from PIL import Image
        from PIL import ImageDraw

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            plt = None

        self.plt = plt
        self.np = np
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.xI, self.xG = xI, xG
        self.planner = planner
        self.env = planner.Env
        self.obs = planner.obs
        self.frames = []
        self.fig_size = (7, 4.5)
        self.fallback_intervals = []
        self.fallback_roots = []
        self.fallback_path = []

    def animation(self, path, visited, name, save_gif=False):
        self.plot_grid(name)
        self.plot_interval_search(visited)
        self.plot_path(path)

        if save_gif:
            self.save_animation_as_gif(name)

    def plot_grid(self, name):
        plt = self.plt
        self.name = name

        if plt is None:
            self.capture_frame()
            return

        plt.figure(figsize=self.fig_size, dpi=110, clear=True)
        self.draw_base(name)
        self.capture_frame()

    def draw_base(self, name):
        plt = self.plt

        ax = plt.gca()
        for ox, oy in self.obs:
            rect = plt.Rectangle(
                (ox - 0.5, oy - 0.5),
                1.0,
                1.0,
                facecolor="black",
                edgecolor="black",
                linewidth=0.2,
            )
            ax.add_patch(rect)

        plt.plot(self.xI[0], self.xI[1], "bs", markersize=8, label="Start")
        plt.plot(self.xG[0], self.xG[1], "gs", markersize=8, label="Goal")
        plt.title(name)
        plt.axis("equal")
        plt.xlim(-1, self.env.x_range)
        plt.ylim(-1, self.env.y_range)
        plt.grid(True, color="#dddddd", linewidth=0.4, alpha=0.8)

    def plot_interval_search(self, visited):
        plt = self.plt

        step = max(1, len(visited) // 90)

        for idx, state in enumerate(visited):
            if idx % step != 0 and idx != len(visited) - 1:
                continue

            a, b = state.interval
            if plt is None:
                self.fallback_intervals.append((a, b))
                self.fallback_roots.append(state.root)
            else:
                plt.plot([a[0], b[0]], [a[1], b[1]], color="#00a7c7", linewidth=1.4, alpha=0.55)
                plt.plot(state.root[0], state.root[1], marker=".", color="#ff8c00", markersize=3, alpha=0.8)

            if idx % (step * 4) == 0 or idx == len(visited) - 1:
                self.capture_frame()

    def plot_path(self, path):
        plt = self.plt

        if plt is None:
            self.fallback_path = path
            self.capture_frame()
            return

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            plt.plot(px, py, color="red", linewidth=3.0, marker="o", markersize=4, label="Path")

        plt.plot(self.xI[0], self.xI[1], "bs", markersize=8)
        plt.plot(self.xG[0], self.xG[1], "gs", markersize=8)
        self.capture_frame()

    def capture_frame(self):
        plt = self.plt
        np = self.np
        Image = self.Image

        if plt is None:
            self.frames.append(np.array(self.render_fallback_frame()))
            return

        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        image = np.array(Image.open(buf).convert("RGB"))
        self.frames.append(image)
        buf.close()

    def render_fallback_frame(self):
        Image = self.Image
        ImageDraw = self.ImageDraw

        width, height = 760, 480
        margin = 24
        x_min, x_max = -1.0, float(self.env.x_range)
        y_min, y_max = -1.0, float(self.env.y_range)
        sx = (width - 2 * margin) / (x_max - x_min)
        sy = (height - 2 * margin) / (y_max - y_min)
        scale = min(sx, sy)

        def to_px(point):
            x = margin + (point[0] - x_min) * scale
            y = height - margin - (point[1] - y_min) * scale
            return (x, y)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img, "RGBA")

        for x in range(self.env.x_range + 1):
            px, _ = to_px((x - 0.5, 0))
            draw.line([(px, margin), (px, height - margin)], fill=(225, 225, 225, 160), width=1)
        for y in range(self.env.y_range + 1):
            _, py = to_px((0, y - 0.5))
            draw.line([(margin, py), (width - margin, py)], fill=(225, 225, 225, 160), width=1)

        for ox, oy in self.obs:
            x0, y0 = to_px((ox - 0.5, oy - 0.5))
            x1, y1 = to_px((ox + 0.5, oy + 0.5))
            draw.rectangle([x0, y1, x1, y0], fill=(0, 0, 0, 255))

        for a, b in self.fallback_intervals:
            draw.line([to_px(a), to_px(b)], fill=(0, 167, 199, 130), width=2)

        for root in self.fallback_roots[-1200:]:
            x, y = to_px(root)
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 140, 0, 160))

        if self.fallback_path:
            points = [to_px(point) for point in self.fallback_path]
            draw.line(points, fill=(220, 0, 0, 255), width=4)
            for x, y in points:
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 0, 0, 255))

        for point, color in ((self.xI, (30, 90, 230, 255)), (self.xG, (35, 160, 60, 255))):
            x, y = to_px(point)
            draw.rectangle([x - 6, y - 6, x + 6, y + 6], fill=color)

        draw.text((margin, 6), getattr(self, "name", "029_Polyanya"), fill=(0, 0, 0, 255))
        return img

    def save_animation_as_gif(self, name, fps=10):
        plt = self.plt
        np = self.np
        Image = self.Image

        gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        if not self.frames:
            print("No frames to save.")
            return

        first_shape = self.frames[0].shape
        fixed_frames = []
        for frame in self.frames:
            if frame.shape != first_shape:
                frame = np.array(
                    Image.fromarray(frame).resize((first_shape[1], first_shape[0]), Image.LANCZOS)
                )
            fixed_frames.append(Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256))

        fixed_frames[0].save(
            gif_path,
            format="GIF",
            append_images=fixed_frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        if plt is not None:
            plt.close()
        print(f"GIF animation saved to {gif_path}")


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


def main():
    """Run the Polyanya any-angle pathfinding demo."""

    print("Polyanya Any-Angle Pathfinding Demo")
    print("=" * 45)

    s_start = (5, 5)
    s_goal = (45, 25)

    planner = Polyanya(s_start, s_goal)
    path, visited = planner.searching()

    print(f"Start: {s_start}")
    print(f"Goal: {s_goal}")
    print(f"Visited interval states: {len(visited)}")

    if path:
        print(f"Path waypoints: {len(path)}")
        print(f"Path length: {path_length(path):.3f}")
        print(f"Path: {path}")
    else:
        print("No path found.")

    plot = PlottingPolyanya(s_start, s_goal, planner)
    plot.animation(path, visited, "029_Polyanya", save_gif=True)


if __name__ == "__main__":
    main()
