"""
Bellman-Ford 2D pathfinding demo.

Bellman-Ford relaxes every edge repeatedly, which makes it slower than
Dijkstra on ordinary non-negative grids, but it is useful as a reference
single-source shortest path algorithm because it can detect negative cycles.
"""

from metrics import install_metrics, latest_metrics_line, print_latest_metrics
install_metrics()

import math
import os

from PIL import Image, ImageDraw


class Env:
    """Environment class for a 2D grid world."""

    def __init__(self):
        self.x_range = 51
        self.y_range = 31
        self.motions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]
        self.obs = self.obs_map()

    def obs_map(self):
        obs = set()

        for i in range(self.x_range):
            obs.add((i, 0))
            obs.add((i, self.y_range - 1))

        for i in range(self.y_range):
            obs.add((0, i))
            obs.add((self.x_range - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))
        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs


class BellmanFord:
    """Bellman-Ford search on an 8-connected grid."""

    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.env = Env()
        self.obs = self.env.obs
        self.nodes = self.build_nodes()
        self.edges = self.build_edges()
        self.dist = {}
        self.parent = {}
        self.visited = []
        self.has_negative_cycle = False

    def searching(self):
        self.dist = {node: math.inf for node in self.nodes}
        self.parent = {self.s_start: self.s_start}
        self.dist[self.s_start] = 0.0

        seen_relaxed = set()

        for _ in range(len(self.nodes) - 1):
            changed = False

            for u, v, weight in self.edges:
                if self.dist[u] == math.inf:
                    continue

                new_cost = self.dist[u] + weight
                if new_cost < self.dist[v]:
                    self.dist[v] = new_cost
                    self.parent[v] = u
                    changed = True

                    if v not in seen_relaxed:
                        self.visited.append(v)
                        seen_relaxed.add(v)

            if not changed:
                break

        self.has_negative_cycle = self.detect_negative_cycle()
        return self.extract_path(), self.visited

    def build_nodes(self):
        return [
            (x, y)
            for x in range(self.env.x_range)
            for y in range(self.env.y_range)
            if (x, y) not in self.obs
        ]

    def build_edges(self):
        edges = []
        for node in self.nodes:
            for neighbor in self.get_neighbors(node):
                edges.append((node, neighbor, self.motion_cost(node, neighbor)))
        return edges

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in self.env.motions:
            nxt = (node[0] + dx, node[1] + dy)
            if self.is_free(nxt) and not self.is_diagonal_collision(node, nxt):
                neighbors.append(nxt)
        return neighbors

    def is_free(self, node):
        x, y = node
        return 0 <= x < self.env.x_range and 0 <= y < self.env.y_range and node not in self.obs

    def is_diagonal_collision(self, current, nxt):
        if current[0] == nxt[0] or current[1] == nxt[1]:
            return False

        side_a = (current[0], nxt[1])
        side_b = (nxt[0], current[1])
        return side_a in self.obs or side_b in self.obs

    @staticmethod
    def motion_cost(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def detect_negative_cycle(self):
        for u, v, weight in self.edges:
            if self.dist[u] < math.inf and self.dist[u] + weight < self.dist[v]:
                return True
        return False

    def extract_path(self):
        if self.s_goal not in self.parent:
            return []

        path = [self.s_goal]
        current = self.s_goal

        while current != self.s_start:
            current = self.parent[current]
            path.append(current)

        return list(reversed(path))


class PlottingBellmanFord:
    """Pillow-based visualization and GIF generation."""

    def __init__(self, planner):
        self.planner = planner
        self.env = planner.env
        self.obs = planner.obs
        self.frames = []
        self.width = 760
        self.height = 480
        self.margin = 24
        self.x_min = -1.0
        self.x_max = float(self.env.x_range)
        self.y_min = -1.0
        self.y_max = float(self.env.y_range)
        self.scale = min(
            (self.width - 2 * self.margin) / (self.x_max - self.x_min),
            (self.height - 2 * self.margin) / (self.y_max - self.y_min),
        )

    def animation(self, path, visited, name, save_gif=True):
        self.capture_frame(name, path=[], visited=[])

        step = max(1, len(visited) // 24)
        for count in range(step, len(visited) + 1, step):
            self.capture_frame(name, path=[], visited=visited[:count])

        path_step = max(1, len(path) // 18)
        for count in range(path_step, len(path) + path_step, path_step):
            self.capture_frame(name, path=path[: min(count, len(path))], visited=visited)

        if save_gif:
            self.save_animation_as_gif(name)

    def capture_frame(self, name, path, visited):
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img, "RGBA")

        self.draw_grid(draw)
        self.draw_visited(draw, visited)
        self.draw_obstacles(draw)
        self.draw_path(draw, path)
        self.draw_points(draw)

        draw.text((self.margin, 6), name, fill=(0, 0, 0, 255))
        metric_text = latest_metrics_line()
        if metric_text:
            draw.text((self.margin, 20), metric_text, fill=(0, 0, 0, 255))

        self.frames.append(img)

    def draw_grid(self, draw):
        for x in range(self.env.x_range + 1):
            px, _ = self.to_px((x - 0.5, 0))
            draw.line([(px, self.margin), (px, self.height - self.margin)], fill=(225, 225, 225, 150), width=1)
        for y in range(self.env.y_range + 1):
            _, py = self.to_px((0, y - 0.5))
            draw.line([(self.margin, py), (self.width - self.margin, py)], fill=(225, 225, 225, 150), width=1)

    def draw_visited(self, draw, visited):
        for node in visited:
            self.draw_cell(draw, node, (150, 190, 240, 150))

    def draw_obstacles(self, draw):
        for obs in self.obs:
            self.draw_cell(draw, obs, (0, 0, 0, 255))

    def draw_path(self, draw, path):
        if len(path) < 2:
            return

        points = [self.to_px(node) for node in path]
        draw.line(points, fill=(220, 0, 0, 255), width=4)
        for x, y in points[-3:]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(220, 0, 0, 255))

    def draw_points(self, draw):
        sx, sy = self.to_px(self.planner.s_start)
        gx, gy = self.to_px(self.planner.s_goal)
        draw.rectangle([sx - 5, sy - 5, sx + 5, sy + 5], fill=(35, 95, 230, 255))
        draw.rectangle([gx - 6, gy - 6, gx + 6, gy + 6], fill=(35, 165, 60, 255))

    def draw_cell(self, draw, node, color):
        x0, y0 = self.to_px((node[0] - 0.5, node[1] - 0.5))
        x1, y1 = self.to_px((node[0] + 0.5, node[1] + 0.5))
        draw.rectangle([x0, y1, x1, y0], fill=color)

    def to_px(self, point):
        x = self.margin + (point[0] - self.x_min) * self.scale
        y = self.height - self.margin - (point[1] - self.y_min) * self.scale
        return (x, y)

    def save_animation_as_gif(self, name, fps=8):
        gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        if not self.frames:
            print("No frames to save.")
            return

        frames = [frame.convert("P", palette=Image.ADAPTIVE, colors=256) for frame in self.frames]
        frames[0].save(
            gif_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        print(f"GIF animation saved to {gif_path}")


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


def main():
    print("Bellman-Ford Pathfinding Demo")
    print("=" * 40)

    s_start = (5, 5)
    s_goal = (45, 25)

    planner = BellmanFord(s_start, s_goal)
    path, visited = planner.searching()
    print_latest_metrics()

    print(f"Start: {s_start}")
    print(f"Goal: {s_goal}")
    print(f"Relaxed nodes: {len(visited)}")
    print(f"Negative cycle detected: {planner.has_negative_cycle}")
    print(f"Path waypoints: {len(path)}")
    print(f"Path length: {path_length(path):.3f}")

    plot = PlottingBellmanFord(planner)
    plot.animation(path, visited, "005_Bellman_Ford", save_gif=True)


if __name__ == "__main__":
    main()
