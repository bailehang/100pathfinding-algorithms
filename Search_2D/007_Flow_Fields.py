"""
Flow Fields Pathfinding Demo
@author: clark bai
@author: assistant

Flow-field pathfinding builds one goal-centered integration field and then
stores, for every free cell, the best local direction to move toward that goal.
It is especially useful when many agents share the same destination.
"""

from metrics import install_metrics, latest_metrics_line, print_latest_metrics
install_metrics()

import heapq
import math
import os
from collections import OrderedDict

from PIL import Image, ImageDraw


class Env:
    """Environment class for 2D grid world."""

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


class FlowFields:
    """Goal-centered cost, integration, and flow-field pathfinding."""

    def __init__(self, starts, goal):
        self.starts = starts
        self.goal = goal
        self.Env = Env()
        self.obs = self.Env.obs
        self.cost_field = {}
        self.integration_field = {}
        self.flow_field = {}
        self.visited = []
        self.paths = OrderedDict()

    def searching(self):
        self.build_cost_field()
        self.build_integration_field()
        self.build_flow_field()

        for start in self.starts:
            self.paths[start] = self.trace_path(start)

        primary_path = self.paths[self.starts[0]]
        return primary_path, self.visited, self.flow_field

    def build_cost_field(self):
        for x in range(self.Env.x_range):
            for y in range(self.Env.y_range):
                node = (x, y)
                self.cost_field[node] = math.inf if node in self.obs else 1.0

    def build_integration_field(self):
        for node in self.cost_field:
            self.integration_field[node] = math.inf

        if self.goal in self.obs:
            raise ValueError("Goal lies inside an obstacle.")

        self.integration_field[self.goal] = 0.0
        open_list = [(0.0, self.goal)]

        while open_list:
            current_cost, current = heapq.heappop(open_list)
            if current_cost > self.integration_field[current]:
                continue

            self.visited.append(current)

            for neighbor in self.get_neighbors(current):
                move_cost = self.motion_cost(current, neighbor)
                new_cost = current_cost + move_cost * self.cost_field[neighbor]

                if new_cost < self.integration_field[neighbor]:
                    self.integration_field[neighbor] = new_cost
                    heapq.heappush(open_list, (new_cost, neighbor))

    def build_flow_field(self):
        for x in range(self.Env.x_range):
            for y in range(self.Env.y_range):
                node = (x, y)
                if node in self.obs or self.integration_field[node] == math.inf:
                    continue

                if node == self.goal:
                    self.flow_field[node] = (0, 0)
                    continue

                best_neighbor = min(
                    self.get_neighbors(node),
                    key=lambda n: self.integration_field.get(n, math.inf),
                    default=None,
                )

                if best_neighbor is None:
                    continue

                self.flow_field[node] = (best_neighbor[0] - x, best_neighbor[1] - y)

    def trace_path(self, start, max_steps=500):
        if start in self.obs:
            raise ValueError("Start lies inside an obstacle.")

        path = [start]
        current = start
        seen = {current}

        for _ in range(max_steps):
            if current == self.goal:
                break

            direction = self.flow_field.get(current)
            if direction is None:
                break

            nxt = (current[0] + direction[0], current[1] + direction[1])
            if nxt in seen or nxt in self.obs:
                break

            path.append(nxt)
            seen.add(nxt)
            current = nxt

        return path

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in self.Env.motions:
            nxt = (node[0] + dx, node[1] + dy)
            if self.is_free(nxt) and not self.is_diagonal_collision(node, nxt):
                neighbors.append(nxt)
        return neighbors

    def is_free(self, node):
        x, y = node
        return 0 <= x < self.Env.x_range and 0 <= y < self.Env.y_range and node not in self.obs

    def is_diagonal_collision(self, current, nxt):
        if current[0] == nxt[0] or current[1] == nxt[1]:
            return False

        side_a = (current[0], nxt[1])
        side_b = (nxt[0], current[1])
        return side_a in self.obs or side_b in self.obs

    @staticmethod
    def motion_cost(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


class PlottingFlowFields:
    """Pillow-based flow-field visualization and GIF generation."""

    def __init__(self, planner):
        self.planner = planner
        self.env = planner.Env
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

    def animation(self, name, save_gif=True):
        self.capture_frame(name, visited_count=0, show_flow=False, visible_paths={})

        step = max(1, len(self.planner.visited) // 22)
        for count in range(step, len(self.planner.visited) + 1, step):
            self.capture_frame(name, visited_count=count, show_flow=False, visible_paths={})

        self.capture_frame(name, visited_count=len(self.planner.visited), show_flow=True, visible_paths={})

        longest = max((len(path) for path in self.planner.paths.values()), default=0)
        path_step = max(1, longest // 18)
        for length in range(path_step, longest + path_step, path_step):
            visible_paths = {
                start: path[: min(length, len(path))]
                for start, path in self.planner.paths.items()
            }
            self.capture_frame(
                name,
                visited_count=len(self.planner.visited),
                show_flow=True,
                visible_paths=visible_paths,
            )

        if save_gif:
            self.save_animation_as_gif(name)

    def capture_frame(self, name, visited_count, show_flow, visible_paths):
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img, "RGBA")

        self.draw_grid(draw)
        self.draw_integration(draw, visited_count)
        self.draw_obstacles(draw)

        if show_flow:
            self.draw_flow(draw)

        self.draw_paths(draw, visible_paths)
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

    def draw_integration(self, draw, visited_count):
        visible = self.planner.visited[:visited_count]
        finite_costs = [
            self.planner.integration_field[node]
            for node in visible
            if self.planner.integration_field[node] < math.inf
        ]
        max_cost = max(finite_costs, default=1.0)

        for node in visible:
            cost = self.planner.integration_field[node]
            if cost == math.inf:
                continue

            ratio = min(1.0, cost / max_cost)
            color = (
                int(250 - 90 * ratio),
                int(245 - 160 * ratio),
                int(210 - 170 * ratio),
                155,
            )
            self.draw_cell(draw, node, color)

    def draw_obstacles(self, draw):
        for obs in self.obs:
            self.draw_cell(draw, obs, (0, 0, 0, 255))

    def draw_flow(self, draw):
        for node, direction in self.planner.flow_field.items():
            if node == self.planner.goal:
                continue
            if node[0] % 2 != 0 or node[1] % 2 != 0:
                continue

            x0, y0 = self.to_px(node)
            end = (node[0] + direction[0] * 0.36, node[1] + direction[1] * 0.36)
            x1, y1 = self.to_px(end)
            draw.line([(x0, y0), (x1, y1)], fill=(0, 115, 155, 190), width=2)
            self.draw_arrow_head(draw, (x0, y0), (x1, y1), (0, 115, 155, 190))

    def draw_paths(self, draw, visible_paths):
        colors = [
            (220, 0, 0, 255),
            (130, 70, 210, 255),
            (25, 130, 80, 255),
            (230, 115, 0, 255),
        ]
        for idx, path in enumerate(visible_paths.values()):
            if len(path) < 2:
                continue
            points = [self.to_px(node) for node in path]
            color = colors[idx % len(colors)]
            draw.line(points, fill=color, width=4)
            for x, y in points[-3:]:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    def draw_points(self, draw):
        for start in self.planner.starts:
            x, y = self.to_px(start)
            draw.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(35, 95, 230, 255))

        x, y = self.to_px(self.planner.goal)
        draw.rectangle([x - 6, y - 6, x + 6, y + 6], fill=(35, 165, 60, 255))

    def draw_cell(self, draw, node, color):
        x0, y0 = self.to_px((node[0] - 0.5, node[1] - 0.5))
        x1, y1 = self.to_px((node[0] + 0.5, node[1] + 0.5))
        draw.rectangle([x0, y1, x1, y0], fill=color)

    def draw_arrow_head(self, draw, start, end, color):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return

        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        size = 4
        p1 = (end[0] - ux * size + px * size * 0.65, end[1] - uy * size + py * size * 0.65)
        p2 = (end[0] - ux * size - px * size * 0.65, end[1] - uy * size - py * size * 0.65)
        draw.polygon([end, p1, p2], fill=color)

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
    """Run the Flow Fields pathfinding demo."""

    print("Flow Fields Pathfinding Demo")
    print("=" * 40)

    starts = [(5, 5), (8, 24), (12, 8), (45, 5)]
    goal = (45, 25)

    planner = FlowFields(starts, goal)
    path, visited, _ = planner.searching()
    print_latest_metrics()

    print(f"Goal: {goal}")
    print(f"Starts: {starts}")
    print(f"Visited cells while building integration field: {len(visited)}")
    print(f"Primary path waypoints: {len(path)}")
    print(f"Primary path length: {path_length(path):.3f}")

    plot = PlottingFlowFields(planner)
    plot.animation("007_Flow_Fields", save_gif=True)


if __name__ == "__main__":
    main()
