"""RRT-Dubins 2D path planning demo."""

from metrics import install_metrics
install_metrics()

import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


class DubinsNode:
    def __init__(self, x, y, yaw, parent=None, path=None):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.parent = parent
        self.path = path or []
        self.cost = 0.0 if parent is None else parent.cost + self.path_length()

    @property
    def state(self):
        return (self.x, self.y, self.yaw)

    @property
    def point(self):
        return (self.x, self.y)

    def path_length(self):
        if len(self.path) < 2:
            return 0.0
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.path, self.path[1:]))


class RRTDubinsPlanner:
    def __init__(self, start=(5.0, 5.0, math.radians(70)), goal=(45.0, 25.0, math.radians(0)), seed=52):
        self.start = DubinsNode(*start)
        self.goal = DubinsNode(*goal)
        self.rng = np.random.default_rng(seed)
        self.x_range = (0.0, 50.0)
        self.y_range = (0.0, 30.0)
        self.max_speed = 1.0
        self.step_dt = 0.42
        self.segment_steps = 10
        self.max_curvature = 0.48
        self.goal_sample_rate = 0.16
        self.goal_tolerance = 2.2
        self.yaw_tolerance = math.radians(34)
        self.nodes = [self.start]
        self.path = []
        self.snapshots = []
        self.obs_circle = [
            (10, 10, 3),
            (15, 22, 3),
            (22, 8, 2.5),
            (27, 16, 2.2),
            (37, 10, 3),
            (37, 23, 3),
            (45, 15, 2),
        ]

    def planning(self, save_gif=False, gif_name="052_dubins_rrt"):
        self.snapshots = [self.snapshot(0, "RRT samples position plus heading")]
        goal_node = None
        for step in range(1, 3600):
            sample = self.sample_state(step)
            nearest = self.nearest_node(sample)
            new_node = self.steer(nearest, sample)
            if new_node is None or self.is_path_collision(new_node.path):
                continue
            self.nodes.append(new_node)
            if step < 95 or step % 70 == 0:
                self.snapshots.append(self.snapshot(step, self.phase_text(step), sample=sample))
            if self.can_connect_goal(new_node):
                goal_node = self.connect_goal(new_node)
                if goal_node is not None:
                    self.nodes.append(goal_node)
                    self.path = self.extract_path(goal_node)
                    self.snapshots.append(self.snapshot(step, "goal pose reached with bounded-curvature motion", final=True))
                    break
        if goal_node is None:
            raise RuntimeError("RRT-Dubins failed to reach the goal pose")
        self.snapshots.append(self.snapshot(len(self.nodes), "final Dubins-constrained route", final=True))
        if save_gif:
            self.save_gif(gif_name)
        return self.path

    def sample_state(self, step):
        if self.rng.random() < self.goal_sample_rate or step % 31 == 0:
            return self.goal.state
        return (
            self.rng.uniform(self.x_range[0] + 2.0, self.x_range[1] - 2.0),
            self.rng.uniform(self.y_range[0] + 2.0, self.y_range[1] - 2.0),
            self.rng.uniform(-math.pi, math.pi),
        )

    def nearest_node(self, sample):
        sx, sy, syaw = sample
        return min(
            self.nodes,
            key=lambda node: math.hypot(node.x - sx, node.y - sy) + 2.2 * abs(self.angle_diff(node.yaw, syaw)),
        )

    def steer(self, node, sample):
        best = None
        best_score = math.inf
        for curvature in np.linspace(-self.max_curvature, self.max_curvature, 13):
            path = self.rollout(node.state, curvature)
            end = path[-1]
            score = math.hypot(end[0] - sample[0], end[1] - sample[1]) + 1.6 * abs(self.angle_diff(end[2], sample[2]))
            if score < best_score:
                best_score = score
                best = (end, path)
        if best is None:
            return None
        end, path = best
        if not self.in_bounds((end[0], end[1])):
            return None
        return DubinsNode(end[0], end[1], end[2], node, path)

    def rollout(self, state, curvature):
        x, y, yaw = state
        path = [(x, y, yaw)]
        for _ in range(self.segment_steps):
            yaw = self.wrap_angle(yaw + curvature * self.max_speed * self.step_dt)
            x += self.max_speed * math.cos(yaw) * self.step_dt
            y += self.max_speed * math.sin(yaw) * self.step_dt
            path.append((x, y, yaw))
        return path

    def can_connect_goal(self, node):
        return math.hypot(node.x - self.goal.x, node.y - self.goal.y) <= self.goal_tolerance * 2.8

    def connect_goal(self, node):
        current = node
        full_path = []
        for _ in range(8):
            target = self.goal.state
            next_node = self.steer(current, target)
            if next_node is None or self.is_path_collision(next_node.path):
                return None
            full_path.extend(next_node.path if not full_path else next_node.path[1:])
            current = next_node
            if math.hypot(current.x - self.goal.x, current.y - self.goal.y) < self.goal_tolerance and abs(self.angle_diff(current.yaw, self.goal.yaw)) < self.yaw_tolerance:
                final_path = self.short_pose_path(current.state, self.goal.state)
                if final_path and not self.is_path_collision(final_path):
                    return DubinsNode(self.goal.x, self.goal.y, self.goal.yaw, current, final_path)
        return None

    def short_pose_path(self, start, goal):
        sx, sy, syaw = start
        gx, gy, gyaw = goal
        path = [(sx, sy, syaw)]
        x, y, yaw = sx, sy, syaw
        for _ in range(24):
            heading_to_goal = math.atan2(gy - y, gx - x)
            yaw_error = self.angle_diff(heading_to_goal, yaw)
            curvature = float(np.clip(yaw_error / max(self.step_dt * 2.4, 1e-6), -self.max_curvature, self.max_curvature))
            if math.hypot(gx - x, gy - y) < 0.45:
                yaw_error = self.angle_diff(gyaw, yaw)
                curvature = float(np.clip(yaw_error / max(self.step_dt * 2.4, 1e-6), -self.max_curvature, self.max_curvature))
            yaw = self.wrap_angle(yaw + curvature * self.max_speed * self.step_dt)
            x += self.max_speed * math.cos(yaw) * self.step_dt
            y += self.max_speed * math.sin(yaw) * self.step_dt
            path.append((x, y, yaw))
            if math.hypot(gx - x, gy - y) < 0.65 and abs(self.angle_diff(gyaw, yaw)) < self.yaw_tolerance:
                path.append((gx, gy, gyaw))
                return path
        return None

    def extract_path(self, node):
        segments = []
        current = node
        while current is not None:
            if current.path:
                segments.append(current.path)
            else:
                segments.append([current.state])
            current = current.parent
        route = []
        for segment in reversed(segments):
            if route:
                route.extend(segment[1:])
            else:
                route.extend(segment)
        return route

    def is_path_collision(self, path):
        return any(self.inside_obstacle((x, y)) for x, y, _ in path)

    def inside_obstacle(self, point):
        x, y = point
        if not self.in_bounds(point):
            return True
        for ox, oy, radius in self.obs_circle:
            if math.hypot(x - ox, y - oy) <= radius + 0.9:
                return True
        return False

    def phase_text(self, step):
        if step < 35:
            return "expanding short arcs with bounded curvature"
        if step % 31 == 0:
            return "goal pose sample biases the tree toward target heading"
        return "nodes store x, y, yaw and Dubins-like arc segments"

    def snapshot(self, step, phase, sample=None, final=False):
        return {
            "step": step,
            "phase": phase,
            "sample": sample,
            "final": final,
            "nodes": list(self.nodes),
            "path": list(self.path),
        }

    def save_gif(self, gif_name, max_frames=58):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(self.snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 5)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=320, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=110)
        self.draw_environment(ax)
        for node in snapshot["nodes"]:
            if len(node.path) > 1:
                ax.plot([p[0] for p in node.path], [p[1] for p in node.path], color="#94a3b8", linewidth=0.62, alpha=0.5, zorder=2)
        if snapshot["sample"]:
            ax.scatter(snapshot["sample"][0], snapshot["sample"][1], s=22, color="#f97316", alpha=0.85, zorder=4)
        if snapshot["path"]:
            ax.plot([p[0] for p in snapshot["path"]], [p[1] for p in snapshot["path"]], color="#dc2626", linewidth=2.8, alpha=0.95, zorder=5)
            for state in snapshot["path"][::max(1, len(snapshot["path"]) // 10)]:
                self.draw_heading(ax, state, "#dc2626", 0.6)
        self.draw_heading(ax, self.start.state, "#2563eb", 1.0)
        self.draw_heading(ax, self.goal.state, "#15803d", 1.0)
        ax.scatter(self.start.x, self.start.y, marker="s", s=75, color="#2563eb", zorder=6)
        ax.scatter(self.goal.x, self.goal.y, marker="s", s=75, color="#15803d", zorder=6)
        ax.text(
            1.3,
            28.0,
            f"RRT-Dubins  step {snapshot['step']:4d}  states {len(snapshot['nodes']):4d}\n{snapshot['phase']}",
            fontsize=8.4,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("052 RRT-Dubins - bounded curvature states")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def draw_environment(self, ax):
        ax.add_patch(patches.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0, zorder=0))
        for ox, oy, radius in self.obs_circle:
            ax.add_patch(patches.Circle((ox, oy), radius, facecolor="#374151", edgecolor="#111827", linewidth=1.0, zorder=1))

    @staticmethod
    def draw_heading(ax, state, color, alpha):
        x, y, yaw = state
        ax.arrow(x, y, math.cos(yaw) * 1.4, math.sin(yaw) * 1.4, color=color, width=0.035, head_width=0.35, alpha=alpha, zorder=7)

    def in_bounds(self, point):
        return self.x_range[0] + 1.0 <= point[0] <= self.x_range[1] - 1.0 and self.y_range[0] + 1.0 <= point[1] <= self.y_range[1] - 1.0

    @staticmethod
    def angle_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main():
    planner = RRTDubinsPlanner()
    path = planner.planning(save_gif=True, gif_name="052_dubins_rrt")
    if not path:
        raise RuntimeError("RRT-Dubins returned no path")


if __name__ == "__main__":
    main()
