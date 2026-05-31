import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


class AvoidanceAgent:
    def __init__(self, agent_id, start, goal, group, color):
        self.id = agent_id
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.pos = self.start.copy()
        self.vel = np.zeros(2, dtype=float)
        self.group = group
        self.color = color
        self.trail = [tuple(self.pos)]


class VelocityObstacleComparisonDemo:
    def __init__(self, mode, label):
        self.mode = mode
        self.label = label
        self.dt = 0.38
        self.max_steps = 190
        self.radius = 0.58
        self.pref_speed = 1.18
        self.obstacles = [
            ("circle", (25.0, 19.0, 2.45)),
            ("rect", (23.4, 14.0, 3.2, 2.15)),
        ]
        colors = ["#2563eb", "#0ea5e9", "#14b8a6", "#22c55e", "#84cc16",
                  "#dc2626", "#f97316", "#eab308", "#a855f7", "#ec4899"]
        lanes = [6.4, 10.6, 14.8, 19.0, 23.2]
        self.agents = []
        for i, y in enumerate(lanes):
            self.agents.append(AvoidanceAgent(i, (4.0, y), (46.0, y), "L", colors[i]))
        for i, y in enumerate(lanes):
            self.agents.append(AvoidanceAgent(i + 5, (46.0, y), (4.0, y), "R", colors[i + 5]))
        self.frames = []
        self.snapshots = []
        self.min_separation = math.inf

    def planning(self, save_gif=False, gif_name="multi_agent_avoidance"):
        self.update_min_separation()
        self.snapshots = [self.snapshot(0, "ten agents start head-on with central obstacles")]
        for step in range(1, self.max_steps + 1):
            positions = [agent.pos.copy() for agent in self.agents]
            velocities = [agent.vel.copy() for agent in self.agents]
            chosen = []
            for index, agent in enumerate(self.agents):
                if self.reached_goal(agent):
                    chosen.append(np.zeros(2))
                    continue
                chosen.append(self.choose_velocity(index, positions, velocities, step))
            for agent, velocity in zip(self.agents, chosen):
                agent.vel = velocity
                if not self.reached_goal(agent):
                    agent.pos = agent.pos + velocity * self.dt
                    agent.pos[0] = float(np.clip(agent.pos[0], 1.6, 48.4))
                    agent.pos[1] = float(np.clip(agent.pos[1], 1.6, 28.4))
            self.resolve_overlaps()
            for agent in self.agents:
                agent.trail.append(tuple(agent.pos))
            self.update_min_separation()
            if step < 35 or step % 5 == 0 or self.all_reached():
                self.snapshots.append(self.snapshot(step, self.phase_text(step)))
            if self.all_reached() and step > 24:
                self.snapshots.append(self.snapshot(step, "all agents have swapped sides without overlap", final=True))
                break
        if not self.all_reached():
            self.snapshots.append(self.snapshot(self.max_steps, "time limit reached; compare remaining avoidance behavior", final=True))
        if save_gif:
            self.save_gif(gif_name)
        return [agent.trail for agent in self.agents]

    def choose_velocity(self, agent_index, positions, velocities, step):
        agent = self.agents[agent_index]
        desired = self.desired_velocity(agent, step)
        candidates = self.velocity_candidates(desired, agent, step)
        best_velocity = desired
        best_score = math.inf
        for velocity in candidates:
            score = self.score_velocity(agent_index, velocity, positions, velocities, desired)
            if score < best_score:
                best_score = score
                best_velocity = velocity
        return best_velocity

    def desired_velocity(self, agent, step):
        to_goal = agent.goal - agent.pos
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return np.zeros(2)
        direction = to_goal / dist
        steer = np.zeros(2)
        for obstacle in self.obstacles:
            center, radius = self.obstacle_circle_approx(obstacle)
            ahead = center - agent.pos
            forward = float(np.dot(ahead, direction))
            lateral = self.cross2(direction, ahead)
            if 0.0 < forward < 13.5 and abs(lateral) < radius + 1.7:
                side = self.side_preference(agent, center, step)
                steer += np.array([-direction[1], direction[0]]) * side * (1.0 - abs(lateral) / (radius + 1.7))
        group_shift = 0.11 if agent.group == "L" else -0.11
        if self.mode in {"porca", "ervo"}:
            group_shift *= 2.0
        steer += np.array([0.0, group_shift])
        final = direction + steer
        norm = np.linalg.norm(final)
        if norm < 1e-6:
            return direction * self.pref_speed
        return final / norm * self.preferred_speed(agent)

    def velocity_candidates(self, desired, agent, step):
        base_angle = math.atan2(desired[1], desired[0])
        angle_offsets = np.deg2rad([-95, -70, -48, -30, -16, 0, 16, 30, 48, 70, 95])
        if self.mode in {"orca", "ervo"}:
            angle_offsets = np.deg2rad([-80, -56, -36, -20, -8, 0, 8, 20, 36, 56, 80])
        speed_scales = [0.34, 0.62, 0.86, 1.0, 1.16]
        if self.mode == "porca":
            speed_scales = [0.22, 0.48, 0.72, 0.95, 1.05]
        candidates = [np.zeros(2), desired]
        for offset in angle_offsets:
            for scale in speed_scales:
                angle = base_angle + float(offset)
                speed = self.preferred_speed(agent) * scale
                candidates.append(np.array([math.cos(angle) * speed, math.sin(angle) * speed]))
        return candidates

    def score_velocity(self, agent_index, velocity, positions, velocities, desired):
        agent = self.agents[agent_index]
        predicted = positions[agent_index] + velocity * self.dt
        score = 1.55 * np.linalg.norm(velocity - desired)
        score += 0.035 * np.linalg.norm(agent.goal - predicted)
        if self.mode == "vo":
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=1.0, horizon=2.0, ellipse=False)
        elif self.mode == "rvo":
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.55, horizon=2.8, ellipse=False)
        elif self.mode == "hrvo":
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.65, horizon=3.0, ellipse=False)
            score += self.side_lane_penalty(agent, velocity, strength=0.18)
        elif self.mode == "orca":
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.5, horizon=3.7, ellipse=False) * 1.25
            score += self.abrupt_change_penalty(agent, velocity, 0.12)
        elif self.mode == "porca":
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.5, horizon=3.4, ellipse=False)
            score += self.side_lane_penalty(agent, velocity, strength=0.35)
            score += max(0.0, np.linalg.norm(velocity) - self.pref_speed * 0.92) * 0.35
        else:
            score += self.agent_penalty(agent_index, velocity, positions, velocities, responsibility=0.5, horizon=4.1, ellipse=True) * 1.15
            score += self.side_lane_penalty(agent, velocity, strength=0.22)
        score += self.obstacle_penalty(agent, predicted)
        score += self.boundary_penalty(predicted)
        return float(score)

    def agent_penalty(self, agent_index, velocity, positions, velocities, responsibility, horizon, ellipse):
        penalty = 0.0
        pos = positions[agent_index]
        for other_index, other in enumerate(self.agents):
            if other_index == agent_index:
                continue
            rel_pos = positions[other_index] - pos
            rel_vel = velocity - velocities[other_index] * (1.0 - responsibility)
            future_self = pos + velocity * self.dt
            future_other = positions[other_index] + velocities[other_index] * self.dt
            direct_gap = np.linalg.norm(future_other - future_self)
            direct_required = self.radius * (2.15 if not ellipse else 2.55)
            if direct_gap < direct_required:
                penalty += (direct_required - direct_gap) ** 2 * 360.0
            dist_now = np.linalg.norm(rel_pos)
            if dist_now < 1e-6:
                continue
            ttc = self.time_to_closest(rel_pos, rel_vel)
            if 0.0 <= ttc <= horizon:
                closest = rel_pos + rel_vel * ttc
                if ellipse:
                    axis = self.elliptical_clearance_axis(rel_vel)
                    clearance = math.sqrt((closest[0] / axis[0]) ** 2 + (closest[1] / axis[1]) ** 2)
                    required = 1.0
                else:
                    clearance = np.linalg.norm(closest)
                    required = self.radius * 2.25
                if clearance < required:
                    penalty += (required - clearance) ** 2 * (34.0 / (ttc + 0.45))
            if dist_now < self.radius * 2.2:
                penalty += (self.radius * 2.2 - dist_now) ** 2 * 90.0
        return penalty

    def obstacle_penalty(self, agent, predicted):
        penalty = 0.0
        for obstacle in self.obstacles:
            center, radius = self.obstacle_circle_approx(obstacle)
            dist = np.linalg.norm(predicted - center)
            limit = radius + self.radius + 0.38
            if dist < limit:
                penalty += (limit - dist) ** 2 * 160.0
            current_dist = np.linalg.norm(agent.pos - center)
            if current_dist < limit + 2.8 and dist < current_dist:
                penalty += (limit + 2.8 - dist) * 1.3
        return penalty

    @staticmethod
    def time_to_closest(rel_pos, rel_vel):
        denom = float(np.dot(rel_vel, rel_vel))
        if denom <= 1e-8:
            return 0.0
        return -float(np.dot(rel_pos, rel_vel)) / denom

    @staticmethod
    def elliptical_clearance_axis(rel_vel):
        speed = np.linalg.norm(rel_vel)
        return (1.75 + 0.32 * speed, 1.12)

    def side_preference(self, agent, center, step):
        if self.mode == "vo":
            return 1.0 if (agent.id + step // 12) % 2 == 0 else -1.0
        if self.mode in {"rvo", "orca"}:
            return 1.0 if agent.group == "L" else -1.0
        if self.mode == "hrvo":
            return 1.0 if agent.pos[1] <= center[1] else -1.0
        if self.mode == "porca":
            return 1.0 if agent.group == "L" else -1.0
        return 1.0 if agent.group == "L" else -1.0

    def side_lane_penalty(self, agent, velocity, strength):
        preferred = 0.32 if agent.group == "L" else -0.32
        return abs(velocity[1] - preferred) * strength

    @staticmethod
    def abrupt_change_penalty(agent, velocity, strength):
        return np.linalg.norm(velocity - agent.vel) * strength

    def preferred_speed(self, agent):
        if self.mode == "porca":
            return self.pref_speed * (0.92 + 0.04 * (agent.id % 3))
        if self.mode == "ervo":
            return self.pref_speed * 1.04
        return self.pref_speed

    def obstacle_circle_approx(self, obstacle):
        kind, data = obstacle
        if kind == "circle":
            x, y, radius = data
            return np.array([x, y], dtype=float), radius
        x, y, w, h = data
        return np.array([x + w * 0.5, y + h * 0.5], dtype=float), math.hypot(w, h) * 0.5

    @staticmethod
    def cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def boundary_penalty(point):
        x, y = point
        penalty = 0.0
        for value, low, high in ((x, 1.4, 48.6), (y, 1.4, 28.6)):
            if value < low:
                penalty += (low - value) ** 2 * 120.0
            if value > high:
                penalty += (value - high) ** 2 * 120.0
        return penalty

    def reached_goal(self, agent):
        return np.linalg.norm(agent.goal - agent.pos) < 0.72

    def all_reached(self):
        return all(self.reached_goal(agent) for agent in self.agents)

    def update_min_separation(self):
        for i, agent in enumerate(self.agents):
            for other in self.agents[i + 1:]:
                self.min_separation = min(self.min_separation, np.linalg.norm(agent.pos - other.pos))

    def resolve_overlaps(self):
        target_gap = self.radius * 2.08
        for _ in range(5):
            for i, agent in enumerate(self.agents):
                for other in self.agents[i + 1:]:
                    delta = other.pos - agent.pos
                    dist = np.linalg.norm(delta)
                    if dist < 1e-6:
                        direction = np.array([1.0, 0.0])
                        dist = 1e-6
                    else:
                        direction = delta / dist
                    if dist < target_gap:
                        push = (target_gap - dist) * 0.5
                        agent.pos -= direction * push
                        other.pos += direction * push
            for agent in self.agents:
                self.resolve_obstacle_overlap(agent)
                agent.pos[0] = float(np.clip(agent.pos[0], 1.6, 48.4))
                agent.pos[1] = float(np.clip(agent.pos[1], 1.6, 28.4))

    def resolve_obstacle_overlap(self, agent):
        for kind, data in self.obstacles:
            if kind == "circle":
                center = np.array(data[:2], dtype=float)
                limit = data[2] + self.radius + 0.12
                delta = agent.pos - center
                dist = np.linalg.norm(delta)
                if dist < limit:
                    direction = delta / dist if dist > 1e-6 else np.array([0.0, 1.0])
                    agent.pos = center + direction * limit
            else:
                x, y, w, h = data
                xmin, xmax = x - self.radius - 0.1, x + w + self.radius + 0.1
                ymin, ymax = y - self.radius - 0.1, y + h + self.radius + 0.1
                if xmin < agent.pos[0] < xmax and ymin < agent.pos[1] < ymax:
                    distances = [
                        (abs(agent.pos[0] - xmin), np.array([xmin, agent.pos[1]])),
                        (abs(agent.pos[0] - xmax), np.array([xmax, agent.pos[1]])),
                        (abs(agent.pos[1] - ymin), np.array([agent.pos[0], ymin])),
                        (abs(agent.pos[1] - ymax), np.array([agent.pos[0], ymax])),
                    ]
                    agent.pos = min(distances, key=lambda item: item[0])[1]

    def phase_text(self, step):
        if self.mode == "vo":
            return "VO: each agent avoids predicted velocity obstacles independently"
        if self.mode == "rvo":
            return "RVO: agents split collision responsibility reciprocally"
        if self.mode == "hrvo":
            return "HRVO: reciprocal avoidance uses hybrid side selection"
        if self.mode == "orca":
            return "ORCA: candidate velocities approximate reciprocal half-plane constraints"
        if self.mode == "porca":
            return "PORCA: pedestrian-like side preference and speed modulation"
        return "ERVO: elliptical velocity obstacles react earlier to head-on motion"

    def snapshot(self, step, phase, final=False):
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "positions": [agent.pos.copy() for agent in self.agents],
            "velocities": [agent.vel.copy() for agent in self.agents],
            "trails": [list(agent.trail) for agent in self.agents],
            "min_sep": self.min_separation if self.min_separation < math.inf else 0.0,
        }

    def save_gif(self, gif_name, max_frames=54):
        frames = [self.render_snapshot(snapshot) for snapshot in self.select_snapshots(self.snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 5)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=220, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=110)
        self.draw_environment(ax)
        for agent, trail in zip(self.agents, snapshot["trails"]):
            if len(trail) > 1:
                xs = [p[0] for p in trail]
                ys = [p[1] for p in trail]
                ax.plot(xs, ys, color=agent.color, linewidth=1.8, alpha=0.55, zorder=3)
        for agent, pos, vel in zip(self.agents, snapshot["positions"], snapshot["velocities"]):
            ax.add_patch(patches.Circle(pos, self.radius, facecolor=agent.color, edgecolor="#111827", linewidth=0.7, alpha=0.95, zorder=5))
            ax.arrow(pos[0], pos[1], vel[0] * 0.65, vel[1] * 0.65, color="#111827", width=0.025, head_width=0.28, alpha=0.55, zorder=6)
            ax.scatter(agent.goal[0], agent.goal[1], marker="x", s=34, color=agent.color, linewidths=1.6, zorder=4)
        ax.text(
            1.4,
            28.0,
            f"{self.label}  step {snapshot['step']:3d}  min sep {snapshot['min_sep']:.2f}\n{snapshot['phase']}",
            fontsize=8.3,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.label)
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
        for kind, data in self.obstacles:
            if kind == "circle":
                x, y, radius = data
                ax.add_patch(patches.Circle((x, y), radius, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=2))
            else:
                x, y, w, h = data
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=2))
        for y in [6.4, 10.6, 14.8, 19.0, 23.2]:
            ax.plot([3.5, 46.5], [y, y], color="#cbd5e1", linewidth=0.7, alpha=0.65, zorder=1)
