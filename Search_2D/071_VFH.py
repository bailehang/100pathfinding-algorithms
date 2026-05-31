"""Vector Field Histogram (VFH) reactive local planning demo."""

from metrics import install_metrics, now_ms, print_metrics_for

install_metrics()

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_2D.remaining_algorithm_helpers import GifRecorder, GridTools, polyline_length


class VectorFieldHistogram:
    def __init__(self, start=(5.0, 5.0), goal=(45.0, 25.0)):
        self.grid = GridTools()
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.position = self.start.copy()
        self.step_size = 0.82
        self.sensor_range = 8.5
        self.sector_count = 72
        self.free_threshold = 0.72
        self.path = [tuple(self.position)]
        self.snapshots = []
        self.recorder = GifRecorder()
        self.raw_guide = self.grid.astar(tuple(map(int, start)), tuple(map(int, goal)))
        self.guide = [np.array(p, dtype=float) for p in self.raw_guide]
        if not np.allclose(self.guide[-1], self.goal):
            self.guide.append(self.goal.copy())

    def planning(self, save_gif=False):
        start_ms = now_ms()
        step = 0
        for waypoint in self.guide[1:]:
            local_goal = waypoint
            while np.linalg.norm(self.position - local_goal) > 0.48 and step < 320:
                histogram = self.build_histogram()
                target_angle = math.atan2(local_goal[1] - self.position[1], local_goal[0] - self.position[0])
                chosen_sector = self.choose_sector(histogram, target_angle)
                heading = self.sector_angle(chosen_sector)
                candidate = self.position + np.array([math.cos(heading), math.sin(heading)]) * self.step_size
                guided_candidate = self.guided_step(local_goal)
                target_sector = self.angle_sector(target_angle)
                if self.is_free(guided_candidate):
                    candidate = guided_candidate
                    chosen_sector = target_sector
                if not self.is_free(candidate):
                    candidate = self.recovery_step(histogram, target_angle)
                    heading = math.atan2(candidate[1] - self.position[1], candidate[0] - self.position[0])
                self.position = candidate
                self.path.append(tuple(self.position))
                if step < 28 or step % 4 == 0:
                    self.snapshots.append((step, self.position.copy(), local_goal.copy(), histogram.copy(), chosen_sector))
                step += 1
            if np.linalg.norm(self.position - self.goal) < 0.75:
                break

        if np.linalg.norm(self.position - self.goal) < 1.0:
            self.path.append(tuple(self.goal))
        elapsed = now_ms() - start_ms
        print_metrics_for(self.path, elapsed, source="vfh")
        if save_gif:
            self.save_gif()
        return self.path

    def closest_guide_index(self, start_index):
        end_index = min(len(self.guide), start_index + 35)
        window = self.guide[start_index:end_index]
        if not window:
            return len(self.guide) - 1
        distances = [np.linalg.norm(self.position - point) for point in window]
        return start_index + int(np.argmin(distances))

    def guided_step(self, local_goal):
        delta = local_goal - self.position
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return self.position.copy()
        return self.position + delta / norm * self.step_size

    def build_histogram(self):
        histogram = np.zeros(self.sector_count, dtype=float)
        for ox, oy in self.grid.obs:
            delta = np.array([ox, oy], dtype=float) - self.position
            dist = np.linalg.norm(delta)
            if dist < 1e-6 or dist > self.sensor_range:
                continue
            angle = math.atan2(delta[1], delta[0])
            sector = int(((angle + math.pi) / (2.0 * math.pi)) * self.sector_count) % self.sector_count
            width = max(1, int((1.3 / max(dist, 0.8)) * self.sector_count / (2.0 * math.pi)))
            strength = ((self.sensor_range - dist) / self.sensor_range) ** 2
            for offset in range(-width, width + 1):
                histogram[(sector + offset) % self.sector_count] += strength * (1.0 - abs(offset) / (width + 1.0))
        return np.clip(histogram, 0.0, 3.0)

    def choose_sector(self, histogram, target_angle):
        target_sector = self.angle_sector(target_angle)
        best_sector = target_sector
        best_score = float("inf")
        for sector, density in enumerate(histogram):
            circular = min(abs(sector - target_sector), self.sector_count - abs(sector - target_sector))
            penalty = 0.0 if density < self.free_threshold else 30.0 * density
            score = circular + penalty + density * 2.5
            if score < best_score:
                best_score = score
                best_sector = sector
        return best_sector

    def recovery_step(self, histogram, target_angle):
        target_sector = self.angle_sector(target_angle)
        sectors = sorted(range(self.sector_count), key=lambda s: (histogram[s], min(abs(s - target_sector), self.sector_count - abs(s - target_sector))))
        for sector in sectors[:18]:
            heading = self.sector_angle(sector)
            candidate = self.position + np.array([math.cos(heading), math.sin(heading)]) * self.step_size * 0.65
            if self.is_free(candidate):
                return candidate
        return self.position.copy()

    def is_free(self, point):
        node = (int(round(point[0])), int(round(point[1])))
        return self.grid.is_valid(node)

    def angle_sector(self, angle):
        return int(((angle + math.pi) / (2.0 * math.pi)) * self.sector_count) % self.sector_count

    def sector_angle(self, sector):
        return (sector + 0.5) / self.sector_count * 2.0 * math.pi - math.pi

    def save_gif(self):
        for step, pos, local_goal, histogram, chosen_sector in self.snapshots:
            fig, ax = plt.subplots(figsize=self.recorder.figsize, dpi=self.recorder.dpi)
            self.grid.draw_grid(ax, f"071 VFH - polar obstacle histogram step {step}")
            ax.plot([p[0] for p in self.raw_guide], [p[1] for p in self.raw_guide], color="#60a5fa", linewidth=1.6, alpha=0.35, label="A* guide")
            history = self.path[: min(len(self.path), step + 2)]
            ax.plot([p[0] for p in history], [p[1] for p in history], color="#dc2626", linewidth=2.5, label="VFH path")
            ax.scatter([self.start[0]], [self.start[1]], marker="s", s=42, c="#2563eb", label="start")
            ax.scatter([self.goal[0]], [self.goal[1]], marker="*", s=95, c="#16a34a", label="goal")
            ax.scatter([local_goal[0]], [local_goal[1]], marker="o", s=42, c="#f97316", label="local target")
            heading = self.sector_angle(chosen_sector)
            ax.arrow(pos[0], pos[1], math.cos(heading) * 2.2, math.sin(heading) * 2.2, color="#111827", width=0.04, head_width=0.45)
            ax.legend(loc="upper left", fontsize=7)

            inset = ax.inset_axes([0.60, 0.06, 0.34, 0.26])
            sectors = np.arange(self.sector_count)
            colors = ["#cbd5e1"] * self.sector_count
            colors[chosen_sector] = "#dc2626"
            inset.bar(sectors, histogram, color=colors, width=1.0)
            inset.axhline(self.free_threshold, color="#111827", linewidth=0.8, alpha=0.6)
            inset.set_title("polar density", fontsize=8)
            inset.set_xticks([])
            inset.tick_params(labelsize=7)
            self.recorder.capture()
            plt.close(fig)
        self.recorder.save("071_VFH", fps=5)
        print(f"VFH path length: {polyline_length(self.path):.3f}")


def main():
    path = VectorFieldHistogram().planning(save_gif=True)
    if len(path) < 2:
        raise RuntimeError("VFH failed to produce a path")


if __name__ == "__main__":
    main()
