"""Shared convex-set world for multi-agent GCS style demos."""

from __future__ import annotations

import heapq
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from Search_2D.remaining_algorithm_helpers import COLORS, polyline_length


class ConvexRegion:
    def __init__(self, region_id, xywh):
        self.id = region_id
        self.x, self.y, self.w, self.h = xywh
        self.center = (self.x + self.w * 0.5, self.y + self.h * 0.5)

    def contains(self, point):
        return self.x <= point[0] <= self.x + self.w and self.y <= point[1] <= self.y + self.h

    def patch(self, **kwargs):
        return patches.Rectangle((self.x, self.y), self.w, self.h, **kwargs)


class ConvexMultiAgentWorld:
    def __init__(self):
        specs = [
            (2, 3, 10, 7), (14, 3, 10, 7), (26, 3, 10, 7), (38, 3, 10, 7),
            (2, 12, 10, 6), (14, 12, 10, 6), (26, 12, 10, 6), (38, 12, 10, 6),
            (2, 21, 10, 6), (14, 21, 10, 6), (26, 21, 10, 6), (38, 21, 10, 6),
        ]
        self.regions = [ConvexRegion(i, spec) for i, spec in enumerate(specs)]
        self.obstacles = [
            ("rect", (20.5, 9.3, 7.5, 3.0)),
            ("rect", (23.0, 18.0, 6.0, 3.0)),
            ("circle", (34.0, 15.0, 2.4)),
        ]
        self.graph = {region.id: [] for region in self.regions}
        self.build_graph()

    def build_graph(self):
        for a in self.regions:
            for b in self.regions:
                if a.id >= b.id:
                    continue
                if self.adjacent(a, b):
                    cost = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                    self.graph[a.id].append((b.id, cost))
                    self.graph[b.id].append((a.id, cost))

    @staticmethod
    def adjacent(a, b):
        horizontal_touch = abs((a.x + a.w) - b.x) < 2.5 or abs((b.x + b.w) - a.x) < 2.5
        vertical_overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y) > 1.5
        vertical_touch = abs((a.y + a.h) - b.y) < 3.5 or abs((b.y + b.h) - a.y) < 3.5
        horizontal_overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x) > 1.5
        return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap)

    def locate_region(self, point):
        for region in self.regions:
            if region.contains(point):
                return region.id
        return min(self.regions, key=lambda r: math.hypot(r.center[0] - point[0], r.center[1] - point[1])).id

    def shortest_region_path(self, start_region, goal_region, region_penalty=None):
        region_penalty = region_penalty or {}
        open_set = [(0.0, start_region)]
        parent = {start_region: None}
        cost = {start_region: 0.0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_region:
                break
            for nxt, edge_cost in self.graph[current]:
                new_cost = cost[current] + edge_cost + region_penalty.get(nxt, 0.0)
                if new_cost < cost.get(nxt, float("inf")):
                    cost[nxt] = new_cost
                    parent[nxt] = current
                    h = math.hypot(
                        self.regions[nxt].center[0] - self.regions[goal_region].center[0],
                        self.regions[nxt].center[1] - self.regions[goal_region].center[1],
                    )
                    heapq.heappush(open_set, (new_cost + h, nxt))
        if goal_region not in parent:
            return [start_region]
        path = []
        current = goal_region
        while current is not None:
            path.append(current)
            current = parent[current]
        return list(reversed(path))

    def route_points(self, start, goal, region_path):
        points = [start]
        points.extend(self.regions[rid].center for rid in region_path[1:-1])
        points.append(goal)
        return points

    def route_risk(self, points):
        risk = 0.0
        for point in points:
            for kind, data in self.obstacles:
                if kind == "circle":
                    ox, oy, radius = data
                    dist = math.hypot(point[0] - ox, point[1] - oy) - radius
                else:
                    ox, oy, w, h = data
                    cx = min(max(point[0], ox), ox + w)
                    cy = min(max(point[1], oy), oy + h)
                    dist = math.hypot(point[0] - cx, point[1] - cy)
                risk += 1.0 / max(dist, 0.6)
        return risk

    def draw(self, ax, title):
        ax.add_patch(patches.Rectangle((0, 0), 50, 30, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.0))
        for region in self.regions:
            ax.add_patch(region.patch(facecolor="#dbeafe", edgecolor="#60a5fa", linewidth=1.0, alpha=0.42))
            ax.text(region.center[0], region.center[1], str(region.id), ha="center", va="center", fontsize=7, color="#1e3a8a", alpha=0.8)
        for region_id, edges in self.graph.items():
            a = self.regions[region_id].center
            for nxt, _ in edges:
                if region_id < nxt:
                    b = self.regions[nxt].center
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="#93c5fd", linewidth=0.8, alpha=0.5)
        for kind, data in self.obstacles:
            if kind == "circle":
                x, y, radius = data
                ax.add_patch(patches.Circle((x, y), radius, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=3))
            else:
                x, y, w, h = data
                ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#334155", edgecolor="#111827", linewidth=1.0, zorder=3))
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)


def draw_agent_routes(ax, starts, goals, routes, upto=1.0):
    for index, points in enumerate(routes):
        color = COLORS[index % len(COLORS)]
        count = max(2, min(len(points), int(math.ceil(len(points) * upto))))
        visible = points[:count]
        ax.plot([p[0] for p in visible], [p[1] for p in visible], "o-", color=color, linewidth=2.4, markersize=4, zorder=5)
        ax.scatter([starts[index][0]], [starts[index][1]], marker="s", s=48, color=color, edgecolor="#111827", zorder=6)
        ax.scatter([goals[index][0]], [goals[index][1]], marker="*", s=92, color=color, edgecolor="#111827", zorder=6)


def total_route_length(routes):
    return sum(polyline_length(route) for route in routes)
