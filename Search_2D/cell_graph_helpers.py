import heapq
import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay


class PrecomputedCellGraph:
    def __init__(self, mode, label):
        self.mode = mode
        self.label = label
        self.cells = {}
        self.edges = {}
        self.start = None
        self.goal = None
        self.open_heap = []
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path = []
        self.region_cells = {}
        self.region_centers = {}
        self.region_edges = {}
        self.changed_cells = set()
        self.obstacles = []
        self.navmesh_points = []
        self.navmesh_triangles = []
        self.build()

    def build(self):
        if self.mode == "quad":
            self.build_quad_cells(dynamic=False)
            self.start, self.goal = "q_0_1", "q_11_6"
        elif self.mode == "hex":
            self.build_hex_cells()
            self.start, self.goal = "h_0_2", "h_11_5"
        elif self.mode == "navmesh":
            self.build_navmesh_cells()
        elif self.mode == "hier":
            self.build_hierarchical_cells()
            self.start, self.goal = "r_0_1", "r_11_6"
        elif self.mode == "dynamic":
            self.build_quad_cells(dynamic=True)
            self.start, self.goal = "q_0_1", "q_11_6"
        self.build_edges()
        if self.mode == "hier":
            self.build_region_graph()

    def build_quad_cells(self, dynamic=False):
        blocked = {(3, 2), (3, 3), (4, 3), (7, 4), (8, 4), (8, 5)}
        if dynamic:
            blocked = {(3, 3), (4, 3), (8, 5)}
        width, height = 3.7, 3.0
        x0, y0 = 3.0, 3.2
        for col in range(12):
            for row in range(8):
                center = (x0 + col * width, y0 + row * height)
                self.add_cell(
                    f"q_{col}_{row}",
                    center,
                    self.box(center, width, height),
                    "quad",
                    (col, row) in blocked,
                    coord=(col, row),
                )

    def build_hex_cells(self):
        blocked = {(3, 2), (4, 2), (5, 3), (7, 4), (8, 4)}
        radius = 1.78
        dx = 1.5 * radius
        dy = math.sqrt(3.0) * radius
        for col in range(12):
            for row in range(7):
                center = (4.0 + col * dx, 4.0 + (row + 0.5 * (col % 2)) * dy)
                self.add_cell(
                    f"h_{col}_{row}",
                    center,
                    self.hexagon(center, radius),
                    "hex",
                    (col, row) in blocked,
                    coord=(col, row),
                )

    def build_navmesh_cells(self):
        self.navmesh_bounds = (2.0, 47.0, 3.0, 25.0)
        self.obstacles = [
            {"center": (16.4, 14.0), "radius": 3.15},
            {"center": (31.0, 13.0), "radius": 3.45},
        ]
        points, tags = self.navmesh_sample_points()
        self.navmesh_points = points
        triangles = self.filtered_delaunay_triangles(points, tags)
        self.navmesh_triangles = [[tuple(points[index]) for index in tri] for tri in triangles]
        merged = self.merge_navmesh_triangles(self.navmesh_triangles)
        for index, polygon in enumerate(merged):
            self.add_cell(f"n_{index}", self.centroid(polygon), polygon, "navmesh")
        self.start = self.pick_cell_containing((4.8, 8.6))
        self.goal = self.pick_cell_containing((43.8, 19.7))

    def navmesh_sample_points(self):
        xmin, xmax, ymin, ymax = self.navmesh_bounds
        points = []
        tags = []

        def add(point, tag):
            if any(self.same_point(point, existing) for existing in points):
                return
            points.append(point)
            tags.append(tag)

        for x in np.linspace(xmin, xmax, 10):
            add((float(x), ymin), ("outer", 0))
            add((float(x), ymax), ("outer", 1))
        for y in np.linspace(ymin, ymax, 6):
            add((xmin, float(y)), ("outer", 2))
            add((xmax, float(y)), ("outer", 3))
        for x in np.linspace(xmin + 4.5, xmax - 4.5, 8):
            for y in np.linspace(ymin + 3.0, ymax - 3.0, 5):
                point = (float(x), float(y))
                if not self.inside_any_circle(point, margin=1.15):
                    add(point, ("field", 0))
        for obstacle_index, obstacle in enumerate(self.obstacles):
            cx, cy = obstacle["center"]
            radius = obstacle["radius"]
            for sample_index in range(18):
                angle = 2.0 * math.pi * sample_index / 18.0
                add(
                    (cx + radius * math.cos(angle), cy + radius * math.sin(angle)),
                    ("circle", obstacle_index, sample_index, 18),
                )
        return np.array(points, dtype=float), tags

    def filtered_delaunay_triangles(self, points, tags):
        delaunay = Delaunay(points)
        triangles = []
        for tri in delaunay.simplices:
            polygon = [tuple(points[index]) for index in tri]
            if abs(self.polygon_area(polygon)) < 0.1:
                continue
            if self.triangle_hits_obstacle(tri, points, tags):
                continue
            triangles.append(tuple(tri))
        return triangles

    def triangle_hits_obstacle(self, tri, points, tags):
        polygon = [tuple(points[index]) for index in tri]
        center = self.centroid(polygon)
        if self.inside_any_circle(center, margin=0.98):
            return True
        for local_index in range(3):
            a_index = tri[local_index]
            b_index = tri[(local_index + 1) % 3]
            a = tuple(points[a_index])
            b = tuple(points[b_index])
            for obstacle_index, obstacle in enumerate(self.obstacles):
                if self.is_circle_boundary_edge(tags[a_index], tags[b_index], obstacle_index):
                    continue
                if self.distance_point_to_segment(obstacle["center"], a, b) < obstacle["radius"] * 0.985:
                    return True
        return False

    @staticmethod
    def is_circle_boundary_edge(a_tag, b_tag, obstacle_index):
        if a_tag[0] != "circle" or b_tag[0] != "circle":
            return False
        if a_tag[1] != obstacle_index or b_tag[1] != obstacle_index:
            return False
        samples = a_tag[3]
        delta = abs(a_tag[2] - b_tag[2])
        return delta == 1 or delta == samples - 1

    def merge_navmesh_triangles(self, triangles):
        polygons = [
            {"points": self.sort_polygon(triangle), "area": abs(self.polygon_area(triangle))}
            for triangle in triangles
        ]
        changed = True
        while changed:
            changed = False
            for left in range(len(polygons)):
                if changed:
                    break
                for right in range(left + 1, len(polygons)):
                    merged = self.try_merge_polygons(polygons[left], polygons[right])
                    if not merged:
                        continue
                    polygons[left] = merged
                    del polygons[right]
                    changed = True
                    break
        return [polygon["points"] for polygon in polygons]

    def try_merge_polygons(self, left, right):
        if len(self.shared_vertex_keys(left["points"], right["points"])) < 2:
            return None
        unique_points = []
        seen = set()
        for point in left["points"] + right["points"]:
            key = self.point_key(point)
            if key in seen:
                continue
            seen.add(key)
            unique_points.append(point)
        if len(unique_points) > 6:
            return None
        ordered = self.sort_polygon(unique_points)
        area = abs(self.polygon_area(ordered))
        if not self.is_convex_polygon(ordered):
            return None
        if abs(area - left["area"] - right["area"]) > 0.04:
            return None
        return {"points": ordered, "area": area}

    def pick_cell_containing(self, point):
        for cell_id, cell in self.cells.items():
            if self.point_in_polygon(point, cell["polygon"]):
                return cell_id
        return min(self.cells, key=lambda cell_id: math.hypot(self.cells[cell_id]["center"][0] - point[0], self.cells[cell_id]["center"][1] - point[1]))

    def build_hierarchical_cells(self):
        blocked = {(3, 3), (4, 3), (5, 3), (6, 1), (7, 4), (8, 4), (8, 5)}
        width, height = 3.7, 3.0
        x0, y0 = 3.0, 3.2
        for col in range(12):
            for row in range(8):
                center = (x0 + col * width, y0 + row * height)
                cluster = (col // 3, row // 2)
                self.add_cell(
                    f"r_{col}_{row}",
                    center,
                    self.box(center, width, height),
                    "quad",
                    (col, row) in blocked,
                    coord=(col, row),
                    cluster=cluster,
                )

    def add_cell(self, cell_id, center, polygon, kind, blocked=False, coord=None, cluster=None, cost=1.0):
        self.cells[cell_id] = {
            "center": center,
            "polygon": polygon,
            "kind": kind,
            "blocked": blocked,
            "coord": coord,
            "cluster": cluster,
            "cost": cost,
        }

    def build_edges(self):
        self.edges = {cell_id: [] for cell_id in self.cells}
        if self.mode == "hex":
            self.build_hex_edges()
        elif self.mode == "navmesh":
            self.build_navmesh_edges()
        else:
            self.build_grid_edges()

    def build_grid_edges(self):
        by_coord = {cell["coord"]: cell_id for cell_id, cell in self.cells.items()}
        for cell_id, cell in self.cells.items():
            col, row = cell["coord"]
            for coord in ((col + 1, row), (col, row + 1)):
                neighbor = by_coord.get(coord)
                if neighbor:
                    self.connect(cell_id, neighbor)

    def build_hex_edges(self):
        by_coord = {cell["coord"]: cell_id for cell_id, cell in self.cells.items()}
        for cell_id, cell in self.cells.items():
            col, row = cell["coord"]
            if col % 2:
                neighbor_coords = ((col + 1, row), (col + 1, row + 1), (col, row + 1))
            else:
                neighbor_coords = ((col + 1, row - 1), (col + 1, row), (col, row + 1))
            for coord in neighbor_coords:
                neighbor = by_coord.get(coord)
                if neighbor:
                    self.connect(cell_id, neighbor)

    def build_navmesh_edges(self):
        portal_index = {}
        for cell_id, cell in self.cells.items():
            for edge in self.polygon_edges(cell["polygon"]):
                portal_index.setdefault(self.edge_key(edge), []).append(cell_id)
        for shared_cells in portal_index.values():
            if len(shared_cells) < 2:
                continue
            for i, a in enumerate(shared_cells):
                for b in shared_cells[i + 1:]:
                    self.connect(a, b)

    def connect(self, a, b):
        if self.cells[a]["blocked"] or self.cells[b]["blocked"]:
            return
        if any(neighbor == b for neighbor, _ in self.edges[a]):
            return
        cost = self.distance(a, b) * 0.5 * (self.cells[a]["cost"] + self.cells[b]["cost"])
        self.edges[a].append((b, cost))
        self.edges[b].append((a, cost))

    def build_region_graph(self):
        self.region_cells = {}
        for cell_id, cell in self.cells.items():
            if cell["blocked"]:
                continue
            self.region_cells.setdefault(cell["cluster"], []).append(cell_id)
        self.region_centers = {
            region: self.average_center(cells)
            for region, cells in self.region_cells.items()
        }
        self.region_edges = {region: set() for region in self.region_cells}
        for cell_id, neighbors in self.edges.items():
            region = self.cells[cell_id]["cluster"]
            for neighbor, _ in neighbors:
                other = self.cells[neighbor]["cluster"]
                if region != other:
                    self.region_edges[region].add(other)
                    self.region_edges[other].add(region)

    def search(self, save_gif=False, gif_name="cell_graph"):
        if self.mode == "hier":
            path, snapshots = self.search_hierarchical()
        elif self.mode == "dynamic":
            path, snapshots = self.search_dynamic()
        elif self.mode == "navmesh":
            path, snapshots = self.search_navmesh()
        else:
            path, snapshots = self.search_cells(initial_phase="precomputed cells loaded")
        if not path:
            raise RuntimeError(f"{self.label} did not reach the goal")
        if save_gif:
            self.save_gif(snapshots, gif_name)
        return [self.cells[c]["center"] for c in path]

    def search_cells(self, initial_phase, allowed_regions=None, old_path=None):
        self.open_heap = [(self.heuristic(self.start), 0.0, self.start)]
        self.open_set = {self.start}
        self.closed = []
        self.closed_set = set()
        self.parent = {self.start: self.start}
        self.g = {self.start: 0.0}
        self.path = []
        snapshots = [self.snapshot(0, initial_phase, old_path=old_path)]

        for step in range(1, 700):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            if current == self.goal:
                self.path = self.extract_path(current)
                snapshots.append(self.snapshot(step, "goal cell reached", final=True, old_path=old_path))
                break
            for neighbor, cost in self.edges[current]:
                if allowed_regions and self.cells[neighbor]["cluster"] not in allowed_regions:
                    continue
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 30 or step % 8 == 0:
                snapshots.append(self.snapshot(step, self.phase_text(), old_path=old_path))

        if self.path:
            snapshots.append(self.snapshot(len(self.closed), "final cell route", final=True, old_path=old_path))
        return self.path, snapshots

    def search_navmesh(self):
        self.open_set = set()
        self.closed = []
        self.closed_set = set()
        self.parent = {}
        self.g = {}
        self.path = []
        process = [
            self.snapshot(0, "Delaunay triangles generated around circular obstacles", show_triangles=True),
            self.snapshot(0, "adjacent triangles merged into 3-6 sided convex polygons"),
        ]
        path, snapshots = self.search_cells(initial_phase="search merged convex navmesh polygons")
        return path, process + snapshots

    def search_hierarchical(self):
        start_region = self.cells[self.start]["cluster"]
        goal_region = self.cells[self.goal]["cluster"]
        region_path, region_snapshots = self.search_regions(start_region, goal_region)
        allowed_regions = set(region_path)
        snapshots = [
            self.snapshot(0, "cells aggregated into cluster/region graph", region_path=region_path),
            *region_snapshots,
        ]
        self.open_heap = [(self.heuristic(self.start), 0.0, self.start)]
        self.open_set = {self.start}
        self.closed = []
        self.closed_set = set()
        self.parent = {self.start: self.start}
        self.g = {self.start: 0.0}
        self.path = []

        for step in range(1, 700):
            if not self.open_heap:
                break
            _, _, current = heapq.heappop(self.open_heap)
            if current in self.closed_set:
                continue
            self.open_set.discard(current)
            self.closed_set.add(current)
            self.closed.append(current)
            if current == self.goal:
                self.path = self.extract_path(current)
                snapshots.append(self.snapshot(step, "fine cell search reaches goal inside region corridor", final=True, region_path=region_path))
                break
            for neighbor, cost in self.edges[current]:
                if self.cells[neighbor]["cluster"] not in allowed_regions:
                    continue
                new_cost = self.g[current] + cost
                if new_cost + 1e-9 >= self.g.get(neighbor, math.inf):
                    continue
                self.g[neighbor] = new_cost
                self.parent[neighbor] = current
                self.open_set.add(neighbor)
                heapq.heappush(self.open_heap, (new_cost + self.heuristic(neighbor), new_cost, neighbor))
            if step < 30 or step % 8 == 0:
                snapshots.append(self.snapshot(step, "fine search is limited to coarse region route", region_path=region_path))
        snapshots.append(self.snapshot(len(self.closed), "hierarchical route: region path plus cell path", final=True, region_path=region_path))
        return self.path, snapshots

    def search_regions(self, start_region, goal_region):
        open_heap = [(self.region_heuristic(start_region, goal_region), 0.0, start_region)]
        open_set = {start_region}
        closed = []
        closed_set = set()
        parent = {start_region: start_region}
        g = {start_region: 0.0}
        snapshots = []
        path = []
        for step in range(1, 80):
            if not open_heap:
                break
            _, _, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            open_set.discard(current)
            closed_set.add(current)
            closed.append(current)
            if current == goal_region:
                path = self.extract_region_path(current, parent)
                snapshots.append(self.snapshot(step, "coarse A* reached goal region", region_closed=closed, region_open=open_set, region_path=path))
                break
            for neighbor in self.region_edges[current]:
                new_cost = g[current] + self.region_heuristic(current, neighbor)
                if new_cost + 1e-9 >= g.get(neighbor, math.inf):
                    continue
                g[neighbor] = new_cost
                parent[neighbor] = current
                open_set.add(neighbor)
                heapq.heappush(open_heap, (new_cost + self.region_heuristic(neighbor, goal_region), new_cost, neighbor))
            snapshots.append(self.snapshot(step, "coarse A* searches cluster/region graph", region_closed=closed, region_open=open_set))
        return path, snapshots

    def search_dynamic(self):
        first_path, snapshots = self.search_cells(initial_phase="precomputed cell graph loaded")
        old_path = list(first_path)
        self.apply_dynamic_changes(old_path)
        snapshots.append(self.snapshot(len(snapshots), "blocked/cost changed: invalidate affected path suffix", old_path=old_path))
        self.build_edges()
        self.start, self.goal = "q_0_1", "q_11_6"
        repaired_path, repair_snapshots = self.search_cells(
            initial_phase="incremental repair reuses the same precomputed cells",
            old_path=old_path,
        )
        snapshots.extend(repair_snapshots[1:])
        snapshots.append(self.snapshot(len(snapshots), "repaired route avoids changed cells", final=True, old_path=old_path))
        return repaired_path, snapshots

    def apply_dynamic_changes(self, old_path):
        blocked = [cell_id for cell_id in old_path[4:7] if cell_id not in (self.start, self.goal)]
        costly = [cell_id for cell_id in old_path[7:9] if cell_id not in (self.start, self.goal)]
        self.changed_cells = set(blocked + costly)
        for cell_id in blocked:
            self.cells[cell_id]["blocked"] = True
        for cell_id in costly:
            self.cells[cell_id]["cost"] = 4.2

    def phase_text(self):
        if self.mode == "quad":
            return "A* follows shared-edge quadrilateral cells"
        if self.mode == "hex":
            return "A* follows six-neighbor hexagonal cells"
        if self.mode == "navmesh":
            return "A* crosses 3-6 sided convex navmesh polygons through portals"
        return "dynamic repair expands around changed blocked/cost cells"

    def heuristic(self, cell_id):
        ax, ay = self.cells[cell_id]["center"]
        bx, by = self.cells[self.goal]["center"]
        return math.hypot(ax - bx, ay - by)

    def region_heuristic(self, a, b):
        ax, ay = self.region_centers[a]
        bx, by = self.region_centers[b]
        return math.hypot(ax - bx, ay - by)

    def extract_path(self, cell_id):
        path = []
        current = cell_id
        seen = set()
        while current in self.parent and current not in seen:
            seen.add(current)
            path.append(current)
            parent = self.parent[current]
            if parent == current:
                break
            current = parent
        return list(reversed(path))

    @staticmethod
    def extract_region_path(region, parent):
        path = []
        current = region
        seen = set()
        while current in parent and current not in seen:
            seen.add(current)
            path.append(current)
            previous = parent[current]
            if previous == current:
                break
            current = previous
        return list(reversed(path))

    def snapshot(self, step, phase, final=False, old_path=None, region_open=None, region_closed=None, region_path=None, show_triangles=False):
        hint = self.extract_path(self.closed[-1]) if self.closed else []
        return {
            "step": step,
            "phase": phase,
            "final": final,
            "open": list(self.open_set),
            "closed": list(self.closed),
            "hint": hint,
            "path": list(self.path),
            "old_path": list(old_path or []),
            "region_open": list(region_open or []),
            "region_closed": list(region_closed or []),
            "region_path": list(region_path or []),
            "show_triangles": show_triangles,
        }

    def save_gif(self, snapshots, gif_name, max_frames=48):
        frames = [self.render_snapshot(s) for s in self.select_snapshots(snapshots, max_frames)]
        if frames:
            frames.extend([frames[-1]] * 4)
        gif_dir = os.path.join(os.path.dirname(__file__), "gif")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=420, loop=0, disposal=2)
        print(f"Saved {gif_path} with {len(frames)} frames")

    @staticmethod
    def select_snapshots(snapshots, max_frames):
        if len(snapshots) <= max_frames:
            return snapshots
        indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
        return [snapshots[i] for i in indices]

    def render_snapshot(self, snapshot):
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
        for cell_id, cell in self.cells.items():
            ax.add_patch(
                patches.Polygon(
                    cell["polygon"],
                    closed=True,
                    edgecolor="#111827" if cell["blocked"] else "#334155",
                    facecolor=self.cell_color(cell_id, snapshot),
                    linewidth=1.0,
                    alpha=0.98,
                    zorder=1,
                )
            )
        if snapshot["show_triangles"]:
            self.draw_navmesh_triangles(ax)
        for obstacle in self.obstacles:
            ax.add_patch(
                patches.Circle(
                    obstacle["center"],
                    obstacle["radius"],
                    edgecolor="#111827",
                    facecolor="#1f2937",
                    linewidth=1.1,
                    alpha=0.98,
                    zorder=3,
                )
            )
        self.draw_cluster_overlays(ax, snapshot)
        self.draw_edges(ax, snapshot)
        self.draw_path(ax, snapshot["old_path"], "#64748b", 2.2, 0.45, linestyle="--")
        self.draw_path(ax, snapshot["hint"], "#f97316", 2.0, 0.82)
        self.draw_path(ax, snapshot["path"], "#d62728", 3.0, 0.95)
        sx, sy = self.cells[self.start]["center"]
        gx, gy = self.cells[self.goal]["center"]
        ax.scatter(sx, sy, marker="s", s=76, color="#2b6cb0", zorder=6)
        ax.scatter(gx, gy, marker="s", s=76, color="#2f855a", zorder=6)
        ax.text(
            1.3,
            self.bounds()[3] - 1.0,
            f"{self.label}  step {snapshot['step']:3d}  open {len(snapshot['open']):3d}  closed {len(snapshot['closed']):3d}\n{snapshot['phase']}",
            fontsize=8.5,
            color="#1f2933",
            bbox={"facecolor": "white", "edgecolor": "#c7d0d9", "alpha": 0.9, "pad": 3},
            zorder=8,
        )
        xmin, xmax, ymin, ymax = self.bounds(pad=1.4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.label)
        fig.tight_layout(pad=0.35)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        buf.close()
        return frame

    def cell_color(self, cell_id, snapshot):
        cell = self.cells[cell_id]
        if cell["blocked"]:
            return "#1f2937"
        if cell_id in snapshot["path"]:
            return "#fecaca"
        if cell_id in snapshot["hint"]:
            return "#fed7aa"
        if cell_id in snapshot["closed"]:
            return "#cbd5e1"
        if cell_id in snapshot["open"]:
            return "#bbf7d0"
        if cell_id in self.changed_cells or cell["cost"] > 1.0:
            return "#fde68a"
        base = {
            "quad": "#dbeafe",
            "hex": "#dcfce7",
            "navmesh": "#fef3c7",
        }[cell["kind"]]
        if self.mode == "hier" and cell["cluster"] in snapshot["region_path"]:
            return "#e0e7ff"
        return base

    def draw_edges(self, ax, snapshot):
        visible = set(snapshot["closed"]) | set(snapshot["open"]) | set(snapshot["hint"]) | set(snapshot["path"])
        if self.mode in {"quad", "hex", "navmesh"} and not visible:
            visible = set(self.cells)
        for cell_id in visible:
            for neighbor, _ in self.edges.get(cell_id, []):
                if neighbor not in visible or cell_id > neighbor:
                    continue
                ax.plot(
                    [self.cells[cell_id]["center"][0], self.cells[neighbor]["center"][0]],
                    [self.cells[cell_id]["center"][1], self.cells[neighbor]["center"][1]],
                    color="#94a3b8",
                    linewidth=0.7,
                    alpha=0.32,
                    zorder=2,
                )

    def draw_navmesh_triangles(self, ax):
        for triangle in self.navmesh_triangles:
            closed = triangle + [triangle[0]]
            ax.plot(
                [point[0] for point in closed],
                [point[1] for point in closed],
                color="#f97316",
                linewidth=0.7,
                alpha=0.52,
                zorder=4,
            )

    def draw_cluster_overlays(self, ax, snapshot):
        if self.mode != "hier":
            return
        for region, cells in self.region_cells.items():
            xs, ys = [], []
            for cell_id in cells:
                for x, y in self.cells[cell_id]["polygon"]:
                    xs.append(x)
                    ys.append(y)
            color = "#2563eb" if region in snapshot["region_path"] else "#64748b"
            width = 2.0 if region in snapshot["region_path"] else 0.9
            ax.add_patch(
                patches.Rectangle(
                    (min(xs), min(ys)),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    fill=False,
                    edgecolor=color,
                    linewidth=width,
                    alpha=0.65,
                    zorder=4,
                )
            )
        self.draw_region_path(ax, snapshot["region_path"], "#2563eb", 2.4, 0.75)

    def draw_region_path(self, ax, regions, color, linewidth, alpha):
        if len(regions) < 2:
            return
        pts = [self.region_centers[region] for region in regions]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=linewidth, alpha=alpha, zorder=5)

    def draw_path(self, ax, path, color, linewidth, alpha, linestyle="-"):
        if len(path) < 2:
            return
        pts = [self.cells[c]["center"] for c in path if c in self.cells]
        if len(pts) < 2:
            return
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=6)

    def distance(self, a, b):
        ax, ay = self.cells[a]["center"]
        bx, by = self.cells[b]["center"]
        return math.hypot(ax - bx, ay - by)

    def average_center(self, cell_ids):
        xs = [self.cells[cell_id]["center"][0] for cell_id in cell_ids]
        ys = [self.cells[cell_id]["center"][1] for cell_id in cell_ids]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def inside_any_circle(self, point, margin=1.0):
        px, py = point
        for obstacle in self.obstacles:
            cx, cy = obstacle["center"]
            if math.hypot(px - cx, py - cy) < obstacle["radius"] * margin:
                return True
        return False

    @staticmethod
    def same_point(a, b):
        return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

    @staticmethod
    def point_key(point):
        return (round(point[0], 5), round(point[1], 5))

    @staticmethod
    def shared_vertex_keys(a_points, b_points):
        a_keys = {PrecomputedCellGraph.point_key(point) for point in a_points}
        b_keys = {PrecomputedCellGraph.point_key(point) for point in b_points}
        return a_keys & b_keys

    @staticmethod
    def distance_point_to_segment(point, a, b):
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        closest = (ax + t * dx, ay + t * dy)
        return math.hypot(px - closest[0], py - closest[1])

    @staticmethod
    def sort_polygon(points):
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        return sorted(points, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))

    @staticmethod
    def polygon_area(points):
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return 0.5 * area

    @staticmethod
    def is_convex_polygon(points):
        if len(points) < 3:
            return False
        signs = []
        for index in range(len(points)):
            a = points[index]
            b = points[(index + 1) % len(points)]
            c = points[(index + 2) % len(points)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) > 1e-7:
                signs.append(cross > 0)
        return bool(signs) and all(sign == signs[0] for sign in signs)

    @staticmethod
    def point_in_polygon(point, polygon):
        x, y = point
        inside = False
        for index, a in enumerate(polygon):
            b = polygon[(index + 1) % len(polygon)]
            if ((a[1] > y) != (b[1] > y)) and (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]):
                inside = not inside
        return inside

    def bounds(self, pad=0.0):
        xs, ys = [], []
        for cell in self.cells.values():
            for x, y in cell["polygon"]:
                xs.append(x)
                ys.append(y)
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    @staticmethod
    def box(center, w, h):
        cx, cy = center
        return [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]

    @staticmethod
    def hexagon(center, radius):
        return [
            (
                center[0] + radius * math.cos(i * math.pi / 3.0),
                center[1] + radius * math.sin(i * math.pi / 3.0),
            )
            for i in range(6)
        ]

    @staticmethod
    def centroid(polygon):
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )

    @staticmethod
    def polygon_edges(polygon):
        return [
            (polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(len(polygon))
        ]

    @staticmethod
    def edge_key(edge):
        a, b = edge
        points = (
            (round(a[0], 5), round(a[1], 5)),
            (round(b[0], 5), round(b[1], 5)),
        )
        return tuple(sorted(points))
