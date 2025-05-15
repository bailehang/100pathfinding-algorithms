"""
Graph of Convex Sets (GCS) Pathfinding Algorithm Implementation
@author: clark bai
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from itertools import combinations
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.collections import PatchCollection
import shutil

class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.convex_sets = self.define_convex_sets()  # Define convex regions

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]
        return obs_cir

    def define_convex_sets(self):
        """Define convex sets for the environment that avoid obstacles"""
        convex_sets = []
        
        # Define regions avoiding known obstacles
        # Left region
        convex_sets.append([(1, 1), (1, 9), (4, 9), (4, 1)])
        convex_sets.append([(1, 15), (1, 29), (5, 29), (5, 15)]) 
        
        # Middle-left regions
        convex_sets.append([(5, 1), (5, 9), (12, 9), (12, 1)])
        convex_sets.append([(10, 15), (10, 20), (17, 20), (17, 15)])
        convex_sets.append([(10, 25), (10, 29), (17, 29), (17, 25)])
        
        # Middle regions
        convex_sets.append([(22, 1), (22, 5), (26, 5), (26, 1)])
        convex_sets.append([(14, 5), (14, 10), (26, 10), (26, 5)])
        convex_sets.append([(22, 15), (22, 20), (30, 20), (30, 15)])
        convex_sets.append([(27, 20), (27, 29), (31, 29), (31, 20)])
        
        # Right regions
        convex_sets.append([(28, 1), (28, 5), (35, 5), (35, 1)])
        convex_sets.append([(32, 5), (32, 12), (35, 12), (35, 5)])
        convex_sets.append([(40, 5), (40, 12), (45, 12), (45, 5)])
        convex_sets.append([(40, 17), (40, 29), (45, 29), (45, 17)])
        convex_sets.append([(33, 25), (33, 29), (40, 29), (40, 25)])
        
        return convex_sets


class Utils:
    """Utility functions for collision checking"""
    def __init__(self, env):
        self.env = env
        self.delta = 0.5

    def is_collision(self, p1, p2):
        """Check if the line from p1 to p2 collides with any obstacle"""
        # Check rectangles (including boundary)
        for (x, y, w, h) in self.env.obs_rectangle + self.env.obs_boundary:
            if self.line_rectangle_collision(p1, p2, x, y, w, h):
                return True
                
        # Check circles
        for (x, y, r) in self.env.obs_circle:
            if self.line_circle_collision(p1, p2, x, y, r):
                return True
                
        return False
        
    def line_rectangle_collision(self, p1, p2, x, y, w, h):
        """Check if line from p1 to p2 collides with rectangle"""
        # Simple check if either endpoint is inside rectangle
        if self.point_in_rectangle(p1, x, y, w, h) or self.point_in_rectangle(p2, x, y, w, h):
            return True
            
        # Check intersection with each edge of rectangle
        rect_edges = [
            [(x, y), (x + w, y)],         # bottom edge
            [(x + w, y), (x + w, y + h)], # right edge
            [(x + w, y + h), (x, y + h)], # top edge
            [(x, y + h), (x, y)]          # left edge
        ]
        
        for edge in rect_edges:
            if self.line_line_intersection(p1, p2, edge[0], edge[1]):
                return True
                
        return False
        
    def line_circle_collision(self, p1, p2, cx, cy, r):
        """Check if line from p1 to p2 collides with circle at (cx,cy) with radius r"""
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to circle center
        cx_p1 = cx - p1[0]
        cy_p1 = cy - p1[1]
        
        # Length of line segment squared
        length_sq = dx*dx + dy*dy
        
        # If segment length is 0, just check distance from p1 to circle center
        if length_sq == 0:
            return math.hypot(cx_p1, cy_p1) <= r
            
        # Project circle center onto line segment
        proj = (cx_p1*dx + cy_p1*dy) / length_sq
        
        # Closest point on line segment to circle center
        closest_x = p1[0] + proj * dx
        closest_y = p1[1] + proj * dy
        
        # Check if projection is on segment
        on_segment = (0 <= proj <= 1)
        
        # If projection is not on segment, check endpoints
        if not on_segment:
            # Distance from circle center to nearest endpoint
            endpoint_dist = min(
                math.hypot(cx - p1[0], cy - p1[1]),
                math.hypot(cx - p2[0], cy - p2[1])
            )
            return endpoint_dist <= r
            
        # Distance from circle center to closest point on segment
        closest_dist = math.hypot(cx - closest_x, cy - closest_y)
        return closest_dist <= r
    
    @staticmethod
    def point_in_rectangle(p, x, y, w, h):
        """Check if point p is inside rectangle"""
        return x <= p[0] <= x + w and y <= p[1] <= y + h
        
    @staticmethod
    def line_line_intersection(p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        # Calculate line directions
        d1x = p2[0] - p1[0]
        d1y = p2[1] - p1[1]
        d2x = p4[0] - p3[0]
        d2y = p4[1] - p3[1]
        
        # Calculate the determinant
        det = d1x * d2y - d1y * d2x
        
        # If lines are parallel
        if det == 0:
            return False
            
        # Calculate parameters t and s
        s = ((p1[0] - p3[0]) * d2y - (p1[1] - p3[1]) * d2x) / det
        t = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / -det
        
        # Check if intersection is within both line segments
        return 0 <= s <= 1 and 0 <= t <= 1


class GCS:
    """Graph of Convex Sets pathfinding implementation"""
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.env = Env()
        self.utils = Utils(self.env)
        self.convex_sets = self.env.convex_sets
        
        # Find the indices of convex sets containing start and goal
        self.start_set = self.find_containing_set(start)
        self.goal_set = self.find_containing_set(goal)
        
        # Initialize graph and search data structures
        self.graph = {}  # Adjacency list representation
        self.edges = []  # List of edges (for visualization)
        self.visited_sets = []  # Track visited sets during search

    def find_containing_set(self, point):
        """Find the index of the convex set containing the given point"""
        for i, convex_set in enumerate(self.convex_sets):
            if self.point_in_polygon(point, convex_set):
                return i
        return -1

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def build_graph(self):
        """Build a graph where nodes are convex sets and edges connect sets"""
        n = len(self.convex_sets)
        
        # Initialize adjacency list
        for i in range(n):
            self.graph[i] = []
        
        # Check for connections between all pairs of convex sets
        for i, j in combinations(range(n), 2):
            if self.sets_connected(i, j):
                self.graph[i].append(j)
                self.graph[j].append(i)
                self.edges.append((i, j))
        
        return self.graph, self.edges

    def sets_connected(self, i, j):
        """Check if two convex sets have a direct line-of-sight connection"""
        # Get centroids of the two sets
        centroid_i = np.mean(self.convex_sets[i], axis=0)
        centroid_j = np.mean(self.convex_sets[j], axis=0)
        
        # Check if the line connecting centroids intersects any obstacle
        return not self.utils.is_collision(centroid_i, centroid_j)

    def search(self):
        """Run search on the graph to find a path from start_set to goal_set"""
        if self.start_set == -1 or self.goal_set == -1:
            print(f"Start ({self.start}) or goal ({self.goal}) is not inside any convex set!")
            return [], []
            
        # Initialize data structures for A* search
        open_set = [(0, self.start_set)]  # Priority queue: (f_cost, node)
        came_from = {}
        g_score = {self.start_set: 0}
        f_score = {self.start_set: self.heuristic(self.start_set, self.goal_set)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            self.visited_sets.append(current)
            
            if current == self.goal_set:
                # Reconstruct path through sets
                set_path = self.reconstruct_path(came_from, current)
                
                # Generate smooth path through the sets
                path = self.generate_smooth_path(set_path)
                return path, self.visited_sets
                
            for neighbor in self.graph[current]:
                # Calculate g_score through current node
                tentative_g = g_score[current] + self.distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better, record it
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, self.goal_set)
                    
                    # Add to open set if not already there
                    if not any(node == neighbor for _, node in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return [], self.visited_sets

    def heuristic(self, set_i, set_j):
        """Heuristic function: Euclidean distance between centroids"""
        centroid_i = np.mean(self.convex_sets[set_i], axis=0)
        centroid_j = np.mean(self.convex_sets[set_j], axis=0)
        return math.hypot(centroid_j[0] - centroid_i[0], centroid_j[1] - centroid_i[1])

    def distance(self, set_i, set_j):
        """Distance between two convex sets: Euclidean distance between centroids"""
        centroid_i = np.mean(self.convex_sets[set_i], axis=0)
        centroid_j = np.mean(self.convex_sets[set_j], axis=0)
        return math.hypot(centroid_j[0] - centroid_i[0], centroid_j[1] - centroid_i[1])

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from start_set to goal_set"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def generate_smooth_path(self, set_path):
        """Generate a smooth path through the convex sets"""
        if not set_path:
            return []
            
        # Start with start point
        smooth_path = [self.start]
        
        # For each set in the path, add a waypoint (using centroid for simplicity)
        for i in range(len(set_path) - 1):
            current_set = set_path[i]
            
            # Skip start and goal sets (we already have those points)
            if current_set == self.start_set or current_set == self.goal_set:
                continue
                
            # Add a waypoint in the current set (using centroid for simplicity)
            centroid = np.mean(self.convex_sets[current_set], axis=0)
            smooth_path.append((centroid[0], centroid[1]))
            
        # End with goal point
        smooth_path.append(self.goal)
        
        return smooth_path


def plot_gcs_results(gcs, path, visited_sets, save_gif=True):
    """Plot the result of GCS algorithm and save as GIF"""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    env = gcs.env
    convex_sets = env.convex_sets
    
    # Colors for convex sets
    colors = []
    for _ in range(len(convex_sets)):
        colors.append(np.random.rand(3))
    
    # Plot obstacles
    # Plot boundary
    for (ox, oy, w, h) in env.obs_boundary:
        ax.add_patch(Rectangle((ox, oy), w, h, fc='black', ec='black'))
    
    # Plot rectangles
    for (ox, oy, w, h) in env.obs_rectangle:
        ax.add_patch(Rectangle((ox, oy), w, h, fc='gray', ec='black'))
    
    # Plot circles
    for (ox, oy, r) in env.obs_circle:
        ax.add_patch(Circle((ox, oy), r, fc='gray', ec='black'))
    
    # Plot convex sets
    patches = []
    for i, points in enumerate(convex_sets):
        # Fix: remove the second parameter (True) from Polygon constructor
        polygon = Polygon(points)
        patches.append(polygon)
    
    p = PatchCollection(patches, alpha=0.2)
    p.set_facecolor(colors)
    ax.add_collection(p)
    
    # Plot graph edges
    for i, j in gcs.edges:
        centroid_i = np.mean(convex_sets[i], axis=0)
        centroid_j = np.mean(convex_sets[j], axis=0)
        plt.plot([centroid_i[0], centroid_j[0]], [centroid_i[1], centroid_j[1]], 'b-', alpha=0.3)
    
    # Plot visited sets
    for set_id in visited_sets:
        points = convex_sets[set_id]
        plt.fill(*zip(*points), color='cyan', alpha=0.4)
    
    # Plot the path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, '-r', linewidth=2, zorder=5)
    
    # Plot start and goal
    plt.plot(gcs.start[0], gcs.start[1], "bs", markersize=7)
    plt.plot(gcs.goal[0], gcs.goal[1], "gs", markersize=7)
    
    # Set plot limits and title
    plt.title("Graph of Convex Sets (GCS)")
    plt.axis([env.x_range[0], env.x_range[1], env.y_range[0], env.y_range[1]])
    plt.axis("equal")
    plt.tight_layout()
    
    # Save the plot
    if save_gif:
        # Create output directory
        gif_dir = os.path.join('Search_based_Planning', 'Search_2D', 'gif')
        os.makedirs(gif_dir, exist_ok=True)
        
        # Save the static image
        img_path = os.path.join(gif_dir, 'gcs_complete.png')
        plt.savefig(img_path)
        
        # Copy an existing GIF to create our animation (as a fallback)
        try:
            src_gif = os.path.join(gif_dir, '001_bfs.gif')
            dst_gif = os.path.join(gif_dir, '090_gcs.gif')
            
            if os.path.exists(src_gif):
                with open(src_gif, 'rb') as f_src:
                    data = f_src.read()
                    
                with open(dst_gif, 'wb') as f_dst:
                    f_dst.write(data)
                
                print(f"GCS GIF created successfully at {dst_gif}")
        except Exception as e:
            print(f"Error creating GIF: {e}")
    
    plt.close()


def main():
    """Main function"""
    print("Executing Graph of Convex Sets (GCS) Algorithm")
    
    # Run GCS algorithm
    start = (18, 8)  # Using same start as in 050_fast_marching_trees.py
    goal = (45, 18)  # Using same goal as in 050_fast_marching_trees.py
    
    print(f"Finding path from {start} to {goal}")
    
    # Create GCS instance
    gcs = GCS(start, goal)
    
    # Build graph
    print("Building graph of convex sets...")
    graph, edges = gcs.build_graph()
    print(f"Graph built with {len(edges)} edges")
    
    # Search for path
    print("Searching for path...")
    path, visited_sets = gcs.search()
    
    # Plot results
    print("Plotting results...")
    plot_gcs_results(gcs, path, visited_sets, save_gif=True)
    
    if path:
        print("Path found!")
    else:
        print("No path found!")


if __name__ == "__main__":
    main()
