"""
Graph of Convex Sets (GCS) Pathfinding Algorithm
@author: clark bai
"""

import os
import sys
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import heapq
import io
from PIL import Image


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

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


class Utils:
    def __init__(self):
        self.env = Env()
        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_boundary = self.env.obs_boundary
        self.obs_rectangle = self.env.obs_rectangle

    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.env.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.env.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False


class ConvexRegion:
    """Represents a single convex polygon region"""
    def __init__(self, vertices, region_id):
        self.vertices = vertices  # List of polygon vertices [(x1,y1), (x2,y2), ...]
        self.region_id = region_id
        self.center = self.compute_center()
        self.neighbors = []  # List of adjacent region IDs
        self.g_cost = float('inf')  # g-value in A* search
        self.f_cost = float('inf')  # f-value in A* search
        self.parent = None  # Parent region in A* search

    def compute_center(self):
        """Compute the centroid of the convex polygon"""
        if not self.vertices:
            return (0, 0)
        x_sum = sum(v[0] for v in self.vertices)
        y_sum = sum(v[1] for v in self.vertices)
        n = len(self.vertices)
        return (x_sum / n, y_sum / n)

    def contains_point(self, point):
        """Check if point is inside convex polygon"""
        x, y = point
        n = len(self.vertices)
        inside = False

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_area(self):
        """Calculate convex polygon area"""
        if len(self.vertices) < 3:
            return 0
        
        area = 0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2


class GCS:
    """Main Graph of Convex Sets algorithm class"""
    def __init__(self, x_start, x_goal):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        
        self.env = Env()
        self.utils = Utils()
        
        self.regions = []  # List of convex regions
        self.region_graph = {}  # Region adjacency graph {region_id: [neighbor_ids]}
        
        self.start_region_id = None
        self.goal_region_id = None
        
        # Search related
        self.open_list = []
        self.closed_list = []
        self.path = []

    def decompose_space(self):
        """Decompose free space into convex regions"""
        print("Starting space decomposition...")
        
        # Use simplified grid-based decomposition method
        self.regions = self.grid_based_decomposition()
        
        # Remove regions that intersect with obstacles
        self.regions = self.filter_valid_regions(self.regions)
        
        print(f"Generated {len(self.regions)} convex regions")
        return self.regions

    def grid_based_decomposition(self):
        """Grid-based simplified convex decomposition"""
        regions = []
        region_id = 0
        
        # Create regular grid with higher resolution for better path finding
        grid_size = 5  # Increased grid size for more regions
        x_step = (self.env.x_range[1] - self.env.x_range[0]) / grid_size
        y_step = (self.env.y_range[1] - self.env.y_range[0]) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = self.env.x_range[0] + i * x_step
                x2 = self.env.x_range[0] + (i + 1) * x_step
                y1 = self.env.y_range[0] + j * y_step
                y2 = self.env.y_range[0] + (j + 1) * y_step
                
                # Create rectangular region vertices
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                region = ConvexRegion(vertices, region_id)
                
                # Check if region is valid (doesn't intersect obstacles)
                if self.is_region_valid(region):
                    regions.append(region)
                    region_id += 1
        
        return regions

    def is_region_valid(self, region):
        """Check if region is valid (doesn't overlap with obstacles)"""
        # Check if region center is in free space
        center_node = Node(region.center)
        if self.utils.is_inside_obs(center_node):
            return False
        
        # Simple check: if center is free, consider region valid
        # This is a simplified approach for better connectivity
        return True

    def filter_valid_regions(self, regions):
        """Filter out valid regions"""
        valid_regions = []
        for region in regions:
            if region.get_area() > 1.0:  # Filter out regions that are too small
                valid_regions.append(region)
        return valid_regions

    def build_region_graph(self):
        """Build region adjacency graph"""
        print("Building region adjacency graph...")
        
        self.region_graph = {region.region_id: [] for region in self.regions}
        
        # Check adjacency relationship for each pair of regions
        for i, region1 in enumerate(self.regions):
            for j, region2 in enumerate(self.regions):
                if i < j and self.are_regions_adjacent(region1, region2):
                    # Add bidirectional connection
                    self.region_graph[region1.region_id].append(region2.region_id)
                    self.region_graph[region2.region_id].append(region1.region_id)
                    
                    region1.neighbors.append(region2.region_id)
                    region2.neighbors.append(region1.region_id)
        
        total_connections = sum(len(neighbors) for neighbors in self.region_graph.values()) // 2
        print(f"Established {total_connections} region connections")

    def are_regions_adjacent(self, region1, region2):
        """Check if two regions are adjacent"""
        # Simplified adjacency detection: check distance between region centers
        center1 = region1.center
        center2 = region2.center
        
        distance = math.hypot(center2[0] - center1[0], center2[1] - center1[1])
        
        # If distance is less than threshold, consider adjacent
        threshold = 12.0  # Adjusted threshold for better connectivity
        return distance < threshold

    def find_region_containing_point(self, point):
        """Find region containing specified point"""
        for region in self.regions:
            if region.contains_point(point):
                return region.region_id
        return None

    def search_path(self):
        """Search path on region graph"""
        print("Starting path search...")
        
        # Find regions containing start and goal points
        self.start_region_id = self.find_region_containing_point((self.x_start.x, self.x_start.y))
        self.goal_region_id = self.find_region_containing_point((self.x_goal.x, self.x_goal.y))
        
        if self.start_region_id is None or self.goal_region_id is None:
            print("Start or goal point not in any valid region")
            return []
        
        print(f"Start in region {self.start_region_id}, goal in region {self.goal_region_id}")
        
        # Initialize search
        start_region = self.get_region_by_id(self.start_region_id)
        start_region.g_cost = 0
        start_region.f_cost = self.heuristic(start_region)
        
        heapq.heappush(self.open_list, (start_region.f_cost, self.start_region_id))
        
        while self.open_list:
            current_f, current_id = heapq.heappop(self.open_list)
            current_region = self.get_region_by_id(current_id)
            
            if current_id in self.closed_list:
                continue
                
            self.closed_list.append(current_id)
            
            if current_id == self.goal_region_id:
                print("Path found!")
                return self.reconstruct_path()
            
            # Check neighbor regions
            for neighbor_id in self.region_graph[current_id]:
                if neighbor_id in self.closed_list:
                    continue
                
                neighbor_region = self.get_region_by_id(neighbor_id)
                tentative_g = current_region.g_cost + self.region_distance(current_region, neighbor_region)
                
                if tentative_g < neighbor_region.g_cost:
                    neighbor_region.parent = current_id
                    neighbor_region.g_cost = tentative_g
                    neighbor_region.f_cost = tentative_g + self.heuristic(neighbor_region)
                    
                    heapq.heappush(self.open_list, (neighbor_region.f_cost, neighbor_id))
        
        print("No path found")
        return []

    def get_region_by_id(self, region_id):
        """Get region object by ID"""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None

    def heuristic(self, region):
        """Heuristic function: Euclidean distance to goal region"""
        goal_region = self.get_region_by_id(self.goal_region_id)
        if goal_region is None:
            return 0
        
        return math.hypot(
            goal_region.center[0] - region.center[0],
            goal_region.center[1] - region.center[1]
        )

    def region_distance(self, region1, region2):
        """Calculate distance between two regions"""
        return math.hypot(
            region2.center[0] - region1.center[0],
            region2.center[1] - region1.center[1]
        )

    def reconstruct_path(self):
        """Reconstruct path"""
        path = []
        current_id = self.goal_region_id
        
        while current_id is not None:
            path.append(current_id)
            current_region = self.get_region_by_id(current_id)
            current_id = current_region.parent if current_region else None
        
        path.reverse()
        self.path = path
        return path

    def optimize_path(self):
        """Optimize path within each region"""
        if not self.path:
            return []
        
        optimized_path = []
        
        # Add start point
        optimized_path.append((self.x_start.x, self.x_start.y))
        
        # Add connection points between regions
        for i in range(len(self.path) - 1):
            current_region = self.get_region_by_id(self.path[i])
            next_region = self.get_region_by_id(self.path[i + 1])
            
            # Use region center as connection point
            optimized_path.append(current_region.center)
        
        # Add goal point
        optimized_path.append((self.x_goal.x, self.x_goal.y))
        
        return optimized_path


class PlottingGCS:
    """GCS algorithm visualization class"""
    def __init__(self, x_start, x_goal, gcs):
        self.x_start = x_start
        self.x_goal = x_goal
        self.gcs = gcs
        self.env = Env()
        
        self.frames = []
        self.fig_size = (12, 8)
        
        # Color configuration
        self.colors = {
            'obstacle': 'gray',
            'boundary': 'black',
            'start': 'blue',
            'goal': 'red',
            'regions': ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
                       'lightcyan', 'wheat', 'lavender', 'mistyrose', 'honeydew'],
            'current': 'orange',
            'open': 'yellow',
            'closed': 'lightgray',
            'path': 'red',
            'connections': 'darkblue'
        }
        
        self.current_step = ""

    def animation_gcs(self, save_gif=True):
        """Complete GCS algorithm animation"""
        print("Starting GCS algorithm animation...")
        
        # Phase 1: Environment setup
        self.animate_environment_setup()
        
        # Phase 2: Convex decomposition
        self.animate_space_decomposition()
        
        # Phase 3: Graph construction
        self.animate_graph_construction()
        
        # Phase 4: Path search
        self.animate_path_search()
        
        # Phase 5: Path optimization
        self.animate_path_optimization()
        
        # Phase 6: Final result
        self.animate_final_result()
        
        if save_gif:
            self.save_animation_as_gif("087_Graph_of_Convex_Sets")
        
        print("Animation completed!")

    def animate_environment_setup(self):
        """Animation: Environment setup"""
        self.current_step = "Environment Setup"
        
        for frame in range(5):
            plt.figure(figsize=self.fig_size, dpi=100, clear=True)
            self.plot_basic_environment()
            
            if frame >= 1:
                plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=10, label="Start")
            if frame >= 2:
                plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=10, label="Goal")
            
            plt.title(f"Graph of Convex Sets - {self.current_step}")
            plt.legend()
            plt.axis("equal")
            plt.xlim(-1, 51)
            plt.ylim(-1, 31)
            
            self.capture_frame()

    def animate_space_decomposition(self):
        """Animation: Space decomposition process"""
        self.current_step = "Space Decomposition"
        
        # Execute decomposition
        regions = self.gcs.decompose_space()
        
        # Show regions one by one
        for i, region in enumerate(regions):
            plt.figure(figsize=self.fig_size, dpi=100, clear=True)
            self.plot_basic_environment()
            
            # Show generated regions
            for j, r in enumerate(regions[:i+1]):
                color = self.colors['regions'][j % len(self.colors['regions'])]
                self.plot_region(r, color, alpha=0.6)
                
                # Show region ID
                plt.text(r.center[0], r.center[1], str(r.region_id), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=10)
            plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=10)
            
            plt.title(f"Graph of Convex Sets - {self.current_step} ({i+1}/{len(regions)})")
            plt.axis("equal")
            plt.xlim(-1, 51)
            plt.ylim(-1, 31)
            
            self.capture_frame()

    def animate_graph_construction(self):
        """Animation: Graph construction process"""
        self.current_step = "Graph Construction"
        
        # Execute graph construction
        self.gcs.build_region_graph()
        
        # Show graph construction process
        for step in range(3):
            plt.figure(figsize=self.fig_size, dpi=100, clear=True)
            self.plot_basic_environment()
            
            # Show all regions
            for region in self.gcs.regions:
                color = self.colors['regions'][region.region_id % len(self.colors['regions'])]
                self.plot_region(region, color, alpha=0.4)
                plt.text(region.center[0], region.center[1], str(region.region_id), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Show connections
            if step >= 1:
                for region_id, neighbors in self.gcs.region_graph.items():
                    region = self.gcs.get_region_by_id(region_id)
                    for neighbor_id in neighbors:
                        neighbor = self.gcs.get_region_by_id(neighbor_id)
                        if region and neighbor:
                            plt.plot([region.center[0], neighbor.center[0]], 
                                   [region.center[1], neighbor.center[1]], 
                                   self.colors['connections'], alpha=0.6, linewidth=1)
            
            plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=10)
            plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=10)
            
            step_names = ["Adjacency Detection", "Build Connections", "Complete Graph"]
            plt.title(f"Graph of Convex Sets - {self.current_step}: {step_names[step]}")
            plt.axis("equal")
            plt.xlim(-1, 51)
            plt.ylim(-1, 31)
            
            self.capture_frame()

    def animate_path_search(self):
        """Animation: Path search process"""
        self.current_step = "Path Search"
        
        # Execute search
        path = self.gcs.search_path()
        
        # Show search process
        if self.gcs.closed_list:
            for step in range(len(self.gcs.closed_list)):
                plt.figure(figsize=self.fig_size, dpi=100, clear=True)
                self.plot_basic_environment()
                
                # Show all regions
                for region in self.gcs.regions:
                    color = self.colors['regions'][region.region_id % len(self.colors['regions'])]
                    alpha = 0.3
                    
                    # Highlight current state
                    if region.region_id in self.gcs.closed_list[:step+1]:
                        color = self.colors['closed']
                        alpha = 0.8
                    elif step < len(self.gcs.closed_list) and region.region_id == self.gcs.closed_list[step]:
                        color = self.colors['current']
                        alpha = 0.9
                    
                    self.plot_region(region, color, alpha=alpha)
                    plt.text(region.center[0], region.center[1], str(region.region_id), 
                            ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Show graph connections
                for region_id, neighbors in self.gcs.region_graph.items():
                    region = self.gcs.get_region_by_id(region_id)
                    for neighbor_id in neighbors:
                        neighbor = self.gcs.get_region_by_id(neighbor_id)
                        if region and neighbor:
                            plt.plot([region.center[0], neighbor.center[0]], 
                                   [region.center[1], neighbor.center[1]], 
                                   self.colors['connections'], alpha=0.3, linewidth=1)
                
                plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=10)
                plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=10)
                
                plt.title(f"Graph of Convex Sets - {self.current_step}: Step {step+1}")
                plt.axis("equal")
                plt.xlim(-1, 51)
                plt.ylim(-1, 31)
                
                self.capture_frame()

    def animate_path_optimization(self):
        """Animation: Path optimization process"""
        self.current_step = "Path Optimization"
        
        optimized_path = self.gcs.optimize_path()
        
        # Show path optimization process
        for step in range(3):
            plt.figure(figsize=self.fig_size, dpi=100, clear=True)
            self.plot_basic_environment()
            
            # Show all regions
            for region in self.gcs.regions:
                color = self.colors['regions'][region.region_id % len(self.colors['regions'])]
                alpha = 0.3
                
                # Highlight path regions
                if region.region_id in self.gcs.path:
                    alpha = 0.7
                
                self.plot_region(region, color, alpha=alpha)
                plt.text(region.center[0], region.center[1], str(region.region_id), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Show graph connections
            for region_id, neighbors in self.gcs.region_graph.items():
                region = self.gcs.get_region_by_id(region_id)
                for neighbor_id in neighbors:
                    neighbor = self.gcs.get_region_by_id(neighbor_id)
                    if region and neighbor:
                        plt.plot([region.center[0], neighbor.center[0]], 
                               [region.center[1], neighbor.center[1]], 
                               self.colors['connections'], alpha=0.2, linewidth=1)
            
            # Show path
            if step >= 1 and optimized_path:
                path_x = [p[0] for p in optimized_path]
                path_y = [p[1] for p in optimized_path]
                plt.plot(path_x, path_y, self.colors['path'], linewidth=3, marker='o', markersize=4)
            
            plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=10)
            plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=10)
            
            step_names = ["Initial Path", "Path Optimization", "Final Path"]
            plt.title(f"Graph of Convex Sets - {self.current_step}: {step_names[step]}")
            plt.axis("equal")
            plt.xlim(-1, 51)
            plt.ylim(-1, 31)
            
            self.capture_frame()

    def animate_final_result(self):
        """Animation: Final result display"""
        self.current_step = "Final Result"
        
        optimized_path = self.gcs.optimize_path()
        
        # Final result display
        for frame in range(5):
            plt.figure(figsize=self.fig_size, dpi=100, clear=True)
            self.plot_basic_environment()
            
            # Show all regions
            for region in self.gcs.regions:
                color = self.colors['regions'][region.region_id % len(self.colors['regions'])]
                alpha = 0.4
                
                # Highlight path regions
                if region.region_id in self.gcs.path:
                    alpha = 0.8
                
                self.plot_region(region, color, alpha=alpha)
                plt.text(region.center[0], region.center[1], str(region.region_id), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Show final path
            if optimized_path:
                path_x = [p[0] for p in optimized_path]
                path_y = [p[1] for p in optimized_path]
                plt.plot(path_x, path_y, self.colors['path'], linewidth=4, marker='o', markersize=6, label='Optimal Path')
            
            plt.plot(self.x_start[0], self.x_start[1], "bs", markersize=12, label="Start")
            plt.plot(self.x_goal[0], self.x_goal[1], "rs", markersize=12, label="Goal")
            
            # Add statistics
            stats_text = f"Regions: {len(self.gcs.regions)}\n"
            stats_text += f"Path Length: {len(self.gcs.path)} regions\n"
            if optimized_path:
                total_distance = sum(math.hypot(optimized_path[i+1][0] - optimized_path[i][0],
                                              optimized_path[i+1][1] - optimized_path[i][1])
                                   for i in range(len(optimized_path)-1))
                stats_text += f"Path Distance: {total_distance:.2f}"
            
            plt.text(2, 28, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            plt.title(f"Graph of Convex Sets - {self.current_step}")
            plt.legend()
            plt.axis("equal")
            plt.xlim(-1, 51)
            plt.ylim(-1, 31)
            
            self.capture_frame()

    def plot_basic_environment(self):
        """Draw basic environment (boundaries and obstacles)"""
        # Draw boundaries
        for (ox, oy, w, h) in self.env.obs_boundary:
            rect = patches.Rectangle((ox, oy), w, h, 
                                   edgecolor=self.colors['boundary'], 
                                   facecolor=self.colors['boundary'], fill=True)
            plt.gca().add_patch(rect)
        
        # Draw rectangular obstacles
        for (ox, oy, w, h) in self.env.obs_rectangle:
            rect = patches.Rectangle((ox, oy), w, h, 
                                   edgecolor='black', 
                                   facecolor=self.colors['obstacle'], fill=True)
            plt.gca().add_patch(rect)
        
        # Draw circular obstacles
        for (ox, oy, r) in self.env.obs_circle:
            circle = patches.Circle((ox, oy), r, 
                                  edgecolor='black', 
                                  facecolor=self.colors['obstacle'], fill=True)
            plt.gca().add_patch(circle)

    def plot_region(self, region, color, alpha=0.5):
        """Draw convex region"""
        if len(region.vertices) >= 3:
            polygon = Polygon(region.vertices, closed=True, 
                            facecolor=color, edgecolor='black', 
                            alpha=alpha, linewidth=1)
            plt.gca().add_patch(polygon)

    def capture_frame(self):
        """Capture current frame"""
        buf = io.BytesIO()
        
        # Get current figure
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Save figure to buffer
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Open image using PIL and convert to RGB
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        
        # Convert to numpy array
        image = np.array(img_rgb)
        
        # Add to frames list
        self.frames.append(image)
        
        # Close buffer
        buf.close()

    def save_animation_as_gif(self, name, fps=10):
        """Save animation as GIF file"""
        # Create gif directory
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Captured frames: {len(self.frames)}")
        
        # Ensure all frames have the same dimensions
        if self.frames:
            first_frame_shape = self.frames[0].shape
            for i, frame in enumerate(self.frames):
                if frame.shape != first_frame_shape:
                    print(f"Warning: Frame {i} has inconsistent shape: {frame.shape} vs {first_frame_shape}")
                    # Resize inconsistent frames
                    resized_frame = np.array(Image.fromarray(frame).resize(
                        (first_frame_shape[1], first_frame_shape[0]), 
                        Image.LANCZOS))
                    self.frames[i] = resized_frame
        
        # Check if frames list is not empty
        if self.frames:
            try:
                # Convert NumPy arrays to PIL Images
                print("Converting frames to PIL Images...")
                frames_p = []
                for i, frame in enumerate(self.frames):
                    try:
                        img = Image.fromarray(frame)
                        img_p = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                        frames_p.append(img_p)
                        if i % 10 == 0:
                            print(f"Converted frame {i+1}/{len(self.frames)}")
                    except Exception as e:
                        print(f"Error converting frame {i}: {e}")
                
                print(f"Successfully converted {len(frames_p)} frames")

                # Save GIF
                print("Saving GIF file...")
                frames_p[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames_p[1:],
                    save_all=True,
                    duration=int(1000 / fps),
                    loop=0,
                    disposal=2  # Replace previous frame
                )
                print(f"GIF animation saved to {gif_path}")
                
                # Verify file was created
                if os.path.exists(gif_path):
                    print(f"File size: {os.path.getsize(gif_path) / 1024:.2f} KB")
                else:
                    print("Warning: File does not exist after saving!")
            except Exception as e:
                print(f"Error during GIF creation: {e}")
        else:
            print("No frames to save!")

        # Close figure
        plt.close()


def main():
    """Main function"""
    print("Graph of Convex Sets Pathfinding Algorithm Demo")
    print("=" * 50)
    
    # Set start and goal points
    x_start = (5, 5)
    x_goal = (45, 25)
    
    print(f"Start: {x_start}")
    print(f"Goal: {x_goal}")
    
    # Create GCS algorithm instance
    gcs = GCS(x_start, x_goal)
    
    # Create visualization instance
    plot = PlottingGCS(x_start, x_goal, gcs)
    
    # Run complete algorithm animation
    print("\nStarting algorithm execution and animation generation...")
    plot.animation_gcs(save_gif=True)
    
    print("\nAlgorithm execution completed!")
    print(f"Generated convex regions: {len(gcs.regions)}")
    if gcs.path:
        print(f"Found path through {len(gcs.path)} regions")
        print(f"Path region sequence: {gcs.path}")
    else:
        print("No path found")


if __name__ == '__main__':
    main()
