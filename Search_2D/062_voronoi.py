"""
Voronoi Pathfinding Algorithm
@author: clark bai
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import distance_transform_edt


class Env:
    """Environment class for 2D grid world"""
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        x = self.x_range
        y = self.y_range
        obs = set()

        # Add boundary obstacles
        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))
        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        # Add additional obstacles
        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))
        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs


class Plotting:
    """Plotting class for visualization"""
    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, voronoi_map, name):
        """Animate the search process and final path"""
        self.plot_grid(name)
        self.plot_voronoi(voronoi_map)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

    def plot_voronoi(self, voronoi_map):
        """Plot the Voronoi diagram"""
        if voronoi_map is not None:
            # Plot Voronoi edges with cyan color and clearer markers
            voronoi_points_x = [x[0] for x in voronoi_map]
            voronoi_points_y = [x[1] for x in voronoi_map]
            plt.scatter(voronoi_points_x, voronoi_points_y, s=10, c='cyan', alpha=0.9)
            
            # Connect nearby Voronoi points to visualize the diagram better
            for i in range(len(voronoi_map)):
                for j in range(i+1, len(voronoi_map)):
                    p1 = voronoi_map[i]
                    p2 = voronoi_map[j]
                    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist < 3.0:  # Connect points that are close to each other
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', alpha=0.6, linewidth=0.8)
            
            plt.pause(0.1)

    def plot_visited(self, visited, cl='gray'):
        """Plot visited nodes during search"""
        if self.xI in visited:
            visited.remove(self.xI)
        if self.xG in visited:
            visited.remove(self.xG)

        count = 0
        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                plt.pause(0.1)
        plt.pause(0.5)

    def plot_path(self, path, cl='r', flag=False):
        """Plot the final path"""
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.5)


class VoronoiPlanner:
    """Voronoi Pathfinding Algorithm Implementation"""
    def __init__(self, s_start, s_goal, _):
        # Initialize parameters
        self.s_start = s_start
        self.s_goal = s_goal
        
        # Initialize environment
        self.Env = Env()
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles
        
        # Grid dimension parameters
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        
        # Voronoi parameters
        self.voronoi_threshold = 0.5  # Threshold to determine Voronoi edges
        self.voronoi_map = None       # Will store Voronoi points
        
        # Initialize sets and dictionaries
        self.PARENT = dict()  # parent nodes
        self.CLOSED = []      # visited nodes
        self.dist = dict()    # distance from start
    
    def searching(self):
        """Main method for Voronoi pathfinding algorithm"""
        # Generate Voronoi diagram
        self.voronoi_map = self.generate_voronoi_diagram()
        
        if not self.voronoi_map:
            # If no Voronoi points found, fall back to A* search
            path = self.a_star_search(self.s_start, self.s_goal)
            return path, self.CLOSED, self.voronoi_map
        
        # Connect start and goal to Voronoi diagram (find closest accessible points)
        v_start = self.find_accessible_voronoi_point(self.s_start)
        v_goal = self.find_accessible_voronoi_point(self.s_goal)
        
        if v_start is None or v_goal is None:
            # If can't connect to Voronoi, fall back to A* search
            path = self.a_star_search(self.s_start, self.s_goal)
            return path, self.CLOSED, self.voronoi_map
        
        # Find path from start to nearest Voronoi point
        path_to_voronoi = self.a_star_search(self.s_start, v_start)
        
        # Find path from Voronoi start to Voronoi goal
        voronoi_path = self.search_on_voronoi(v_start, v_goal)
        
        # Find path from Voronoi goal to actual goal
        path_from_voronoi = self.a_star_search(v_goal, self.s_goal)
        
        # Combine paths and remove duplicates
        if path_to_voronoi and voronoi_path and path_from_voronoi:
            complete_path = path_to_voronoi[:-1] + voronoi_path + path_from_voronoi[1:]
        else:
            # If any path segment failed, fall back to A* search
            complete_path = self.a_star_search(self.s_start, self.s_goal)
        
        return complete_path, self.CLOSED, self.voronoi_map
    
    def a_star_search(self, start, goal):
        """Standard A* search algorithm"""
        if start == goal:
            return [start]
            
        open_set = []
        closed_set = set()
        g_cost = {start: 0}
        f_cost = {start: self.calc_distance(start, goal)}
        parent = {start: start}
        open_set.append((f_cost[start], start))
        
        while open_set:
            # Get node with lowest f_cost
            open_set.sort()
            _, current = open_set.pop(0)
            self.CLOSED.append(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for motion in self.u_set:
                neighbor = (current[0] + motion[0], current[1] + motion[1])
                
                # Skip if in closed set or is an obstacle or would cause collision
                if neighbor in closed_set or neighbor in self.obs or self.is_collision(current, neighbor):
                    continue
                
                # Calculate costs
                tentative_g = g_cost[current] + self.calc_distance(current, neighbor)
                
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost[neighbor] = tentative_g + self.calc_distance(neighbor, goal)
                    parent[neighbor] = current
                    
                    # Add to open_set if not already there
                    if not any(neighbor == x[1] for x in open_set):
                        open_set.append((f_cost[neighbor], neighbor))
        
        # If no path is found
        return []
        
    def generate_voronoi_diagram(self):
        """
        Generate the Voronoi diagram using distance transform.
        The Voronoi diagram consists of points that are equidistant from 
        obstacles, creating a roadmap that maximizes the distance from obstacles.
        """
        # Create a binary image of obstacles (1s) and free space (0s)
        binary_map = np.zeros((self.x_range, self.y_range))
        
        for obs in self.obs:
            if 0 <= obs[0] < self.x_range and 0 <= obs[1] < self.y_range:
                binary_map[obs[0], obs[1]] = 1
        
        # Calculate distance transform (distance to nearest obstacle)
        distance_map = distance_transform_edt(1 - binary_map)
        
        # Identify Voronoi edges using a local maximum filter
        voronoi_points = []
        threshold_value = 1.5  # Minimum distance to be considered part of Voronoi
        
        # Create a gradient of the distance map to find ridges
        gx, gy = np.gradient(distance_map)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        for x in range(2, self.x_range - 2):
            for y in range(2, self.y_range - 2):
                # Skip obstacles and near-obstacle areas
                if (x, y) in self.obs or distance_map[x, y] < threshold_value:
                    continue
                
                # Check if this point is a local maximum or has low gradient (ridge)
                center_val = distance_map[x, y]
                
                # Use more directions to better identify ridges
                directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (1, 2), (2, 1)]
                for dx, dy in directions:
                    nx1, ny1 = x + dx, y + dy
                    nx2, ny2 = x - dx, y - dy
                    
                    # Check bounds
                    if (0 <= nx1 < self.x_range and 0 <= ny1 < self.y_range and
                        0 <= nx2 < self.x_range and 0 <= ny2 < self.y_range):
                        # This is a ridge if current point is higher than both points along this direction
                        if center_val > distance_map[nx1, ny1] and center_val > distance_map[nx2, ny2]:
                            voronoi_points.append((x, y))
                            break
        
        return voronoi_points
    
    def find_accessible_voronoi_point(self, point):
        """Find the nearest Voronoi point that can be reached without collision"""
        voronoi_points_with_dist = []
        
        # Calculate distances to all Voronoi points
        for v_point in self.voronoi_map:
            dist = self.calc_distance(point, v_point)
            # Check if there's a direct path without collision
            if not self.is_collision(point, v_point):
                voronoi_points_with_dist.append((dist, v_point))
        
        # If no direct path to any Voronoi point, try A* to find the closest reachable one
        if not voronoi_points_with_dist:
            # Sort Voronoi points by distance to target point
            sorted_v_points = sorted([(self.calc_distance(point, v), v) for v in self.voronoi_map])
            
            # Try to find A* path to the closest few points
            for _, v_point in sorted_v_points[:10]:  # Try top 10 closest points
                path = self.a_star_search(point, v_point)
                if path:
                    return v_point
            return None
        
        # Return the closest accessible point
        voronoi_points_with_dist.sort()
        return voronoi_points_with_dist[0][1]
    
    def is_collision(self, s_start, s_end):
        """Check if there's a collision between two nodes"""
        if s_start in self.obs or s_end in self.obs:
            return True
        
        # Check for diagonal movement collision
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            
            if s1 in self.obs or s2 in self.obs:
                return True
                
        # Line collision check for non-adjacent cells
        if abs(s_end[0] - s_start[0]) + abs(s_end[1] - s_start[1]) > 2:
            # Bresenham's line algorithm for collision detection
            x0, y0 = s_start
            x1, y1 = s_end
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            
            while x0 != x1 or y0 != y1:
                if (x0, y0) in self.obs:
                    return True
                    
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
        
        return False

    
    def search_on_voronoi(self, v_start, v_goal):
        """Find a path along the Voronoi diagram from v_start to v_goal"""
        # A* algorithm on the Voronoi diagram
        open_set = []
        visited = set()
        g_cost = {v_start: 0}
        f_cost = {v_start: self.calc_distance(v_start, v_goal)}
        parent = {v_start: v_start}
        open_set.append((f_cost[v_start], v_start))
        
        while open_set:
            open_set.sort()
            _, current = open_set.pop(0)
            
            if current == v_goal:
                path = []
                while current != v_start:
                    path.append(current)
                    current = parent[current]
                path.append(v_start)
                path.reverse()
                return path
            
            if current in visited:
                continue
                
            visited.add(current)
            self.CLOSED.append(current)
            
            # Get neighbors on Voronoi diagram
            neighbors = self.get_voronoi_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Check for collision between current and neighbor
                if self.is_collision(current, neighbor):
                    continue
                
                tentative_g = g_cost[current] + self.calc_distance(current, neighbor)
                
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost[neighbor] = tentative_g + self.calc_distance(neighbor, v_goal)
                    parent[neighbor] = current
                    open_set.append((f_cost[neighbor], neighbor))
        
        # If no path is found, return empty list
        return []
    
    def get_voronoi_neighbors(self, point):
        """Get neighboring points on the Voronoi diagram"""
        neighbors = []
        MAX_NEIGHBORS = 8  # Maximum number of neighbors
        
        # Find all Voronoi points within distance
        candidate_neighbors = []
        for v_point in self.voronoi_map:
            if v_point == point:
                continue
                
            dist = self.calc_distance(point, v_point)
            # Use adaptive radius based on density of Voronoi points
            if dist <= 5.0:  # Increased radius to ensure connectivity
                candidate_neighbors.append((dist, v_point))
        
        # Sort by distance and take the closest MAX_NEIGHBORS
        candidate_neighbors.sort()
        for _, v_point in candidate_neighbors[:MAX_NEIGHBORS]:
            # Only add if there's no collision
            if not self.is_collision(point, v_point):
                neighbors.append(v_point)
        
        return neighbors
    
    def calc_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    """Main function to run the Voronoi pathfinding algorithm"""
    s_start = (5, 5)
    s_goal = (45, 25)
    
    voronoi_planner = VoronoiPlanner(s_start, s_goal, None)  # Third parameter is ignored
    plot = Plotting(s_start, s_goal)
    
    path, visited, voronoi_map = voronoi_planner.searching()
    plot.animation(path, visited, voronoi_map, "Voronoi Pathfinding Algorithm")


if __name__ == '__main__':
    main()
