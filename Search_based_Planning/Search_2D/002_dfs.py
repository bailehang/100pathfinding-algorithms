"""
Depth-first Searching_2D (DFS)
@author: clark bai
"""
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class DFS:
    """Depth-First Search implementation with shortest path tracking
    
    This is a DFS-based algorithm with path cost tracking to ensure
    the shortest path is found. While it uses DFS expansion order (LIFO stack),
    it also tracks costs like Dijkstra's algorithm to guarantee optimal paths.
    """
    def __init__(self, s_start, s_goal, _):
        # Initialize parameters
        self.s_start = s_start
        self.s_goal = s_goal
        
        # Initialize environment
        self.Env = env.Env()
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles
        
        # Initialize sets and dictionaries
        self.OPEN = []            # stack for DFS (LIFO)
        self.OPEN_set = set()     # set for O(1) lookups in OPEN
        self.CLOSED = set()       # visited nodes as a set for O(1) lookup
        self.PARENT = dict()      # parent nodes for path reconstruction
        self.g = dict()           # cost to come (needed for shortest path)
    
    def searching(self):
        """DFS algorithm modified to maintain shortest path information"""
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # Cost from start to start is 0
        self.OPEN.append(self.s_start)
        visited_list = []  # For animation purposes
        
        while self.OPEN:
            s = self.OPEN.pop()  # LIFO for DFS
            
            # Add to visualization
            if s not in self.CLOSED:
                visited_list.append(s)
            
            # Add to closed set
            self.CLOSED.add(s)
            
            # Check if goal reached
            if s == self.s_goal:
                break
            
            # Explore neighbors
            for s_n in self.get_neighbor(s):
                # Calculate new cost to this neighbor
                new_cost = self.g[s] + self.calculate_distance(s, s_n)
                
                # Check if new cost is better or node is unvisited
                if s_n not in self.g or new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost  # Update cost
                    self.PARENT[s_n] = s    # Update parent
                    
                    # If we find a better path to a node in CLOSED,
                    # remove it from CLOSED to reconsider it
                    if s_n in self.CLOSED:
                        self.CLOSED.remove(s_n)
                        
                    # Add to OPEN if not already there (using O(1) set lookup)
                    if s_n not in self.OPEN_set:
                        self.OPEN.append(s_n)
                        self.OPEN_set.add(s_n)
        
        return self.extract_path(self.PARENT), visited_list
    
    def get_neighbor(self, s):
        """Get valid neighboring nodes - optimized for speed without heuristics"""
        neighbors = []
        x, y = s
        
        # Check each of the 8 possible moves
        # Process all neighbors in the standard order defined by u_set
        for dx, dy in self.u_set:
            s_n = (x + dx, y + dy)
            
            # Quick collision check for obstacles
            if s_n in self.obs:
                continue
                
            # More detailed collision check for diagonal moves
            if dx != 0 and dy != 0:
                # Check for diagonal collision
                if (x + dx, y) in self.obs and (x, y + dy) in self.obs:
                    continue
                    
            neighbors.append(s_n)
        
        return neighbors
    
    def calculate_distance(self, s_start, s_goal):
        """Calculate Euclidean distance between two points - for path cost calculation"""
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
    
    def is_collision(self, s_start, s_end):
        """Check if there's a collision between two nodes - optimized"""
        # Quick check for endpoint obstacles
        if s_start in self.obs or s_end in self.obs:
            return True
        
        # Only check diagonal movements (more expensive)
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            # Calculate diagonal passing cells
            dx = s_end[0] - s_start[0]
            dy = s_end[1] - s_start[1]
            
            if dx * dy > 0:  # Same sign
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:  # Different sign
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            
            # Check if either diagonal cell is an obstacle
            if s1 in self.obs or s2 in self.obs:
                return True
        
        return False
    
    def extract_path(self, PARENT):
        """Extract path from parent dictionary - optimized version"""
        # Check if path exists
        if self.s_goal not in PARENT:
            return []
            
        # Reconstruct path efficiently
        path = []
        s = self.s_goal
        
        while s != self.s_start:
            path.append(s)
            s = PARENT[s]
        
        path.append(self.s_start)
        return path


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    
    dfs = DFS(s_start, s_goal, None)  # Third parameter is ignored in DFS
    plot = plotting.Plotting(s_start, s_goal)
    
    path, visited = dfs.searching()
    plot.animation(path, visited, "Depth-first Searching (DFS)")


if __name__ == '__main__':
    main()
