"""
Breadth-first Searching_2D (BFS)
@author: clark bai
"""
import os
import sys
import math
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class BFS:
    """Breadth-First Search implementation without using A* structure"""
    def __init__(self, s_start, s_goal, _):
        # Initialize parameters
        self.s_start = s_start
        self.s_goal = s_goal
        
        # Initialize environment
        self.Env = env.Env()
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles
        
        # Initialize sets and dictionaries
        self.OPEN = deque()  # queue for BFS
        self.CLOSED = []  # visited nodes
        self.PARENT = dict()  # parent nodes
        self.g = dict()  # cost to come
    
    def searching(self):
        """BFS algorithm implementation"""
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        self.OPEN.append(self.s_start)
        
        while self.OPEN:
            s = self.OPEN.popleft()  # FIFO for BFS
            self.CLOSED.append(s)
            
            # Check if goal reached
            if s == self.s_goal:
                break
            
            # Explore neighbors
            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)
                
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    self.OPEN.append(s_n)  # Add to the end for BFS
        
        return self.extract_path(self.PARENT), self.CLOSED
    
    def get_neighbor(self, s):
        """Get valid neighboring nodes"""
        neighbors = []
        
        # Generate all possible neighbors
        for u in self.u_set:
            s_n = (s[0] + u[0], s[1] + u[1])
            
            # Check if valid move (not collision)
            if not self.is_collision(s, s_n):
                neighbors.append(s_n)
        
        return neighbors
    
    def cost(self, s_start, s_goal):
        """Calculate cost between two nodes"""
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
    
    def is_collision(self, s_start, s_end):
        """Check if there's a collision between two nodes"""
        if s_start in self.obs or s_end in self.obs:
            return True
        
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
            
            if s1 in self.obs or s2 in self.obs:
                return True
        
        return False
    
    def extract_path(self, PARENT):
        """Extract path from parent dictionary"""
        path = [self.s_goal]
        s = self.s_goal
        
        while True:
            s = PARENT[s]
            path.append(s)
            
            if s == self.s_start:
                break
        
        return list(path)


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    
    bfs = BFS(s_start, s_goal, None)  # Third parameter is ignored in BFS
    plot = plotting.Plotting(s_start, s_goal)
    
    path, visited = bfs.searching()
    plot.animation(path, visited, "Breadth-first Searching (BFS)")


if __name__ == '__main__':
    main()
