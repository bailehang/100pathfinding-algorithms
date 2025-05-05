"""
Debug script for Enhanced Parallel A* Algorithm
"""

import os
import sys
import traceback

print("Starting debug script...")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

try:
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
    print("Added parent directory to path")
    
    # Import env module first
    print("Importing env module...")
    from Search_based_Planning.Search_2D import env
    print("Successfully imported env module")
    
    # Create environment instance
    print("Creating environment...")
    env_instance = env.Env()
    print(f"Environment created with dimensions: {env_instance.x_range} x {env_instance.y_range}")
    
    # Import parallel A* module
    print("Importing parallel A* module...")
    astar_module = __import__('Search_based_Planning.Search_2D.009_Parallel_Astar', 
                             fromlist=['EnhancedParallelAStar'])
    EnhancedParallelAStar = astar_module.EnhancedParallelAStar
    print("Successfully imported EnhancedParallelAStar class")
    
    # Create instance with minimal parameters
    print("Creating EnhancedParallelAStar instance...")
    s_start = (5, 5)
    s_goals = [(45, 25)]
    parallel_astar = EnhancedParallelAStar(s_start, s_goals, "euclidean", 2)
    print("Successfully created EnhancedParallelAStar instance")
    
    # Print region information
    print("\nRegion information:")
    for region_id, region in parallel_astar.regions.items():
        print(f"Region {region_id}: {region['boundaries']}")
        print(f"  Neighbors: {region['neighbors']}")
    
    # Try to run the algorithm
    print("\nRunning search algorithm...")
    goal_paths, visited = parallel_astar.searching()
    print(f"Search completed. Found paths to {len(goal_paths)}/{len(s_goals)} goals")
    
    # Print path details
    for goal, (path, _) in goal_paths.items():
        print(f"Path to {goal}: length={len(path)}")
        if len(path) > 0:
            print(f"  First 3 steps: {path[:3]}")
            print(f"  Last 3 steps: {path[-3:]}")

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nStacktrace:")
    print(traceback.format_exc())
