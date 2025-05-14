"""
Depth-first Searching_2D (DFS)
@author: clark bai
"""
import math
import os
import sys
import time
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
        self.frames = []

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name, save_gif=False):
        """Animate the search process and final path"""
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()
        if save_gif:
            self.save_animation_as_gif(name)

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

        # Capture the initial grid frame
        self.capture_frame()

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
                plt.pause(0.001)
                self.capture_frame()

        plt.pause(0.001)
        self.capture_frame()

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

        plt.pause(0.1)
        self.capture_frame()

    def capture_frame(self):
        """Capture the current figure as a frame with correct color handling"""
        fig = plt.gcf()
        fig.canvas.draw()

        # Get the RGBA buffer from the canvas
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()

        # Convert to numpy array and reshape
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        r = data[1::4]  # Red channel
        g = data[2::4]  # Green channel
        b = data[3::4]  # Blue channel
        image = np.stack([r, g, b], axis=-1).reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Add to frames
        self.frames.append(image)

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with proper color handling"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames to save: {len(self.frames)}")

        # Check if frames list is not empty before saving
        if self.frames:
            # Convert NumPy arrays to PIL Images, then to GIF-compatible mode (P with palette)
            frames_p = [Image.fromarray(frame).convert('P', palette=Image.ADAPTIVE, colors=256) for frame in self.frames]

            # Save with proper disposal method to avoid artifacts
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
        else:
            print("No frames to save!")

        # Close the figure
        plt.close()


class DFS:
    """Depth-First Search implementation with shortest path tracking"""
    def __init__(self, s_start, s_goal, _, use_depth_limit=True):
        # Initialize parameters
        self.s_start = s_start
        self.s_goal = s_goal

        # Initialize environment
        self.Env = Env()
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles
        
        # Pre-compute obstacle set for faster lookups
        self.obs_set = set(self.obs)

        # Initialize sets and dictionaries
        self.OPEN = []            # stack for DFS (LIFO)
        self.OPEN_set = set()     # set for O(1) lookups in OPEN
        self.CLOSED = set()       # visited nodes as a set for O(1) lookup
        self.PARENT = dict()      # parent nodes for path reconstruction
        self.g = dict()           # cost to come (needed for shortest path)
        
        # Max depth for DFS to prevent excessive memory usage
        self.depth = {}           # Track depth of each node
        self.max_depth = 1000     # Maximum depth limit (prevent stack overflow)
        self.use_depth_limit = use_depth_limit  # Control whether to use depth limiting
        
        # Performance tracking
        self.collision_time = 0
        self.neighbor_time = 0
        self.node_process_time = 0
        self.nodes_processed = 0
        self.revisited_nodes = 0
        self.closed_ops_time = 0
        self.open_ops_time = 0
        self.dict_ops_time = 0
        self.depth_pruned = 0

    def searching(self):
        """DFS algorithm modified to maintain shortest path information"""
        # Get logger
        logger = logging.getLogger('')
        
        # Initialize start node
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # Cost from start to start is 0
        self.depth[self.s_start] = 0  # Depth of start node is 0
        self.OPEN.append(self.s_start)
        self.OPEN_set.add(self.s_start)
        visited_list = []  # For animation purposes
        
        logger.info("Starting DFS search with optimized implementation...")
        start_time = time.time()
        last_log_time = start_time
        last_nodes_count = 0
        
        # Store the maximum queue size for diagnostics
        max_queue_size = 1
        max_depth_reached = 0

        while self.OPEN:
            # Performance logging every second
            current_time = time.time()
            if current_time - last_log_time > 1.0:
                elapsed = current_time - start_time
                nodes_per_second = (self.nodes_processed - last_nodes_count) / (current_time - last_log_time)
                logger.info(f"[{elapsed:.2f}s] Processed: {self.nodes_processed} nodes | "
                      f"Rate: {nodes_per_second:.1f} nodes/s | "
                      f"Queue: {len(self.OPEN)} | Closed: {len(self.CLOSED)} | "
                      f"Max depth: {max_depth_reached}")
                
                # Track slowdown trends
                if nodes_per_second < 100 and self.nodes_processed > 1000:
                    logger.warning("Processing speed has decreased significantly")
                    logger.info(f"  * CLOSED operations: {self.closed_ops_time:.3f}s")
                    logger.info(f"  * OPEN operations: {self.open_ops_time:.3f}s")
                    logger.info(f"  * Dict operations: {self.dict_ops_time:.3f}s")
                    logger.info(f"  * Depth-pruned nodes: {self.depth_pruned}")
                
                last_log_time = current_time
                last_nodes_count = self.nodes_processed
            
            # Update max queue size
            max_queue_size = max(max_queue_size, len(self.OPEN))
            
            # Node processing timing
            node_start = time.time()
            
            # Get node from OPEN and process it
            s = self.OPEN.pop()  # LIFO for DFS
            
            # Track time for OPEN operations
            open_start = time.time()
            self.OPEN_set.remove(s)
            self.open_ops_time += time.time() - open_start
            
            self.nodes_processed += 1
            
            # Track maximum depth
            current_depth = self.depth[s]
            max_depth_reached = max(max_depth_reached, current_depth)

            # Skip if already in CLOSED (optimization to avoid reprocessing)
            closed_start = time.time()
            if s in self.CLOSED:
                self.revisited_nodes += 1
                self.closed_ops_time += time.time() - closed_start
                continue
                
            # Add to visualization list (no duplicate checking needed now)
            visited_list.append(s)

            # Add to closed set
            self.CLOSED.add(s)
            self.closed_ops_time += time.time() - closed_start

            # Check if goal reached
            if s == self.s_goal:
                logger.info(f"Goal reached in {time.time() - start_time:.3f} seconds")
                logger.info(f"Nodes processed: {self.nodes_processed}, Closed size: {len(self.CLOSED)}")
                logger.info(f"Maximum queue size: {max_queue_size}, Maximum depth: {max_depth_reached}")
                break

            # Skip expanding if max depth reached (only if depth limiting is enabled)
            if self.use_depth_limit and current_depth >= self.max_depth:
                self.depth_pruned += 1
                continue

            # Neighbor processing timing
            neighbor_start = time.time()
            neighbors = self.get_neighbor(s)
            self.neighbor_time += time.time() - neighbor_start
            
            # Explore neighbors - optimized loop
            for s_n in neighbors:
                # Check if neighbor is already processed
                if s_n in self.CLOSED:
                    continue
                    
                # Calculate new cost to this neighbor
                dict_start = time.time()
                new_cost = self.g[s] + self.calculate_distance(s, s_n)
                
                # Only process if better path or unvisited
                needs_update = s_n not in self.g or new_cost < self.g[s_n]
                self.dict_ops_time += time.time() - dict_start
                
                if needs_update:
                    dict_start = time.time()
                    self.g[s_n] = new_cost  # Update cost
                    self.PARENT[s_n] = s    # Update parent
                    self.depth[s_n] = current_depth + 1  # Update depth
                    self.dict_ops_time += time.time() - dict_start

                    # Add to OPEN if not already there
                    open_start = time.time()
                    if s_n not in self.OPEN_set:
                        self.OPEN.append(s_n)
                        self.OPEN_set.add(s_n)
                    self.open_ops_time += time.time() - open_start
            
            # Update node processing time
            self.node_process_time += time.time() - node_start

        # Final performance stats
        search_time = time.time() - start_time
        logger.info(f"\nSearch completed in {search_time:.3f} seconds")
        logger.info(f"Total nodes processed: {self.nodes_processed}")
        logger.info(f"Nodes in CLOSED: {len(self.CLOSED)}")
        logger.info(f"Revisited nodes: {self.revisited_nodes}")
        logger.info(f"Depth-pruned nodes: {self.depth_pruned}")
        logger.info(f"Maximum depth reached: {max_depth_reached}")
        logger.info(f"Collision checking: {self.collision_time:.3f}s ({self.collision_time/search_time*100:.1f}%)")
        logger.info(f"Neighbor generation: {self.neighbor_time:.3f}s ({self.neighbor_time/search_time*100:.1f}%)")
        logger.info(f"Node processing: {self.node_process_time:.3f}s ({self.node_process_time/search_time*100:.1f}%)")
        logger.info(f"CLOSED set operations: {self.closed_ops_time:.3f}s ({self.closed_ops_time/search_time*100:.1f}%)")
        logger.info(f"OPEN queue/set operations: {self.open_ops_time:.3f}s ({self.open_ops_time/search_time*100:.1f}%)")
        logger.info(f"Dictionary operations: {self.dict_ops_time:.3f}s ({self.dict_ops_time/search_time*100:.1f}%)")

        return self.extract_path(self.PARENT), list(visited_list)

    def get_neighbor(self, s):
        """Get valid neighboring nodes - optimized for speed without heuristics"""
        neighbors = []

        # Generate all possible neighbors
        for u in self.u_set:
            s_n = (s[0] + u[0], s[1] + u[1])

            # Check if valid move (not collision)
            collision_start = time.time()
            is_valid = not self.is_collision(s, s_n)
            self.collision_time += time.time() - collision_start
            
            if is_valid:
                neighbors.append(s_n)

        return neighbors

    def calculate_distance(self, s_start, s_goal):
        """Calculate Euclidean distance between two points - for path cost calculation"""
        # Using direct formula is faster than hypot for simple cases
        dx = s_goal[0] - s_start[0]
        dy = s_goal[1] - s_start[1]
        return math.sqrt(dx*dx + dy*dy)

    def is_collision(self, s_start, s_end):
        """Check if there's a collision between two nodes - optimized"""
        # Quick check for endpoint obstacles using the cached set
        if s_start in self.obs_set or s_end in self.obs_set:
            return True

        # Only check diagonal movements (more expensive)
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            # Calculate diagonal passing cells
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            # Check if either diagonal cell is an obstacle using cached set
            if s1 in self.obs_set or s2 in self.obs_set:
                return True

        return False

    def extract_path(self, PARENT):
        """Extract path from parent dictionary - optimized version"""
        # Check if path exists
        if self.s_goal not in PARENT:
            return []

        # Reconstruct path efficiently
        path = [self.s_goal]
        s = self.s_goal

        while s != self.s_start:
            s = PARENT[s]
            path.append(s)

        return path


def main():
    """Main function to run the DFS algorithm"""
    # Set up logging to a file
    logging.basicConfig(
        filename='dfs_performance.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('')
    
    # Check for command line arguments
    visualize = True
    save_gif = True
    compare_modes = False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if 'novis' in sys.argv:
            visualize = False
            logger.info("Running without visualization")
        if 'savegif' in sys.argv:
            save_gif = True
            logger.info("Will save GIF animation")
        if 'compare' in sys.argv:
            compare_modes = True
            logger.info("Comparing with and without depth limit")
    
    s_start = (5, 5)
    s_goal = (45, 25)

    if compare_modes:
        # Run with depth limiting first
        logger.info("RUNNING DFS WITH DEPTH LIMITING")
        logger.info(f"Running optimized DFS from {s_start} to {s_goal}")
        total_start = time.time()
        
        dfs = DFS(s_start, s_goal, None, use_depth_limit=True)
        path_with_limit, visited_with_limit = dfs.searching()
        
        # Report results for depth-limited run
        total_time_with_limit = time.time() - total_start
        logger.info(f"Total execution time WITH depth limit: {total_time_with_limit:.3f} seconds")
        logger.info(f"Path length: {len(path_with_limit)}")
        logger.info(f"Total nodes visited: {len(visited_with_limit)}")
        logger.info(f"Depth-pruned nodes: {dfs.depth_pruned}")
        
        # Run without depth limiting for comparison
        logger.info("\n\nRUNNING DFS WITHOUT DEPTH LIMITING")
        logger.info(f"Running standard DFS from {s_start} to {s_goal}")
        total_start = time.time()
        
        dfs = DFS(s_start, s_goal, None, use_depth_limit=False)
        path_no_limit, visited_no_limit = dfs.searching()
        
        # Report results for unlimited run
        total_time_no_limit = time.time() - total_start
        logger.info(f"Total execution time WITHOUT depth limit: {total_time_no_limit:.3f} seconds")
        logger.info(f"Path length: {len(path_no_limit)}")
        logger.info(f"Total nodes visited: {len(visited_no_limit)}")
        
        # Compare results
        logger.info("\n\nCOMPARISON RESULTS:")
        logger.info(f"Time with depth limit: {total_time_with_limit:.3f}s")
        logger.info(f"Time without depth limit: {total_time_no_limit:.3f}s")
        if total_time_no_limit > 0:
            logger.info(f"Speed improvement: {total_time_no_limit/total_time_with_limit:.1f}x faster with depth limiting")
        
        # Only visualize the depth-limited version if requested
        if visualize:
            logger.info("Generating visualization...")
            plot = Plotting(s_start, s_goal)
            plot.animation(path_with_limit, visited_with_limit, "002_dfs", save_gif=save_gif)
    else:
        # Run standard optimized version
        logger.info(f"Running DFS from {s_start} to {s_goal}")
        total_start = time.time()
        
        dfs = DFS(s_start, s_goal, None, use_depth_limit=True)
        path, visited = dfs.searching()
        
        # Only visualize if requested
        if visualize:
            logger.info("Generating visualization...")
            plot = Plotting(s_start, s_goal)
            plot.animation(path, visited, "002_dfs", save_gif=save_gif)
        
        # Report total time
        total_time = time.time() - total_start
        logger.info(f"Total execution time: {total_time:.3f} seconds")
        logger.info(f"Path length: {len(path)}")
        logger.info(f"Total nodes visited: {len(visited)}")


if __name__ == '__main__':
    main()
