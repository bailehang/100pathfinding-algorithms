"""
Parallel A* Pathfinding Algorithm
@author: Modified by Cline from original code
"""

import io
import os
import math
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque


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
        """
        Initialize the plotting class
        :param xI: start point [x, y]
        :param xG: goal point [x, y] or list of goal points
        """
        self.xI = xI
        
        if isinstance(xG, list):
            self.xG = xG[0]
            self.xG_multiple = xG
        else:
            self.xG = xG
            self.xG_multiple = [xG]
            
        self.frames = []
        self.fig_size = (6, 4)
        
        self.env = Env()
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        """Update obstacles"""
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
        # Create figure with fixed size
        plt.figure(figsize=self.fig_size, dpi=100, clear=True)
        
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
                plt.pause(0.01)
                self.capture_frame()

        plt.pause(0.1)
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
        buf = io.BytesIO()
        
        # Get the current figure
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Save the figure to a buffer with a standard DPI
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Open the image using PIL and convert to RGB
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        
        # Convert to numpy array
        image = np.array(img_rgb)
        
        # Add to frames
        self.frames.append(image)
        
        # Close the buffer
        buf.close()

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with consistent size"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames captured: {len(self.frames)}")
        
        # Verify all frames have the same dimensions
        if self.frames:
            first_frame_shape = self.frames[0].shape
            for i, frame in enumerate(self.frames):
                if frame.shape != first_frame_shape:
                    print(f"WARNING: Frame {i} has inconsistent shape: {frame.shape} vs {first_frame_shape}")
                    # Resize inconsistent frames to match the first frame
                    resized_frame = np.array(Image.fromarray(frame).resize(
                        (first_frame_shape[1], first_frame_shape[0]), 
                        Image.LANCZOS))
                    self.frames[i] = resized_frame
        
        # Check if frames list is not empty before saving
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

                # Save with proper disposal method to avoid artifacts
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
                    print(f"File exists with size: {os.path.getsize(gif_path) / 1024:.2f} KB")
                else:
                    print("WARNING: File does not exist after saving!")
            except Exception as e:
                print(f"Error during GIF creation: {e}")
        else:
            print("No frames to save!")

        # Close the figure
        plt.close()


class AStar:
    """A* path planning algorithm"""
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()  # Initialize environment
        self.u_set = self.Env.motions  # feasible movements
        self.obs = self.Env.obs  # obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set
        self.PARENT = dict()  # parent dictionary
        self.g = dict()  # cost to come

    def searching(self):
        """A* search algorithm"""
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # goal reached
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # update cost if better path found
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(), self.CLOSED

    def get_neighbor(self, s):
        """Get valid neighboring nodes"""
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set
                if 0 <= s[0] + u[0] < self.Env.x_range
                and 0 <= s[1] + u[1] < self.Env.y_range
                and (s[0] + u[0], s[1] + u[1]) not in self.obs]

    def f_value(self, s):
        """Calculate f value (cost to come + heuristic)"""
        return self.g[s] + self.heuristic(s, self.s_goal)

    def heuristic(self, s, goal):
        """Calculate heuristic based on type"""
        if self.heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:  # euclidean by default
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

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

    def extract_path(self):
        """Extract path from parent dictionary"""
        if self.s_goal not in self.PARENT:
            return []
            
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = self.PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(reversed(path))


class EnhancedParallelAStar:
    def __init__(self, s_start, s_goals, heuristic_type, num_regions=4):
        self.s_start = s_start
        self.s_goals = s_goals if isinstance(s_goals, list) else [s_goals]
        self.heuristic_type = heuristic_type
        self.num_regions = num_regions

        self.Env = Env()
        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range

        self.regions = self.divide_map()

        self.heuristic_cache = {}
        self.path_segments = {}
        self.node_costs = {}

        self.region_paths = {}
        self.region_visited = {}
        self.expanded_count = 0
        self.goal_paths = {}

    def divide_map(self):
        n_rows = int(math.sqrt(self.num_regions))
        while self.num_regions % n_rows != 0:
            n_rows -= 1
        n_cols = self.num_regions // n_rows

        region_width = math.ceil(self.x_range / n_cols)
        region_height = math.ceil(self.y_range / n_rows)

        regions = {}

        region_id = 0
        for row in range(n_rows):
            for col in range(n_cols):
                x_min = col * region_width
                y_min = row * region_height
                x_max = min((col + 1) * region_width - 1, self.x_range - 1)
                y_max = min((row + 1) * region_height - 1, self.y_range - 1)

                regions[region_id] = {
                    'boundaries': (x_min, y_min, x_max, y_max),
                    'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                    'neighbors': []
                }

                region_id += 1

        for i in regions:
            for j in regions:
                if i != j:
                    i_xmin, i_ymin, i_xmax, i_ymax = regions[i]['boundaries']
                    j_xmin, j_ymin, j_xmax, j_ymax = regions[j]['boundaries']

                    if ((i_xmin <= j_xmax and i_xmax >= j_xmin) and
                            ((i_ymax + 1 == j_ymin) or (j_ymax + 1 == i_ymin))):
                        regions[i]['neighbors'].append(j)
                    elif ((i_ymin <= j_ymax and i_ymax >= j_ymin) and
                          ((i_xmax + 1 == j_xmin) or (j_xmax + 1 == i_xmin))):
                        regions[i]['neighbors'].append(j)

        return regions

    def get_region_for_node(self, node):
        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']
            if x_min <= node[0] <= x_max and y_min <= node[1] <= y_max:
                return region_id
        return -1

    def identify_border_nodes(self):
        border_nodes = {}

        for region_id, region in self.regions.items():
            for neighbor_id in region['neighbors']:
                if (region_id, neighbor_id) not in border_nodes:
                    border_nodes[(region_id, neighbor_id)] = []

        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']

            border_cells = []

            for y in range(y_min, y_max + 1):
                border_cells.append((x_min, y))

            for y in range(y_min, y_max + 1):
                border_cells.append((x_max, y))

            for x in range(x_min + 1, x_max):
                border_cells.append((x, y_max))

            for x in range(x_min + 1, x_max):
                border_cells.append((x, y_min))

            border_cells = [cell for cell in border_cells if cell not in self.obs]

            for cell in border_cells:
                x, y = cell
                for u in self.u_set:
                    nx, ny = x + u[0], y + u[1]

                    if not (0 <= nx < self.x_range and 0 <= ny < self.y_range) or (nx, ny) in self.obs:
                        continue

                    neighbor_region = self.get_region_for_node((nx, ny))
                    if neighbor_region != region_id and neighbor_region != -1:
                        if (region_id, neighbor_region) in border_nodes:
                            border_nodes[(region_id, neighbor_region)].append(cell)

        for key in border_nodes:
            border_nodes[key] = list(set(border_nodes[key]))

        return border_nodes

    def heuristic(self, s, goal):
        cache_key = (s, goal)
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]

        if self.heuristic_type == "manhattan":
            h_value = abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            h_value = math.hypot(goal[0] - s[0], goal[1] - s[1])

        self.heuristic_cache[cache_key] = h_value
        return h_value

    def multi_goal_heuristic(self, s):
        return min(self.heuristic(s, goal) for goal in self.s_goals)

    def get_neighbors(self, s):
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set
                if 0 <= s[0] + u[0] < self.x_range
                and 0 <= s[1] + u[1] < self.y_range
                and (s[0] + u[0], s[1] + u[1]) not in self.obs]

    def cost(self, s_start, s_goal):
        if self.is_collision(s_start, s_goal):
            return float('inf')

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
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

    def extract_path(self, parent, start, goal):
        if goal not in parent:
            return []

        path = [goal]
        current = goal

        while current != start:
            current = parent[current]
            path.append(current)

        return list(reversed(path))

    def store_path_segment(self, start, end, path, cost):
        if not path:
            return

        self.path_segments[(start, end)] = (path, cost)

        reversed_path = list(reversed(path))
        self.path_segments[(end, start)] = (reversed_path, cost)

        if start not in self.node_costs or cost < self.node_costs[start]:
            self.node_costs[start] = cost

        if end not in self.node_costs or cost < self.node_costs[end]:
            self.node_costs[end] = cost

    def check_path_segment(self, start, end):
        if (start, end) in self.path_segments:
            return self.path_segments[(start, end)]
        return None, float('inf')

    def region_a_star(self, region_id, start_node, goal_nodes, region_boundaries):
        x_min, y_min, x_max, y_max = region_boundaries

        open_set = []
        closed_set = []
        parent = {start_node: start_node}
        g = {start_node: 0}

        for goal in goal_nodes:
            g[goal] = float('inf')

        heapq.heappush(open_set, (self.multi_goal_heuristic(start_node), start_node))

        goals_reached = {}
        visited_nodes = []

        while open_set and len(goals_reached) < len(goal_nodes):
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.append(current)
            visited_nodes.append(current)
            self.expanded_count += 1

            if current in goal_nodes and current not in goals_reached:
                path = self.extract_path(parent, start_node, current)
                goals_reached[current] = (path, g[current])

                self.store_path_segment(start_node, current, path, g[current])

                if len(goals_reached) == len(goal_nodes):
                    break

            for s_n in self.get_neighbors(current):
                nx, ny = s_n

                if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                    continue

                path_segment, segment_cost = self.check_path_segment(current, s_n)

                if path_segment:
                    new_cost = g[current] + segment_cost
                else:
                    new_cost = g[current] + self.cost(current, s_n)

                if s_n not in g or new_cost < g[s_n]:
                    g[s_n] = new_cost
                    parent[s_n] = current

                    self.node_costs[s_n] = new_cost

                    f_value = new_cost + self.multi_goal_heuristic(s_n)
                    heapq.heappush(open_set, (f_value, s_n))

        result = {}
        for goal, (path, _) in goals_reached.items():
            result[goal] = (path, visited_nodes)

        return result

    def search_global(self):
        start_region = self.get_region_for_node(self.s_start)
        goal_regions = {goal: self.get_region_for_node(goal) for goal in self.s_goals}

        border_nodes = self.identify_border_nodes()

        results = {}
        all_visited = []

        for goal, goal_region in goal_regions.items():
            if start_region == goal_region:
                region_results = self.region_a_star(
                    start_region,
                    self.s_start,
                    [goal],
                    self.regions[start_region]['boundaries']
                )

                if goal in region_results:
                    path, visited = region_results[goal]
                    results[goal] = (path, visited)
                    all_visited.extend(visited)
                    self.region_paths[start_region] = path
                    self.region_visited[start_region] = visited
                continue

            best_path = []
            best_cost = float('inf')
            path_visited = []

            if goal_region in self.regions[start_region]['neighbors']:
                border_points = border_nodes.get((start_region, goal_region), [])

                for border in border_points:
                    start_to_border_results = self.region_a_star(
                        start_region,
                        self.s_start,
                        [border],
                        self.regions[start_region]['boundaries']
                    )

                    if border not in start_to_border_results:
                        continue

                    start_to_border_path, start_to_border_visited = start_to_border_results[border]

                    border_to_goal_results = self.region_a_star(
                        goal_region,
                        border,
                        [goal],
                        self.regions[goal_region]['boundaries']
                    )

                    if goal not in border_to_goal_results:
                        continue

                    border_to_goal_path, border_to_goal_visited = border_to_goal_results[goal]

                    total_cost = 0
                    for i in range(len(start_to_border_path) - 1):
                        total_cost += self.cost(start_to_border_path[i], start_to_border_path[i + 1])

                    for i in range(len(border_to_goal_path) - 1):
                        total_cost += self.cost(border_to_goal_path[i], border_to_goal_path[i + 1])

                    if total_cost < best_cost:
                        best_path = start_to_border_path[:-1] + border_to_goal_path
                        best_cost = total_cost
                        path_visited = start_to_border_visited + border_to_goal_visited

            if not best_path:
                region_sequence = self.find_region_sequence(start_region, goal_region)

                if region_sequence:
                    current_node = self.s_start
                    current_path = []
                    current_visited = []

                    for i in range(len(region_sequence) - 1):
                        current_region = region_sequence[i]
                        next_region = region_sequence[i + 1]

                        border_points = border_nodes.get((current_region, next_region), [])

                        if not border_points:
                            break

                        border = border_points[0]

                        region_results = self.region_a_star(
                            current_region,
                            current_node,
                            [border],
                            self.regions[current_region]['boundaries']
                        )

                        if border not in region_results:
                            break

                        segment_path, segment_visited = region_results[border]

                        if current_path:
                            current_path = current_path[:-1] + segment_path
                        else:
                            current_path = segment_path

                        current_visited.extend(segment_visited)
                        current_node = border

                    if current_node != self.s_start:
                        final_region = region_sequence[-1]
                        final_results = self.region_a_star(
                            final_region,
                            current_node,
                            [goal],
                            self.regions[final_region]['boundaries']
                        )

                        if goal in final_results:
                            final_path, final_visited = final_results[goal]
                            current_path = current_path[:-1] + final_path
                            current_visited.extend(final_visited)

                            best_path = current_path
                            path_visited = current_visited

            if best_path:
                results[goal] = (best_path, path_visited)
                all_visited.extend(path_visited)

        return results, all_visited

    def find_region_sequence(self, start_region, goal_region):
        if start_region == goal_region:
            return [start_region]

        queue = [(start_region, [start_region])]
        visited = {start_region}

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.regions[current]['neighbors']:
                if neighbor == goal_region:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def searching(self):
        start_time = time.time()

        self.expanded_count = 0
        self.region_paths = {}
        self.region_visited = {}
        self.heuristic_cache = {}
        self.path_segments = {}
        self.node_costs = {}
        self.goal_paths = {}

        goal_paths, all_visited = self.search_global()

        self.goal_paths = goal_paths

        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds")

        return goal_paths, all_visited

    def visualize_regions(self):
        region_lines = []

        for region_id, region in self.regions.items():
            x_min, y_min, x_max, y_max = region['boundaries']

            region_lines.append([(x_min, y_min), (x_max, y_min)])
            region_lines.append([(x_min, y_min), (x_min, y_max)])
            region_lines.append([(x_max, y_min), (x_max, y_max)])
            region_lines.append([(x_min, y_max), (x_max, y_max)])

        return region_lines


class ParallelAStarPlotting(Plotting):
    """Extended plotting class for visualizing parallel A* results"""
    def __init__(self, xI, xG):
        """
        Initialize the plotting class for parallel A*
        :param xI: start point [x, y]
        :param xG: goal point or list of goal points
        """
        if isinstance(xG, list):
            super().__init__(xI, xG[0])
            self.xG_multiple = xG
        else:
            super().__init__(xI, xG)
            self.xG_multiple = [xG]

    def animation_parallel_astar(self, goal_paths, visited, region_lines, name, save_gif=False):
        """Animate parallel A* search with multiple goals and regions"""
        self.plot_grid(name)

        # Plot region boundaries
        for line in region_lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                     color='blue', linestyle='--', alpha=0.5)

        # Plot start point
        plt.scatter(self.xI[0], self.xI[1], color='green', s=100, zorder=5, label='Start')

        # Plot all goal points
        for i, goal in enumerate(self.xG_multiple):
            plt.scatter(goal[0], goal[1], color='red', s=100, marker='*',
                      label=f'Goal {i + 1}' if i == 0 else "", edgecolors='black', zorder=5)

        # Plot visited nodes (if any)
        if visited:
            self.plot_visited(visited, 'gray')

        # Plot paths to each goal with different colors if there are any paths
        if goal_paths:
            num_paths = len(goal_paths)
            colors = plt.cm.rainbow(np.linspace(0, 1, num_paths))
            
            for i, (goal, (path, _)) in enumerate(goal_paths.items()):
                if not path:
                    continue

                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]

                color = colors[i]
                plt.plot(path_x, path_y, linewidth=2, color=color,
                       label=f'Path to Goal {i + 1}')

                # Add points along the path
                plt.scatter(path_x, path_y, color=color, s=30, alpha=0.7)
        else:
            plt.title(f"{name} - No paths found")

        plt.legend(loc='upper right', fontsize='small')
        
        # Capture the final frame and save gif if requested
        self.capture_frame()
        plt.pause(0.5)
        
        if save_gif:
            self.save_animation_as_gif(name)
            
        plt.show()


def main():
    """Main function to run the Parallel A* algorithm"""
    print("Starting Parallel A* search...")
    
    # Set parameters
    s_start = (5, 5)
    s_goals = [(45, 25), (25, 5), (45, 15)]
    num_regions = 4
    
    print(f"Start: {s_start}")
    print(f"Goals: {s_goals}")
    print(f"Using {num_regions} regions")
    
    # Try parallel A* first
    parallel_astar = EnhancedParallelAStar(s_start, s_goals, "euclidean", num_regions)
    goal_paths, visited = parallel_astar.searching()
    
    # Print statistics
    print(f"Paths found to {len(goal_paths)}/{len(s_goals)} goals")
    print(f"Total nodes expanded: {parallel_astar.expanded_count}")
    print(f"Total nodes visited: {len(visited)}")
    
    # If no paths found with parallel approach, use regular A* for each goal
    if not goal_paths:
        print("Parallel A* failed to find paths. Falling back to individual A* searches...")
        all_paths = {}
        all_visited = []
        
        # Use regular A* to find paths to each goal
        for i, goal in enumerate(s_goals):
            print(f"Finding path to goal {i + 1}: {goal}")
            astar = AStar(s_start, goal, "euclidean")
            path, visited_nodes = astar.searching()
            
            if path:
                path_length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                                for i in range(len(path) - 1))
                print(f"Path found, length: {path_length:.2f}, steps: {len(path)}")
                all_paths[goal] = (path, visited_nodes)
                all_visited.extend(visited_nodes)
            else:
                print(f"No path found to {goal}")
        
        # Use regular A* results if they're better
        if len(all_paths) > len(goal_paths):
            goal_paths = all_paths
            visited = all_visited
            print(f"Using A* paths: found {len(goal_paths)}/{len(s_goals)} paths")
    
    # Get region lines for visualization
    region_lines = parallel_astar.visualize_regions()
    
    # Initialize plotting and visualize results
    plot = ParallelAStarPlotting(s_start, s_goals)
    plot.animation_parallel_astar(goal_paths, visited, region_lines, "009_Parallel_Astar", save_gif=True)
    
    print("Search completed successfully!")


if __name__ == '__main__':
    main()
