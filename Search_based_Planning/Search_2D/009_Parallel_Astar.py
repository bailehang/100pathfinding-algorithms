import os
import sys
import math
import heapq
import numpy as np
import time
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env


class EnhancedParallelAStar:
    def __init__(self, s_start, s_goals, heuristic_type, num_regions=4):
        self.s_start = s_start
        self.s_goals = s_goals if isinstance(s_goals, list) else [s_goals]
        self.heuristic_type = heuristic_type
        self.num_regions = num_regions

        self.Env = env.Env()
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


class EnhancedParallelAStarPlotting(plotting.Plotting):
    def __init__(self, xI, xG):
        if isinstance(xG, list):
            super().__init__(xI, xG[0])
            self.xG_multiple = xG
        else:
            super().__init__(xI, xG)
            self.xG_multiple = [xG]

    def animation_enhanced_parallel_astar(self, goal_paths, visited, region_lines, name):
        self.plot_grid(name)

        for line in region_lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                     color='blue', linestyle='--', alpha=0.5)

        plt.scatter(self.xI[0], self.xI[1], color='green', s=100, zorder=5, label='Start')

        num_paths = len(goal_paths)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_paths))
        for i, goal in enumerate(self.xG_multiple):
            color = colors[i]
            plt.scatter(goal[0], goal[1], color=color, s=100, marker='*',
                        label=f'Goal {i + 1}', edgecolors='black', zorder=5)

        self.plot_visited(visited, 'gray')

        for i, (goal, (path, _)) in enumerate(goal_paths.items()):
            if not path:
                continue

            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]

            color = colors[i]
            plt.plot(path_x, path_y, linewidth=2, color=color,
                     label=f'Path to Goal {i + 1} {goal}')

            plt.scatter(path_x, path_y, color=color, s=30, alpha=0.7)

        plt.legend(loc='upper right', fontsize='small')

        plt.show()


def run_test_without_gui():
    s_start = (5, 5)
    s_goals = [(45, 25), (25, 5), (45, 15)]

    parallel_astar = EnhancedParallelAStar(s_start, s_goals, "euclidean", 4)

    start_time = time.time()
    goal_paths, visited = parallel_astar.searching()
    end_time = time.time()

    total_path_length = 0
    for goal, (path, _) in goal_paths.items():
        if path:
            path_length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                              for i in range(len(path) - 1))
            total_path_length += path_length

    with open("parallel_astar_results.txt", "w") as f:
        f.write("=== Enhanced Parallel A* Results ===\n")
        f.write(f"Start: {s_start}\n")
        f.write(f"Goals: {s_goals}\n")
        f.write(f"Search completed in {end_time - start_time:.4f} seconds\n")
        f.write(f"Paths found to {len(goal_paths)}/{len(s_goals)} goals\n")
        f.write(f"Total nodes expanded: {parallel_astar.expanded_count}\n")
        f.write(f"Total nodes visited: {len(visited)}\n\n")

        for goal, (path, _) in goal_paths.items():
            f.write(f"Path to {goal}:\n")
            for i, point in enumerate(path):
                f.write(f"  Step {i}: {point}\n")
            f.write("\n")

    return goal_paths, visited


def main():
    print("Starting Enhanced Parallel A* simplified demo...")

    s_start = (5, 5)
    s_goals = [(45, 25), (25, 5), (45, 15)]

    print(f"Start: {s_start}")
    print(f"Goals: {s_goals}")

    all_paths = {}
    all_visited = []

    for i, goal in enumerate(s_goals):
        print(f"\nFinding path to goal {i + 1}: {goal}")

        astar_module = __import__('005_Astar', fromlist=['AStar'])
        AStar = astar_module.AStar
        astar = AStar(s_start, goal, "euclidean")

        path, visited = astar.searching()

        if path:
            path_length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                              for i in range(len(path) - 1))
            print(f"Path found, length: {path_length:.2f}, steps: {len(path)}")
            all_paths[goal] = (path, visited)
            all_visited.extend(visited)
        else:
            print(f"No path found to {goal}")

    print("\nVisualizing results with start and goals in one window...")
    try:
        plot = EnhancedParallelAStarPlotting(s_start, s_goals)

        region_lines = []

        plot.animation_enhanced_parallel_astar(all_paths, all_visited, region_lines, "Multi-Goal A*")

    except Exception as e:
        print(f"Visualization failed: {e}")

    print("Demo completed successfully.")


if __name__ == '__main__':
    main()