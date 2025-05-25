"""
S-Theta* 2D: Low steering path-planning algorithm
@author: clark bai
"""

import io
import os
import sys
import math
import heapq
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

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

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
    """Plotting class for visualization with GIF generation capability"""

    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = (6, 4)

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


class SThetaStar:
    """
    S-Theta*: Low steering path-planning algorithm
    S-Theta* improves on Theta* by considering steering constraints and minimizing steering changes,
    producing smoother paths suitable for vehicles with turning constraints.
    """
    def __init__(self, s_start, s_goal, heuristic_type, steering_weight=1.0):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.steering_weight = steering_weight  # Weight for steering cost

        self.Env = Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.heading = dict()  # heading direction at each node
        
        # For visualization
        self.los_checks = []
        self.fig = plt.figure()
        self.plot = Plotting(s_start, s_goal)
        
        # Current search state
        self.current_path = []
        self.current_visited = []
        self.current_los_checks = []

    def searching(self):
        """
        S-Theta* path searching with gif generation.
        :return: path, visited order
        """
        print("Starting S-Theta* algorithm with GIF generation...")
        
        # Initialize plot
        self.plot.plot_grid("S-Theta*")

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        self.heading[self.s_start] = (0, 0)  # Initial heading (no specific direction)
        
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            self.current_visited.append(s)

            # Update current path and visualization
            if s == self.s_goal:
                self.current_path = self.extract_path(self.PARENT)
            else:
                # Show path from start to current node
                temp_path = self.extract_temp_path(s)
                self.current_path = temp_path
            
            # Update visualization periodically
            if len(self.CLOSED) % 5 == 0 or s == self.s_goal:
                self.update_plot()

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                # Calculate heading to neighbor
                heading_to_neighbor = self.calculate_heading(s, s_n)
                
                # Path 1 - Regular path through current node
                new_cost_regular = self.g[s] + self.cost(s, s_n)
                
                # Add steering cost for Path 1
                if s != self.s_start:
                    steering_change_regular = self.steering_cost(self.heading[s], heading_to_neighbor)
                    new_cost_regular += self.steering_weight * steering_change_regular
                
                # Path 2 - Try to use line-of-sight from parent
                new_cost_los = math.inf
                
                # Check line-of-sight from parent
                los_result = self.line_of_sight(self.PARENT[s], s_n)
                self.los_checks.append((self.PARENT[s], s_n, los_result))
                self.current_los_checks.append((self.PARENT[s], s_n, los_result))
                
                if los_result:
                    # Line-of-sight exists, consider path from parent
                    heading_from_parent = self.calculate_heading(self.PARENT[s], s_n)
                    new_cost_los = self.g[self.PARENT[s]] + self.cost(self.PARENT[s], s_n)
                    
                    # Add steering cost for Path 2
                    if self.PARENT[s] != self.s_start:
                        steering_change_los = self.steering_cost(self.heading[self.PARENT[s]], heading_from_parent)
                        new_cost_los += self.steering_weight * steering_change_los

                # Choose the better path
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # Compare costs and update if better path found
                if new_cost_regular <= new_cost_los:
                    # Path 1 is better or equal
                    if new_cost_regular < self.g[s_n]:
                        self.g[s_n] = new_cost_regular
                        self.PARENT[s_n] = s
                        self.heading[s_n] = heading_to_neighbor
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
                else:
                    # Path 2 is better
                    if new_cost_los < self.g[s_n]:
                        self.g[s_n] = new_cost_los
                        self.PARENT[s_n] = self.PARENT[s]  # Skip a step in the path
                        self.heading[s_n] = heading_from_parent
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # Final update and generate GIF
        self.update_plot(final=True)
        
        path = self.extract_path(self.PARENT)
        print(f"Path found with {len(path)} nodes, visited {len(self.CLOSED)} nodes")
        
        # Generate GIF
        print("Generating GIF animation...")
        self.plot.save_animation_as_gif("024_STheta_star")
        
        return path, self.CLOSED

    def calculate_heading(self, s_start, s_end):
        """
        Calculate heading direction from s_start to s_end
        :param s_start: start node
        :param s_end: end node
        :return: normalized heading vector
        """
        dx = s_end[0] - s_start[0]
        dy = s_end[1] - s_start[1]
        distance = math.hypot(dx, dy)
        
        if distance == 0:
            return (0, 0)  # No specific direction if same point
        
        return (dx/distance, dy/distance)

    def steering_cost(self, heading1, heading2):
        """
        Calculate steering cost between two headings
        :param heading1: first heading vector
        :param heading2: second heading vector
        :return: steering cost (angle change)
        """
        # Calculate dot product
        dot_product = heading1[0] * heading2[0] + heading1[1] * heading2[1]
        # Clamp to avoid numerical issues
        dot_product = max(-1, min(1, dot_product))
        # Calculate angle in radians
        angle = math.acos(dot_product)
        
        return angle

    def update_plot(self, final=False):
        """
        Update the plot to show current search state
        """
        # Clear current figure
        plt.cla()
        
        # Redraw grid
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title("S-Theta*")
        plt.axis("equal")
        
        # Draw visited nodes
        if self.current_visited:
            for node in self.current_visited:
                if node != self.s_start and node != self.s_goal:
                    plt.plot(node[0], node[1], color='gray', marker='o')
        
        # Draw line-of-sight checks
        if self.current_los_checks:
            for start, end, result in self.current_los_checks:
                color = 'g' if result else 'r'
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.3)
        
        # Draw current path
        if self.current_path:
            path_x = [self.current_path[i][0] for i in range(len(self.current_path))]
            path_y = [self.current_path[i][1] for i in range(len(self.current_path))]
            plt.plot(path_x, path_y, linewidth='3', color='r')
            
            # Draw heading arrows for current path (optional)
            if len(self.current_path) > 1:
                for i in range(len(self.current_path) - 1):
                    if self.current_path[i] in self.heading and self.heading[self.current_path[i]] != (0, 0):
                        x, y = self.current_path[i]
                        dx, dy = self.heading[self.current_path[i]]
                        # Scale arrows
                        scale = 2.0
                        plt.arrow(x, y, dx*scale, dy*scale, head_width=0.5, head_length=0.5, 
                                 fc='blue', ec='blue', alpha=0.5)
        
        # Redraw start and goal to ensure they are on top
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        
        # Capture frame
        self.plot.capture_frame()
        
        # Update figure
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        
        # Pause longer for final result
        if final:
            plt.pause(0.5)
        else:
            plt.pause(0.01)
    
    def extract_temp_path(self, current):
        """
        Extract temporary path from start to current node
        """
        path = [current]
        s = current
        
        while s != self.s_start:
            s = self.PARENT[s]
            path.append(s)
        
        return list(reversed(path))

    def get_neighbor(self, s):
        """
        Find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        nei_list = []
        for u in self.u_set:
            s_next = (s[0] + u[0], s[1] + u[1])
            # Check boundary constraints
            if (0 <= s_next[0] < self.Env.x_range and 
                0 <= s_next[1] < self.Env.y_range and
                s_next not in self.obs):  # Filter out obstacles and boundary violations
                nei_list.append(s_next)
                
        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        """
        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        Check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # Check if points are within grid boundaries
        x_range, y_range = self.Env.x_range, self.Env.y_range
        
        # Check boundary constraints
        if (s_start[0] < 0 or s_start[0] >= x_range or 
            s_start[1] < 0 or s_start[1] >= y_range or
            s_end[0] < 0 or s_end[0] >= x_range or
            s_end[1] < 0 or s_end[1] >= y_range):
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        # Basic check for diagonal segments
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        # Bresenham's line algorithm for more thorough checking
        x0, y0 = s_start
        x1, y1 = s_end
        
        # If the line is steep, transpose the grid
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        # Swap points if needed to ensure x increases
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        y = y0
        
        # Determine step direction
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        
        # Check each point on the line
        for x in range(x0, x1 + 1):
            if steep:
                # If steep, the coordinates are transposed
                if (y, x) in self.obs:
                    return True
            else:
                if (x, y) in self.obs:
                    return True
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        
        return False

    def line_of_sight(self, s_start, s_end):
        """
        Check if there is a line-of-sight between two nodes
        :param s_start: start node
        :param s_end: end node
        :return: True if line-of-sight exists
        """
        return not self.is_collision(s_start, s_end)

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(reversed(path))

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """
        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def path_smoothing(self, path):
        """
        Apply path smoothing to reduce unnecessary heading changes
        :param path: original path
        :return: smoothed path
        """
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]
        current = path[0]
        i = 1
        
        while i < len(path) - 1:
            # Try to connect current node to nodes further ahead in the path
            for j in range(len(path) - 1, i, -1):
                if not self.is_collision(current, path[j]):
                    # Found a valid shortcut
                    current = path[j]
                    smoothed_path.append(current)
                    i = j + 1
                    break
            else:
                # No shortcuts found, add the next node
                current = path[i]
                smoothed_path.append(current)
                i += 1
        
        # Make sure goal is in the path
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
            
        return smoothed_path


def main():
    """
    S-Theta*: Low Steering Path-Planning Algorithm
    
    S-Theta* extends Theta* by considering steering constraints and minimizing steering changes.
    This makes it especially suitable for vehicle navigation where smooth paths with minimal
    heading changes are preferred. The algorithm considers not just path length but also
    steering costs to produce paths that are more suitable for vehicles with turning constraints.
    
    Like Theta*, S-Theta* allows for paths that can go through any angle, not just along grid edges.
    It checks for line-of-sight between non-adjacent nodes but adds a penalty for steering changes,
    resulting in smoother, more natural paths with fewer heading changes.
    
    The steering_weight parameter controls how much to prioritize minimizing steering changes
    versus path length. Higher values will result in smoother paths with fewer turns,
    potentially at the cost of slightly longer paths.
    """
    s_start = (5, 5)
    s_goal = (45, 25)
    
    # Create S-Theta* instance with euclidean heuristic and steering weight of 2.0
    s_theta_star = SThetaStar(s_start, s_goal, "euclidean", steering_weight=2.0)
    path, visited = s_theta_star.searching()
    
    print("S-Theta* algorithm completed successfully!")
    print(f"Final path length: {len(path)} nodes")
    print(f"Total nodes visited: {len(visited)} nodes")
    
    # Optional: perform additional path smoothing
    # smoothed_path = s_theta_star.path_smoothing(path)
    # s_theta_star.plot.plot_path(smoothed_path)
    # plt.title("S-Theta* with Additional Path Smoothing")
    # plt.show()


if __name__ == '__main__':
    main()
