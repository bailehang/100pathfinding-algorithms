"""
LPA_star 2D
@author: huiming zhou
@author: clark bai
"""

import io
import math
import os
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
    """Plotting class for visualization with GIF generation capability"""

    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = (6, 4)
        self.count = 0

    def update_obs(self, obs):
        self.obs = obs

    def plot_grid(self, name):
        """Plot the grid with obstacles, start and goal points"""
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs", label="Start")
        plt.plot(self.xG[0], self.xG[1], "gs",  label="Goal")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.xlim(-2, self.env.x_range)
        plt.ylim(-2, self.env.y_range)

    def plot_visited(self, visited):
        """Plot visited nodes during search"""
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            if x != self.xI and x != self.xG:  # Don't cover start and goal
                plt.plot(x[0], x[1], marker='s', color=color[self.count])

    def plot_path(self, path):
        """Plot the final path"""
        if len(path) > 1:
            px = [x[0] for x in path]
            py = [x[1] for x in path]
            plt.plot(px, py, linewidth=3, color='red', alpha=0.8, label="Path")
            
        # Redraw start and goal to ensure they're visible
        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

    def capture_frame(self):
        """Capture current frame for GIF generation"""
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

    def save_animation_as_gif(self, name, fps=2):
        """Save frames as a GIF animation"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")
        print(f"Number of frames captured: {len(self.frames)}")
        
        # Check if frames list is not empty before saving
        if self.frames:
            try:
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

                # Convert NumPy arrays to PIL Images
                print("Converting frames to PIL Images...")
                frames_p = []
                for i, frame in enumerate(self.frames):
                    try:
                        img = Image.fromarray(frame)
                        img_p = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                        frames_p.append(img_p)
                        if i % 5 == 0:
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


class LPAStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()
        self.Plot = Plotting(self.s_start, self.s_goal)

        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.g, self.rhs, self.U = {}, {}, {}

        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_start] = 0
        self.U[self.s_start] = self.CalculateKey(self.s_start)
        self.visited = set()
        self.count = 0
        self.interaction_count = 0

        self.fig = plt.figure(figsize=self.Plot.fig_size)

    def run(self):
        print("Starting LPA* with interactive obstacle modification and GIF generation...")
        print("Click on the plot to add/remove obstacles. Close the window when finished to generate GIF.")
        
        # Initial computation and visualization
        plt.clf()
        self.Plot.plot_grid("Lifelong Planning A* (Interactive)")
        self.ComputeShortestPath()
        self.plot_path(self.extract_path())
        self.Plot.capture_frame()
        
        # Set up interactive callback
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        
        # Show plot and wait for user interaction
        plt.show()
        
        # Generate GIF after window is closed
        print("Generating GIF from captured frames...")
        self.Plot.save_animation_as_gif("013_LPAstar", fps=1.5)
        print("GIF generation completed!")

    def on_press(self, event):
        if event.inaxes is None:
            return
            
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
            return
            
        x, y = int(x), int(y)
        print(f"Interaction {self.interaction_count + 1}: Change position: x={x}, y={y}")

        self.visited = set()
        self.count += 1
        self.interaction_count += 1

        # Modify obstacles
        action = ""
        if (x, y) not in self.obs:
            self.obs.add((x, y))
            action = "Added obstacle"
        else:
            self.obs.remove((x, y))
            self.UpdateVertex((x, y))
            action = "Removed obstacle"
            
        print(f"  {action} at ({x}, {y})")

        self.Plot.update_obs(self.obs)

        # Update affected vertices
        for s_n in self.get_neighbor((x, y)):
            self.UpdateVertex(s_n)

        # Recompute shortest path
        self.ComputeShortestPath()

        # Update visualization
        plt.cla()
        self.Plot.plot_grid(f"LPA* - Interaction {self.interaction_count}")
        self.plot_visited(self.visited)
        self.plot_path(self.extract_path())
        
        # Capture frame for GIF
        self.Plot.capture_frame()
        
        # Redraw
        self.fig.canvas.draw_idle()
        
        print(f"  Path updated, frame {len(self.Plot.frames)} captured")

    def ComputeShortestPath(self):
        while True:
            if not self.U:
                break
                
            s, v = self.TopKey()

            if v >= self.CalculateKey(self.s_goal) and \
                    self.rhs[self.s_goal] == self.g[self.s_goal]:
                break

            self.U.pop(s)
            self.visited.add(s)

            if self.g[s] > self.rhs[s]:
                # over-consistent (eg: deleted obstacles)
                self.g[s] = self.rhs[s]
            else:
                # under-consistent (eg: added obstacles)
                self.g[s] = float("inf")
                self.UpdateVertex(s)

            for s_n in self.get_neighbor(s):
                self.UpdateVertex(s_n)

    def UpdateVertex(self, s):
        """update the status and the current cost to come of state s."""
        if s != self.s_start:
            neighbors = self.get_neighbor(s)
            if neighbors:
                self.rhs[s] = min(self.g[s_n] + self.cost(s_n, s)
                                  for s_n in neighbors)
            else:
                self.rhs[s] = float("inf")

        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.CalculateKey(s)

    def TopKey(self):
        """return the min key and its value."""
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def CalculateKey(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(s),
                min(self.g[s], self.rhs[s])]

    def get_neighbor(self, s):
        """find neighbors of state s that not in obstacles."""
        s_list = set()

        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if (0 <= s_next[0] < self.x and 
                0 <= s_next[1] < self.y and 
                s_next not in self.obs):
                s_list.add(s_next)

        return s_list

    def h(self, s):
        """Calculate heuristic."""
        heuristic_type = self.heuristic_type
        goal = self.s_goal

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def cost(self, s_start, s_goal):
        """Calculate Cost for this motion"""
        if self.is_collision(s_start, s_goal):
            return float("inf")

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

    def extract_path(self):
        """Extract the path based on the g values."""
        path = [self.s_goal]
        s = self.s_goal

        for k in range(200):  # Increased iteration limit
            neighbors = self.get_neighbor(s)
            if not neighbors:
                break
                
            g_list = {}
            for x in neighbors:
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            
            if not g_list:
                break
                
            s = min(g_list, key=g_list.get)
            path.append(s)
            
            if s == self.s_start:
                break

        return list(reversed(path))

    def plot_path(self, path):
        """Plot the path"""
        if len(path) > 1:
            self.Plot.plot_path(path)

    def plot_visited(self, visited):
        """Plot visited nodes"""
        self.Plot.count = self.count
        self.Plot.plot_visited(visited)


def main():
    print("LPA* 2D - Self-contained with GIF Generation")
    print("=============================================")
    
    x_start = (5, 5)
    x_goal = (45, 25)

    print(f"Start: {x_start}")
    print(f"Goal: {x_goal}")
    
    lpastar = LPAStar(x_start, x_goal, "euclidean")
    lpastar.run()


if __name__ == '__main__':
    main()
