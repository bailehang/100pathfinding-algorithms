"""
Anytime_D_star 2D
@author: huiming zhou
@author: clark bai
"""

import io
import os
import sys
import math
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
    """Plotting class for visualization with GIF support"""
    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = (6, 4)

    def update_obs(self, obs):
        self.obs = obs

    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")
        
        # Capture frame for gif
        self.capture_frame()

    def capture_frame(self):
        """Capture current matplotlib figure as a frame for GIF"""
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

    def save_animation_as_gif(self, name, fps=5):
        """Save frames as a GIF animation"""
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


class ADStar:
    def __init__(self, s_start, s_goal, eps, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()  # class Env
        self.Plot = Plotting(s_start, s_goal)

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.g, self.rhs, self.OPEN = {}, {}, {}

        for i in range(1, self.Env.x_range - 1):
            for j in range(1, self.Env.y_range - 1):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.eps = eps
        self.OPEN[self.s_goal] = self.Key(self.s_goal)
        self.CLOSED, self.INCONS = set(), dict()

        self.visited = set()
        self.count = 0
        self.count_env_change = 0
        self.obs_add = set()
        self.obs_remove = set()
        self.title = "Anytime D*: Small changes"  # Significant changes
        self.fig = plt.figure()

    def run(self):
        print("Starting Anytime D* algorithm...")
        self.Plot.plot_grid(self.title)
        self.ComputeOrImprovePath()
        self.plot_visited()
        self.plot_path(self.extract_path())
        self.visited = set()

        iteration = 0
        while True:
            if self.eps <= 1.0:
                break
            iteration += 1
            print(f"Iteration {iteration}: eps = {self.eps}")
            self.eps -= 0.5
            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.plot_visited()
            self.plot_path(self.extract_path())
            self.visited = set()
            plt.pause(0.5)

        # Save GIF after the algorithm completes
        self.Plot.save_animation_as_gif("020_Anytime_D_star")
        print("GIF generation completed!")
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            self.count_env_change += 1
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            # for small changes
            if self.title == "Anytime D*: Small changes":
                if (x, y) not in self.obs:
                    self.obs.add((x, y))
                    self.g[(x, y)] = float("inf")
                    self.rhs[(x, y)] = float("inf")
                else:
                    self.obs.remove((x, y))
                    self.UpdateState((x, y))

                self.Plot.update_obs(self.obs)

                for sn in self.get_neighbor((x, y)):
                    self.UpdateState(sn)

                plt.cla()
                self.Plot.plot_grid(self.title)

                while True:
                    if len(self.INCONS) == 0:
                        break
                    self.OPEN.update(self.INCONS)
                    for s in self.OPEN:
                        self.OPEN[s] = self.Key(s)
                    self.CLOSED = set()
                    self.ComputeOrImprovePath()
                    self.plot_visited()
                    self.plot_path(self.extract_path())
                    # plt.plot(self.title)
                    self.visited = set()

                    if self.eps <= 1.0:
                        break

            else:
                if (x, y) not in self.obs:
                    self.obs.add((x, y))
                    self.obs_add.add((x, y))
                    plt.plot(x, y, 'sk')
                    if (x, y) in self.obs_remove:
                        self.obs_remove.remove((x, y))
                else:
                    self.obs.remove((x, y))
                    self.obs_remove.add((x, y))
                    plt.plot(x, y, marker='s', color='white')
                    if (x, y) in self.obs_add:
                        self.obs_add.remove((x, y))

                self.Plot.update_obs(self.obs)

                if self.count_env_change >= 15:
                    self.count_env_change = 0
                    self.eps += 2.0
                    for s in self.obs_add:
                        self.g[(x, y)] = float("inf")
                        self.rhs[(x, y)] = float("inf")

                        for sn in self.get_neighbor(s):
                            self.UpdateState(sn)

                    for s in self.obs_remove:
                        for sn in self.get_neighbor(s):
                            self.UpdateState(sn)
                        self.UpdateState(s)

                    plt.cla()
                    self.Plot.plot_grid(self.title)

                    while True:
                        if self.eps <= 1.0:
                            break
                        self.eps -= 0.5
                        self.OPEN.update(self.INCONS)
                        for s in self.OPEN:
                            self.OPEN[s] = self.Key(s)
                        self.CLOSED = set()
                        self.ComputeOrImprovePath()
                        self.plot_visited()
                        self.plot_path(self.extract_path())
                        plt.title(self.title)
                        self.visited = set()
                        plt.pause(0.5)

            self.fig.canvas.draw_idle()

    def ComputeOrImprovePath(self):
        while True:
            s, v = self.TopKey()
            if v >= self.Key(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            self.OPEN.pop(s)
            self.visited.add(s)

            if self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                self.CLOSED.add(s)
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
            else:
                self.g[s] = float("inf")
                for sn in self.get_neighbor(s):
                    self.UpdateState(sn)
                self.UpdateState(s)

    def UpdateState(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_neighbor(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.OPEN:
            self.OPEN.pop(s)

        if self.g[s] != self.rhs[s]:
            if s not in self.CLOSED:
                self.OPEN[s] = self.Key(s)
            else:
                self.INCONS[s] = 0

    def Key(self, s):
        if self.g[s] > self.rhs[s]:
            return [self.rhs[s] + self.eps * self.h(self.s_start, s), self.rhs[s]]
        else:
            return [self.g[s] + self.h(self.s_start, s), self.g[s]]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.OPEN, key=self.OPEN.get)
        return s, self.OPEN[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type

        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

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

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if s_next not in self.obs:
                nei_list.add(s_next)

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(100):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        return list(path)

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")
        
        # Capture frame for gif
        self.Plot.capture_frame()

    def plot_visited(self):
        self.count += 1

        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in self.visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])
        
        # Capture frame for gif
        self.Plot.capture_frame()


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    print("Starting Anytime D* algorithm")
    dstar = ADStar(s_start, s_goal, 2.5, "euclidean")
    dstar.run()


if __name__ == '__main__':
    main()
