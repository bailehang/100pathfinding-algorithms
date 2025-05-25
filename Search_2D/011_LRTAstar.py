"""
LRTA_star 2D (Learning Real-time A*)
@author: huiming zhou
"""

import math
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import heapq
import copy


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


class QueuePrior:
    """Priority Queue implementation for LRTA*"""
    def __init__(self):
        self.queue = []
        self.count = 0
    
    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, self.count, item))
        self.count += 1
    
    def get(self):
        return heapq.heappop(self.queue)[2]
    
    def empty(self):
        return len(self.queue) == 0


class Plotting:
    """Plotting class for visualization"""

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

    def animation_lrta(self, path, visited, name, save_gif=False):
        """Animation for LRTA* algorithm with multiple iterations"""
        self.plot_grid(name)
        
        for i in range(len(path)):
            # Plot visited nodes for each iteration
            if i < len(visited):
                self.plot_visited(visited[i], cl=plt.cm.viridis(i / len(path)))
            
            # Plot path for each iteration
            self.plot_path(path[i], cl=plt.cm.cool(i / len(path)), flag=True)
            plt.pause(0.5)
        
        plt.pause(1)
        
        if save_gif:
            self.save_animation_as_gif(name)
        
        plt.show()

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
        if visited:
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
        if path:
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
        """Capture the current figure as a frame with consistent size using a simpler approach"""
        # Save figure to a temporary file and read it back with PIL
        import io
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


class LrtAStarN:
    """Learning Real-time A* (LRTA*) algorithm implementation"""
    def __init__(self, s_start, s_goal, N, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env()
        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.N = N  # number of expand nodes each iteration
        self.visited = []  # order of visited nodes in planning
        self.path = []  # path of each iteration
        self.h_table = {}  # h_value table

    def init(self):
        """
        initialize the h_value of all nodes in the environment.
        it is a global table.
        """
        for i in range(self.Env.x_range):
            for j in range(self.Env.y_range):
                self.h_table[(i, j)] = self.h((i, j))

    def searching(self):
        """Main search function for LRTA*"""
        self.init()
        s_start = self.s_start  # initialize start node

        while True:
            OPEN, CLOSED = self.AStar(s_start, self.N)  # OPEN, CLOSED sets in each iteration

            if OPEN == "FOUND":  # reach the goal node
                self.path.append(CLOSED)
                break

            h_value = self.iteration(CLOSED)  # h_value table of CLOSED nodes

            for x in h_value:
                self.h_table[x] = h_value[x]

            s_start, path_k = self.extract_path_in_CLOSE(s_start, h_value)  # x_init -> expected node in OPEN set
            self.path.append(path_k)

    def extract_path_in_CLOSE(self, s_start, h_value):
        """Extract path from the CLOSED set to the next starting node"""
        path = [s_start]
        s = s_start

        while True:
            h_list = {}

            for s_n in self.get_neighbor(s):
                if s_n in h_value:
                    h_list[s_n] = h_value[s_n]
                else:
                    h_list[s_n] = self.h_table[s_n]

            s_key = min(h_list, key=h_list.get)  # move to the smallest node with min h_value
            path.append(s_key)  # generate path
            s = s_key  # use end of this iteration as the start of next

            if s_key not in h_value:  # reach the expected node in OPEN set
                return s_key, path

    def iteration(self, CLOSED):
        """Update h-values through iteration until convergence"""
        h_value = {}

        for s in CLOSED:
            h_value[s] = float("inf")  # initialize h_value of CLOSED nodes

        while True:
            h_value_rec = copy.deepcopy(h_value)
            for s in CLOSED:
                h_list = []
                for s_n in self.get_neighbor(s):
                    if s_n not in CLOSED:
                        h_list.append(self.cost(s, s_n) + self.h_table[s_n])
                    else:
                        h_list.append(self.cost(s, s_n) + h_value[s_n])
                h_value[s] = min(h_list)  # update h_value of current node

            if h_value == h_value_rec:  # h_value table converged
                return h_value

    def AStar(self, x_start, N):
        """A* search for a limited number of expansions"""
        OPEN = QueuePrior()  # OPEN set
        OPEN.put(x_start, self.h(x_start))
        CLOSED = []  # CLOSED set
        g_table = {x_start: 0, self.s_goal: float("inf")}  # Cost to come
        PARENT = {x_start: x_start}  # relations
        count = 0  # counter

        while not OPEN.empty():
            count += 1
            s = OPEN.get()
            CLOSED.append(s)

            if s == self.s_goal:  # reach the goal node
                self.visited.append(CLOSED)
                return "FOUND", self.extract_path(x_start, PARENT)

            for s_n in self.get_neighbor(s):
                if s_n not in CLOSED:
                    new_cost = g_table[s] + self.cost(s, s_n)
                    if s_n not in g_table:
                        g_table[s_n] = float("inf")
                    if new_cost < g_table[s_n]:  # conditions for updating Cost
                        g_table[s_n] = new_cost
                        PARENT[s_n] = s
                        OPEN.put(s_n, g_table[s_n] + self.h_table[s_n])

            if count == N:  # expand needed CLOSED nodes
                break

        self.visited.append(CLOSED)  # visited nodes in each iteration

        return OPEN, CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        s_list = []

        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if s_next not in self.obs:
                s_list.append(s_next)

        return s_list

    def extract_path(self, x_start, parent):
        """
        Extract the path based on the relationship of nodes.
        :return: The planning path
        """
        path_back = [self.s_goal]
        x_current = self.s_goal

        while True:
            x_current = parent[x_current]
            path_back.append(x_current)

            if x_current == x_start:
                break

        return list(reversed(path_back))

    def h(self, s):
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
        """
        Check if the path between two nodes collides with obstacles
        :param s_start: start node
        :param s_end: end node
        :return: True: collision, False: no collision
        """
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


def main():
    s_start = (10, 5)
    s_goal = (45, 25)

    print("Starting LRTA* algorithm")
    lrta = LrtAStarN(s_start, s_goal, 250, "euclidean")
    plot = Plotting(s_start, s_goal)

    print("Searching for path...")
    lrta.searching()
    print(f"Path found with {len(lrta.path)} iterations")
    
    print("Starting animation and GIF creation...")
    plot.animation_lrta(lrta.path, lrta.visited, "011_LRTAstar", save_gif=True)  # Save animation as GIF
    print("Animation and GIF creation completed")


if __name__ == '__main__':
    main()
