"""
Plotting tools for creating GIF animations of pathfinding algorithms
@author: Modified from original code by clark bai
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Plotting:
    """Plotting class for visualization with GIF support"""
    
    def __init__(self, xI, xG):
        """
        Initialize the plotting class
        :param xI: start point [x, y]
        :param xG: goal point [x, y]
        """
        self.xI, self.xG = xI, xG
        self.frames = []
        
        # Import Env class dynamically to avoid circular imports
        from Search_2D.env import Env
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
        """Capture the current figure as a frame with correct color handling"""
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Get the RGBA buffer from the canvas
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        
        # Convert to numpy array
        data = np.frombuffer(buf, dtype=np.uint8)
        
        # Calculate the correct dimensions based on the data size
        # Each pixel has 4 channels (ARGB), so total size = w * h * 4
        # Therefore, each color channel has w * h elements
        total_pixels = len(data) // 4
        
        # Calculate width and height that will work with the data size
        # We can use the aspect ratio from get_width_height() but ensure total pixels match
        aspect_ratio = w / h
        calculated_h = int(np.sqrt(total_pixels / aspect_ratio))
        calculated_w = int(total_pixels / calculated_h)
        
        # Extract color channels
        r = data[1::4]  # Red channel
        g = data[2::4]  # Green channel
        b = data[3::4]  # Blue channel
        
        # Reshape using calculated dimensions
        image = np.stack([r, g, b], axis=-1).reshape((calculated_h, calculated_w, 3))

        # Add to frames
        self.frames.append(image)

    def save_animation_as_gif(self, name, fps=15):
        """Save frames as a GIF animation with proper color handling"""
        # Create directory for GIFs
        gif_dir = "gif"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{name}.gif")

        print(f"Saving GIF animation to {gif_path}...")

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
