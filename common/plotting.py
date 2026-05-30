"""Reusable 2D grid plotter with GIF capture.

This consolidates the ``Plotting`` class that was copy-pasted into ~33 demos
(and the ``capture_frame`` / ``save_animation_as_gif`` helpers copied into 45 /
34 demos respectively). Behaviour matches the per-file copies so demos can drop
their inline plotter and ``from common.plotting import GifPlotter`` instead.

A demo stays a standalone runnable script; it just imports the plotter rather
than re-declaring it.
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common.env import Env


class GifPlotter:
    """Grid visualiser that records frames and can export them as a GIF.

    Parameters
    ----------
    xI, xG : tuple[int, int]
        Start and goal cells.
    env : Env, optional
        Environment providing ``obs`` / ``obs_map()``. Defaults to the shared
        :class:`common.env.Env`.
    fig_size : tuple[int, int], optional
        Matplotlib figure size in inches.
    gif_dir : str, optional
        Directory GIFs are written to (created on demand). Defaults to the
        repository's ``Search_2D/gif`` directory.
    """

    def __init__(self, xI, xG, env=None, fig_size=(6, 4), gif_dir=None):
        self.xI, self.xG = xI, xG
        self.env = env or Env()
        self.obs = self.env.obs_map()
        self.frames = []
        self.fig_size = fig_size
        self.gif_dir = gif_dir or self.default_gif_dir()

    @staticmethod
    def default_gif_dir():
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo_root, "Search_2D", "gif")

    def update_obs(self, obs):
        self.obs = obs

    # -- high level ---------------------------------------------------------
    def animation(self, path, visited, name, save_gif=False):
        """Render grid + visited order + final path, then optionally save a GIF."""
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        if save_gif:
            self.save_animation_as_gif(name)
        else:
            plt.show()

    # -- primitives ---------------------------------------------------------
    def plot_grid(self, name):
        plt.figure(figsize=self.fig_size, dpi=100, clear=True)

        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

        self.capture_frame()

    def plot_visited(self, visited, cl="gray"):
        visited = list(visited)
        if self.xI in visited:
            visited.remove(self.xI)
        if self.xG in visited:
            visited.remove(self.xG)

        count = 0
        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker="o")
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                self.pause(0.01)
                self.capture_frame()

        self.pause(0.1)
        self.capture_frame()

    def plot_path(self, path, cl="r", flag=False):
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]

        color = "r" if not flag else cl
        plt.plot(path_x, path_y, linewidth="3", color=color)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        self.pause(0.1)
        self.capture_frame()

    @staticmethod
    def pause(interval):
        if "agg" not in matplotlib.get_backend().lower():
            plt.pause(interval)

    # -- frame capture / export --------------------------------------------
    def capture_frame(self):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_rgb = Image.open(buf).convert("RGB")
        self.frames.append(np.array(img_rgb))
        buf.close()

    def save_animation_as_gif(self, name, fps=15):
        os.makedirs(self.gif_dir, exist_ok=True)
        gif_path = os.path.join(self.gif_dir, f"{name}.gif")

        if not self.frames:
            print("No frames to save!")
            plt.close()
            return gif_path

        # Normalise frame sizes (bbox_inches='tight' can vary by a pixel).
        first_shape = self.frames[0].shape
        for i, frame in enumerate(self.frames):
            if frame.shape != first_shape:
                self.frames[i] = np.array(
                    Image.fromarray(frame).resize(
                        (first_shape[1], first_shape[0]), Image.LANCZOS
                    )
                )

        frames_p = [
            Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=256)
            for f in self.frames
        ]
        frames_p[0].save(
            gif_path,
            format="GIF",
            append_images=frames_p[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        plt.close()
        return gif_path


__all__ = ["GifPlotter"]
