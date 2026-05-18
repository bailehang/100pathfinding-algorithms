"""
Shared pathfinding demo metrics.

The helpers in this file intentionally accept several path shapes used across
the demos: coordinate tuples, Node objects with x/y attributes, dictionaries of
multi-agent paths, and lists of paths.
"""

import math
import os
import sys
import time
import atexit
import builtins
import functools


def now_ms():
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms):
    return now_ms() - start_ms


def point_xy(point):
    if point is None:
        return None

    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)

    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        return None

    if isinstance(point, (tuple, list)) and len(point) >= 2:
        x, y = point[0], point[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)

    return None


def single_path_length(path):
    if path is None:
        return None

    points = []
    try:
        iterator = iter(path)
    except TypeError:
        return None

    for item in iterator:
        xy = point_xy(item)
        if xy is None:
            return None
        points.append(xy)

    if not points:
        return None

    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def path_lengths(path_like):
    if path_like is None:
        return []

    direct = single_path_length(path_like)
    if direct is not None:
        return [direct]

    if isinstance(path_like, dict):
        lengths = []
        for value in path_like.values():
            lengths.extend(path_lengths(value))
        return lengths

    if isinstance(path_like, (tuple, list)):
        lengths = []
        for item in path_like:
            lengths.extend(path_lengths(item))
        return lengths

    return []


def describe_path_length(path_like):
    lengths = path_lengths(path_like)
    if not lengths:
        return "N/A"

    if len(lengths) == 1:
        return f"{lengths[0]:.3f}"

    total = sum(lengths)
    average = total / len(lengths)
    return f"total {total:.3f}, avg {average:.3f}, n={len(lengths)}"


def metrics_line(path_like, elapsed):
    return f"路径长度: {describe_path_length(path_like)} | 耗时: {elapsed:.3f} ms"


def latest_metrics_line():
    if _STATE["path"] is None or _STATE["elapsed_ms"] is None:
        return ""
    return metrics_line(_STATE["path"], _STATE["elapsed_ms"])


_INSTALLED = False
_STATE = {
    "path": None,
    "elapsed_ms": None,
    "source": None,
    "printed": False,
    "pending_elapsed_ms": 0.0,
}
_START_MS = None
_PATCHED_PYPLOT = False
_ORIGINAL_BUILD_CLASS = None

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PATH_NAMES = (
    "path",
    "paths",
    "fine_path",
    "coarse_path",
    "goal_paths",
    "optimized_path",
    "all_paths",
    "solution",
    "solutions",
    "result_path",
    "final_path",
)
_DEMO_FUNCTIONS = {
    "animation",
    "animation_lrta",
    "animation_ara_star",
    "animation_bi_astar",
    "animation_hierarchical_astar",
    "animation_parallel_astar",
    "animation_vehicle",
    "animation_connect",
    "animation_gcs",
    "plot_path",
    "plot_grid",
    "plot_visited",
    "plot_visited_bi",
    "save_animation_as_gif",
    "capture_frame",
}
_ALGORITHM_FUNCTIONS = {
    "searching",
    "searching_online",
    "searching_bidirectional",
    "search",
    "search_global",
    "region_a_star",
    "planning",
    "plan",
    "plan_abstract",
    "plan_concrete",
    "solve",
    "solve_low_level",
    "AStar",
    "a_star_search",
    "search_on_voronoi",
    "ComputeShortestPath",
    "ComputePath",
    "ComputeOrImprovePath",
}
_PATH_SINK_FUNCTIONS = {
    "extract_path",
    "ExtractPath",
    "plot_path",
    "animation",
    "animation_lrta",
    "animation_ara_star",
    "animation_bi_astar",
    "animation_hierarchical_astar",
    "animation_parallel_astar",
    "animation_vehicle",
    "animation_connect",
    "animation_gcs",
}


def _has_path(path_like):
    return describe_path_length(path_like) != "N/A"


def _record_path(path_like, source, elapsed):
    if not _has_path(path_like):
        return

    _STATE["path"] = path_like
    _STATE["elapsed_ms"] = elapsed
    _STATE["source"] = source
    _STATE["pending_elapsed_ms"] = 0.0
    _patch_matplotlib_title()
    _refresh_current_matplotlib_title()


def _record_pending_elapsed(elapsed):
    _STATE["pending_elapsed_ms"] += elapsed


def _candidate_from_return(value):
    if isinstance(value, tuple):
        for item in value:
            if _has_path(item):
                return item

    if _has_path(value):
        return value

    return None


def _candidate_from_locals(locals_dict):
    for name in _PATH_NAMES:
        if name in locals_dict and _has_path(locals_dict[name]):
            return locals_dict[name]

    for name in ("planner", "rrt", "rrt_conn", "gcs", "theta_star", "voronoi"):
        obj = locals_dict.get(name)
        if obj is None:
            continue
        for attr in _PATH_NAMES:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if _has_path(value):
                    return value

    return None


def _patch_matplotlib_title():
    global _PATCHED_PYPLOT

    if _PATCHED_PYPLOT:
        return

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None or not hasattr(plt, "title"):
        return

    original_title = plt.title

    def title_with_metrics(label, *args, **kwargs):
        if _STATE["path"] is not None and isinstance(label, str) and "路径长度:" not in label:
            label = f"{label}\n{metrics_line(_STATE['path'], _STATE['elapsed_ms'])}"
        return original_title(label, *args, **kwargs)

    plt.title = title_with_metrics
    _PATCHED_PYPLOT = True


def _refresh_current_matplotlib_title():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None or not hasattr(plt, "gca") or not hasattr(plt, "title"):
        return

    try:
        current_title = plt.gca().get_title()
    except Exception:
        return

    if current_title and "路径长度:" not in current_title:
        plt.title(current_title)


def _candidate_from_object(obj):
    if obj is None:
        return None

    for attr in _PATH_NAMES:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if _has_path(value):
                return value

    return None


def _wrap_algorithm_callable(func, source_name):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = now_ms()
        result = func(*args, **kwargs)
        elapsed = now_ms() - start

        candidate = _candidate_from_return(result)
        if candidate is None and args:
            candidate = _candidate_from_object(args[0])

        if candidate is not None:
            _record_path(candidate, source_name, elapsed)
        else:
            _record_pending_elapsed(elapsed)

        return result

    return wrapped


def _wrap_path_sink_callable(func, source_name):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        candidate = None
        for value in list(args) + list(kwargs.values()):
            if _has_path(value):
                candidate = value
                break

        result = func(*args, **kwargs)

        if candidate is None:
            candidate = _candidate_from_return(result)
        if candidate is not None and _STATE["pending_elapsed_ms"] > 0:
            _record_path(candidate, source_name, _STATE["pending_elapsed_ms"])

        return result

    return wrapped


def _wrap_class_algorithms(cls):
    for name, value in list(cls.__dict__.items()):
        if name in _ALGORITHM_FUNCTIONS:
            wrapper_factory = _wrap_algorithm_callable
        elif name in _PATH_SINK_FUNCTIONS or name.startswith("animation"):
            wrapper_factory = _wrap_path_sink_callable
        else:
            continue

        if isinstance(value, staticmethod):
            wrapped = staticmethod(wrapper_factory(value.__func__, f"{cls.__name__}.{name}"))
        elif isinstance(value, classmethod):
            wrapped = classmethod(wrapper_factory(value.__func__, f"{cls.__name__}.{name}"))
        elif callable(value):
            wrapped = wrapper_factory(value, f"{cls.__name__}.{name}")
        else:
            continue

        setattr(cls, name, wrapped)

    return cls


def _build_class_wrapper(func, name, *bases, **kwargs):
    cls = _ORIGINAL_BUILD_CLASS(func, name, *bases, **kwargs)

    module_name = getattr(cls, "__module__", "")
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", "")
    if module_file and os.path.abspath(module_file).startswith(_ROOT):
        _wrap_class_algorithms(cls)

    return cls


def print_metrics():
    if _STATE["printed"] or _START_MS is None:
        return

    elapsed = _STATE["elapsed_ms"]
    if elapsed is None:
        elapsed = elapsed_ms(_START_MS)

    print(f"[Metrics] {metrics_line(_STATE['path'], elapsed)}")
    _STATE["printed"] = True


def install_metrics():
    """Install automatic console and matplotlib title metrics for a demo script."""

    global _INSTALLED, _START_MS, _ORIGINAL_BUILD_CLASS

    if _INSTALLED:
        return

    _INSTALLED = True
    _START_MS = now_ms()
    _ORIGINAL_BUILD_CLASS = builtins.__build_class__
    builtins.__build_class__ = _build_class_wrapper
    atexit.register(print_metrics)
