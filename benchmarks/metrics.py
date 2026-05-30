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


def _numeric_sequence(value):
    if not isinstance(value, (tuple, list)):
        return None

    numbers = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        numbers.append(float(item))

    return numbers


def _split_coordinate_points(path_like):
    if not isinstance(path_like, (tuple, list)) or len(path_like) != 2:
        return None

    if isinstance(path_like, list) and point_xy(path_like[0]) is not None and point_xy(path_like[1]) is not None:
        return None

    xs = _numeric_sequence(path_like[0])
    ys = _numeric_sequence(path_like[1])
    if xs is None or ys is None or len(xs) != len(ys) or not xs:
        return None

    return list(zip(xs, ys))


def _length_from_points(points):
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def single_path_length(path):
    if path is None:
        return None

    split_points = _split_coordinate_points(path)
    if split_points is not None:
        return _length_from_points(split_points)

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

    return _length_from_points(points)


def path_lengths(path_like):
    if path_like is None:
        return []

    if isinstance(path_like, dict):
        lengths = []
        for value in path_like.values():
            if isinstance(value, (tuple, list)) and value and single_path_length(value[0]) is not None:
                lengths.extend(path_lengths(value[0]))
            else:
                lengths.extend(path_lengths(value))
        return lengths

    direct = single_path_length(path_like)
    if direct is not None:
        return [direct]

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
    return f"Path length: {describe_path_length(path_like)} | Algorithm time: {elapsed:.3f} ms"


def latest_metrics_line():
    if _STATE["path"] is None or _STATE["elapsed_ms"] is None:
        return ""
    return metrics_line(_STATE["path"], _STATE["elapsed_ms"])


def print_latest_metrics():
    if _STATE["printed"] or _START_MS is None:
        return

    elapsed = _STATE["elapsed_ms"]
    if elapsed is None:
        elapsed = elapsed_ms(_START_MS)

    print(f"[Metrics] {metrics_line(_STATE['path'], elapsed)}")
    _STATE["printed"] = True


def print_metrics_for(path_like, elapsed, source="explicit"):
    _STATE["path"] = path_like
    _STATE["elapsed_ms"] = elapsed
    _STATE["source"] = source
    _STATE["pending_elapsed_ms"] = 0.0
    if _has_path(path_like):
        _patch_matplotlib_title()
        _refresh_current_matplotlib_title()
    print_latest_metrics()


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
_PATCHED_PYPLOT_CALLS = False
_ORIGINAL_BUILD_CLASS = None
_CALL_STACK = []
_DEMO_DEPTH = 0

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Search_2D"))
_PATH_NAMES = (
    "path",
    "paths",
    "best_path",
    "fine_path",
    "coarse_path",
    "goal_paths",
    "optimized_path",
    "all_paths",
    "solution",
    "solutions",
    "result_path",
    "final_path",
    "current_path",
    "smoothed_path",
    "raw_path",
    "concrete_path",
    "abstract_path",
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
    "run",
    "searching",
    "searching_repeated_astar",
    "searching_online",
    "searching_bidirectional",
    "repeated_searching",
    "search",
    "search_path",
    "search_global",
    "high_level_search",
    "low_level_search",
    "jps_local_search",
    "low_level_refine",
    "region_a_star",
    "planning",
    "replanning",
    "plan",
    "plan_paths",
    "plan_abstract",
    "plan_concrete",
    "solve",
    "solve_low_level",
    "AStar",
    "a_star_search",
    "search_on_voronoi",
    "compute_path",
    "ComputeShortestPath",
    "ComputePath",
    "ComputeOrImprovePath",
    "run_demonstration",
    "run_obstacle_removal_demonstration",
    "run_complete_demonstration",
}
_PATH_SINK_FUNCTIONS = {
    "extract_path",
    "extract_temp_path",
    "extract_field_path",
    "extract_partial_path",
    "ExtractPath",
    "reconstruct_path",
    "optimize_path",
    "post_process_path",
    "path_smoothing",
    "generate_final_course",
    "plot_path",
    "plot_paths",
    "animation",
    "animation_lrta",
    "animation_ara_star",
    "animation_bi_astar",
    "animation_hierarchical_astar",
    "animation_parallel_astar",
    "animation_vehicle",
    "animation_connect",
    "animation_gcs",
    "animate_solution",
    "animate_paths",
}
_PYPLOT_DEMO_FUNCTIONS = (
    "figure",
    "subplots",
    "subplot",
    "cla",
    "clf",
    "plot",
    "scatter",
    "fill",
    "fill_between",
    "imshow",
    "text",
    "title",
    "axis",
    "legend",
    "pause",
    "show",
    "draw",
    "savefig",
)


def _has_path(path_like):
    return describe_path_length(path_like) != "N/A"


def _current_algorithm_frame():
    for frame in reversed(_CALL_STACK):
        if frame["kind"] == "algorithm":
            return frame
    return None


def _remember_frame_path(path_like):
    if not _has_path(path_like):
        return

    frame = _current_algorithm_frame()
    if frame is not None:
        frame["path"] = path_like


def _add_demo_elapsed(elapsed):
    frame = _current_algorithm_frame()
    if frame is not None:
        frame["demo_ms"] += elapsed


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


def _patch_matplotlib_demo_calls():
    global _PATCHED_PYPLOT_CALLS

    if _PATCHED_PYPLOT_CALLS:
        return

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return

    for name in _PYPLOT_DEMO_FUNCTIONS:
        original = getattr(plt, name, None)
        if not callable(original) or getattr(original, "_metrics_wrapped", False):
            continue

        @functools.wraps(original)
        def wrapped(*args, __func=original, **kwargs):
            global _DEMO_DEPTH

            outer_demo = _DEMO_DEPTH == 0
            _DEMO_DEPTH += 1
            start = now_ms()
            try:
                return __func(*args, **kwargs)
            finally:
                elapsed = now_ms() - start
                _DEMO_DEPTH -= 1
                if outer_demo:
                    _add_demo_elapsed(elapsed)

        wrapped._metrics_wrapped = True
        setattr(plt, name, wrapped)

    _PATCHED_PYPLOT_CALLS = True


def _patch_matplotlib_title():
    global _PATCHED_PYPLOT

    if _PATCHED_PYPLOT:
        return

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None or not hasattr(plt, "title"):
        return

    original_title = plt.title

    def title_with_metrics(label, *args, **kwargs):
        if _STATE["path"] is not None and isinstance(label, str) and "Path length:" not in label:
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

    if current_title and "Path length:" not in current_title:
        plt.title(current_title)


def _candidate_from_object(obj):
    if obj is None:
        return None

    if hasattr(obj, "path_x") and hasattr(obj, "path_y"):
        candidate = (getattr(obj, "path_x"), getattr(obj, "path_y"))
        if _has_path(candidate):
            return candidate

    for attr in _PATH_NAMES:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if _has_path(value):
                return value

    if hasattr(obj, "agents"):
        paths = []
        for agent in getattr(obj, "agents"):
            for attr in ("concrete_path", "path", "final_path"):
                if hasattr(agent, attr):
                    value = getattr(agent, attr)
                    if _has_path(value):
                        paths.append(value)
                        break
        if _has_path(paths):
            return paths

    return None


def _wrap_algorithm_callable(func, source_name):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _patch_matplotlib_demo_calls()

        frame = {
            "kind": "algorithm",
            "demo_ms": 0.0,
            "path": None,
        }
        _CALL_STACK.append(frame)
        start = now_ms()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = now_ms() - start
            _CALL_STACK.pop()

        elapsed = max(0.0, elapsed - frame["demo_ms"])

        candidate = _candidate_from_return(result)
        if candidate is None and args:
            candidate = _candidate_from_object(args[0])
        if candidate is None:
            candidate = frame["path"]

        if candidate is not None:
            _record_path(candidate, source_name, elapsed)
        else:
            _record_pending_elapsed(elapsed)

        return result

    return wrapped


def _candidate_from_arguments(args, kwargs):
    for value in list(args) + list(kwargs.values()):
        if _has_path(value):
            return value

    if args:
        return _candidate_from_object(args[0])

    return None


def _wrap_demo_callable(func, source_name, capture_path=False):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        global _DEMO_DEPTH

        _patch_matplotlib_demo_calls()

        candidate = _candidate_from_arguments(args, kwargs) if capture_path else None
        outer_demo = _DEMO_DEPTH == 0
        _DEMO_DEPTH += 1
        start = now_ms()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = now_ms() - start
            _DEMO_DEPTH -= 1
            if outer_demo:
                _add_demo_elapsed(elapsed)

        if not capture_path:
            return result

        if candidate is None:
            candidate = _candidate_from_return(result)
        if candidate is None and args:
            candidate = _candidate_from_object(args[0])
        if candidate is not None:
            _remember_frame_path(candidate)
            if _current_algorithm_frame() is None and _STATE["pending_elapsed_ms"] > 0:
                _record_path(candidate, source_name, _STATE["pending_elapsed_ms"])

        return result

    return wrapped


def _wrap_class_algorithms(cls):
    for name, value in list(cls.__dict__.items()):
        if name in _ALGORITHM_FUNCTIONS:
            wrapper_factory = _wrap_algorithm_callable
        elif name in _PATH_SINK_FUNCTIONS or name.startswith("animation"):
            wrapper_factory = functools.partial(_wrap_demo_callable, capture_path=True)
        elif (
            name in _DEMO_FUNCTIONS
            or name.startswith("plot_")
            or name.startswith("draw_")
            or name.startswith("animate_")
        ):
            wrapper_factory = _wrap_demo_callable
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
    _patch_matplotlib_demo_calls()

    module_name = getattr(cls, "__module__", "")
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", "")
    if module_file and os.path.abspath(module_file).startswith(_ROOT):
        _wrap_class_algorithms(cls)

    return cls


def print_metrics():
    print_latest_metrics()


def install_metrics():
    """Install automatic console and matplotlib title metrics for a demo script."""

    global _INSTALLED, _START_MS, _ORIGINAL_BUILD_CLASS

    if _INSTALLED:
        return

    _INSTALLED = True
    _START_MS = now_ms()
    _ORIGINAL_BUILD_CLASS = builtins.__build_class__
    builtins.__build_class__ = _build_class_wrapper
    _patch_matplotlib_demo_calls()
    atexit.register(print_metrics)


# ---------------------------------------------------------------------------
# Explicit, reusable benchmarking API
#
# install_metrics() above auto-discovers timings by monkeypatching
# builtins.__build_class__ and matplotlib. That is convenient but name-coupled
# and silent (a new algorithm whose method/variable names are not in the
# hard-coded sets produces no metric).
#
# measure() is the recommended path for new and refactored demos: explicit,
# deterministic, composable (works in a loop, returns data), and decoupled from
# plotting. It reuses the same path-length helpers.
# ---------------------------------------------------------------------------

import dataclasses


@dataclasses.dataclass
class Result:
    """A single benchmark measurement."""

    elapsed_ms: float
    path_length: float = None
    extra: dict = dataclasses.field(default_factory=dict)

    def line(self):
        parts = []
        if self.path_length is not None:
            parts.append(f"Path length: {self.path_length:.3f}")
        parts.append(f"Algorithm time: {self.elapsed_ms:.3f} ms")
        for key, value in self.extra.items():
            parts.append(f"{key}: {value}")
        return " | ".join(parts)


class Measurement:
    """Mutable handle yielded by measure(); times the ``with`` block."""

    def __init__(self):
        self._start = None
        self.result = Result(elapsed_ms=0.0)

    def __enter__(self):
        self._start = now_ms()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.result.elapsed_ms = elapsed_ms(self._start)
        return False

    def record(self, path=None, **extra):
        """Attach a path (length computed) and/or named extra metrics."""
        if path is not None:
            lengths = path_lengths(path)
            self.result.path_length = sum(lengths) if lengths else None
        self.result.extra.update(extra)
        return self.result

    def line(self):
        return self.result.line()


def measure():
    """Return a Measurement context manager.

    >>> with measure() as m:
    ...     path, visited = astar.searching()
    >>> m.record(path, expanded=len(visited))
    >>> print(m.line())
    Path length: 54.042 | Algorithm time: 4.956 ms | expanded: 312
    """
    return Measurement()
