# Code Review — Professionalism & Reusability

Scope: `Search_2D/` (61 implemented demos, ~33k LOC), `common/`, `benchmarks/`.
Focus requested: reusability of the algorithm code against the environment, and
reusability of the performance-statistics layer. Direction: repo-wide overview,
report + concrete fixes.

---

## TL;DR

The repository already contains the *right* shared modules (`common/env.py`,
`benchmarks/metrics.py`) and legacy wrappers (`Search_2D/env.py`,
`Search_2D/metrics.py`), but the refactor was never propagated into the demos.
As a result:

- **41 / 61** demos still embed their own copy of `class Env`.
- **33 / 61** still embed their own copy of `class Plotting`.
- **45 / 61** re-implement `capture_frame`, **34** re-implement
  `save_animation_as_gif` (~63 lines each → **~1,460 duplicated lines for the GIF
  saver alone**).
- Only **12 / 61** import the shared environment.

The performance-statistics layer (`benchmarks/metrics.py`, 628 lines) is built on
**global monkeypatching** of `builtins.__build_class__` and `matplotlib.pyplot`,
auto-installed via `sitecustomize.py`. It is clever but the opposite of
reusable: it guesses which method is "the algorithm" and which variable is "the
path" from hard-coded name lists, and silently produces no metric when a new
demo doesn't match those names.

Two concrete, verifiable fixes are shipped with this review (see *Fixes
implemented*): a reusable `common/plotting.py` GIF plotter, and an explicit
`measure()` benchmarking API in `benchmarks/metrics.py`. `008_Astar.py` is
converted to both as the reference pattern.

---

## 1. Algorithm ↔ environment reusability

### 1.1 The shared layer exists but is bypassed

`common/env.py` defines the canonical 51×31 grid `Env`. `Search_2D/env.py` is a
thin compatibility wrapper that re-exports it, and `tests/test_refactor_layout.py`
guards the boundary. This is good design — but the demos don't use it.

| Pattern | Count | Note |
|---|---:|---|
| demos total (`NNN_*.py`) | 61 | |
| embed their own `class Env` | 41 | should `from common.env import Env` |
| embed their own `class Plotting` | 33 | no shared plotter existed until now |
| both Env **and** Plotting inline (fully self-contained) | 33 | |
| re-implement `is_collision` | 40 | identical 8-connected corner logic |
| re-implement `obs_map` | 29 | 19 are byte-for-byte identical |
| re-implement `heuristic` | 19 | same manhattan/euclidean switch |
| re-implement `capture_frame` | 45 | identical PIL/BytesIO buffer code |
| re-implement `save_animation_as_gif` | 34 | ~63 lines each, near-identical |
| actually import shared env | 12 | |

The cost of this is not just line count. Every bug fix or behavioural tweak (e.g.
the corner-cutting rule in `is_collision`, or the obstacle layout) has to be made
in dozens of places, and they have already drifted: among the 29 inline
`obs_map` bodies there are **6 distinct variants** of what should be one map.

### 1.2 Two legitimate but unmanaged `Env` conventions

There are genuinely two world models in the repo:

- **Grid** demos: `x_range = 51, y_range = 31`, obstacles as a `set` of cells.
- **Sampling** demos (RRT family): `x_range = (0, 50)`, obstacles as boundary /
  rectangle / circle lists.

Both reuse the name `Env`, so a reader cannot tell which contract a file follows
without reading it. Recommendation: keep both, but name them distinctly and put
them in `common/` — e.g. `common/env.py: GridEnv` and
`common/env_continuous.py: ContinuousEnv` — so the algorithm files declare their
world by import, not by copy-paste.

### 1.3 `is_collision` corner-cutting is duplicated *and* subtly debatable

The 8-connected `is_collision` (copied into 40 files) blocks a diagonal move only
when **both** orthogonal neighbours are obstacles in one diagonal orientation.
This is a reasonable "no corner cutting" rule, but because it is duplicated it
can't be swapped for an alternative policy (e.g. allow cutting, or block if
*either* neighbour is an obstacle) without touching 40 files. This belongs on the
shared `Env`/grid as a single `is_collision(s, s_n)` method.

### 1.4 A* expands nodes more than once

In `Search_2D/008_Astar.py` `searching()` there is no closed-set membership check
on pop — `CLOSED` is an append-only list used for visualisation, and a node can be
re-popped and re-expanded after a cheaper `g` is found. The `g`-comparison keeps
the result correct, but on the 51×31 map it does redundant work and on larger
maps it is an O(E log V) → worse blow-up. A shared `AStar` base would fix this
once. (Lower priority than the reuse issues, but it is a professionalism smell
that recurs across the heuristic-search family because each file is hand-written.)

### 1.5 Recommended target architecture

```
common/
  env.py            # GridEnv (canonical 51x31), is_collision, neighbors, cost
  env_continuous.py # ContinuousEnv for RRT/sampling demos
  plotting.py       # GifPlotter: grid + visited + path + GIF capture  <-- NEW
  search.py         # optional: AStarBase the heuristic demos can subclass
benchmarks/
  metrics.py        # measure()/Result explicit API  <-- NEW, + legacy shim
Search_2D/
  001_*.py ...      # thin demos: import the above, implement only what's novel
```

A demo stays a standalone runnable script — it just imports the shared pieces
instead of copying them. A typical demo should shrink from ~300 lines to ~80.

---

## 2. Performance-statistics reusability

### 2.1 What exists today

`benchmarks/metrics.py` exposes `install_metrics()`, which:

1. replaces `builtins.__build_class__` so that **every class defined under
   `Search_2D/`** is auto-wrapped;
2. inspects each method name against hard-coded sets — `_ALGORITHM_FUNCTIONS`
   (~40 names), `_DEMO_FUNCTIONS`, `_PATH_SINK_FUNCTIONS` — to decide which method
   is "the algorithm" and time it;
3. guesses the resulting path by scanning local-variable names in `_PATH_NAMES`
   (`path`, `best_path`, `fine_path`, …) and object attributes;
4. monkeypatches ~20 `matplotlib.pyplot` functions to subtract plotting time and
   to inject the metric into the plot title;
5. is auto-installed by `Search_2D/sitecustomize.py` on interpreter start, and
   *also* called explicitly at the top of all 61 demos.

### 2.2 Why this is not reusable

- **Name-coupled and silent.** A new algorithm whose method is `expand()` or
  whose result variable is `route` produces **no metric**, with no error. The
  "API" is an ever-growing list of magic strings that the rest of the code must
  conform to.
- **Global side effects.** Patching `builtins.__build_class__` and importing via
  `sitecustomize.py` means *any* Python process started in that directory is
  affected — including pytest and any tool a user runs. This is surprising and
  hard to reason about, and makes the timing non-deterministic (it depends on
  import order and which pyplot calls happen).
- **Not composable.** You cannot ask "time *this* block and give me back a
  number." Everything flows through module-global `_STATE`, so you can't bench
  two algorithms in one process, run repeats, or collect a table without fighting
  the global.
- **Conflates concerns.** Measurement, path-length math, and matplotlib
  title-rendering live in one 628-line module. The path-length helpers
  (`describe_path_length`, `single_path_length`) are genuinely reusable and good;
  they're buried under the monkeypatch machinery.

### 2.3 Recommended direction

Keep the path-length helpers; demote the auto-magic to an *opt-in* convenience,
and add an **explicit, importable** measurement API that returns data:

```python
from benchmarks.metrics import measure

with measure() as m:
    path, visited = astar.searching()
m.record(path, expanded=len(visited))
print(m.line())     # "Path length: 54.042 | Algorithm time: 4.956 ms | expanded: 312"
result = m.result   # Result(path_length=54.042, elapsed_ms=4.956, expanded=312)
```

This is deterministic, composable (works in a loop, returns a `Result` you can
put in a table), decoupled from plotting, and discoverable (it's a function with
a docstring, not a name convention). The legacy `install_metrics()` stays for the
existing demos so nothing breaks; new and refactored demos use `measure()`.

---

## 3. Other professionalism notes (lower priority)

- **Import-before-import bootstrap.** Every demo opens with
  `from metrics import install_metrics; install_metrics()` *above* the stdlib
  imports. It works only because of the `sys.path` hacks; it reads as a code
  smell and breaks the usual "imports are grouped at top" convention. Once
  `measure()` is adopted this line disappears.
- **`Search_based_Planning/.../__pycache__/*.pyc`** is committed. Add to
  `.gitignore` / `git rm --cached`.
- **Vendored env drift.** Because `obs_map` is copied, the 6 variants should be
  reconciled to the single `common` map; otherwise "same benchmark map" claims in
  the README are not strictly true across demos.
- **`extract_path` has no cycle guard** in several files (`while True` on
  `PARENT`); a broken parent chain hangs instead of raising.

---

## Fixes implemented with this review

1. **`common/plotting.py` (new)** — `GifPlotter`, a reusable grid plotter with
   `plot_grid` / `plot_visited` / `plot_path` / `capture_frame` /
   `save_animation_as_gif`, behaviour-compatible with the per-file copies. Removes
   the need for the ~1,460 duplicated GIF-saver lines.
2. **`benchmarks/metrics.py`** — added an explicit, reusable `measure()` context
   manager + `Result` dataclass (deterministic, composable, plotting-independent).
   Legacy `install_metrics()` untouched.
3. **`Search_2D/008_Astar.py`** — converted to the reference pattern: imports
   `common.env.Env` and `common.plotting.GifPlotter`, uses `measure()`, and drops
   ~250 lines of duplicated `Env`/`Plotting`. Remains a standalone runnable script
   and import-clean for CI.

### Suggested rollout (not done here — repo-wide mechanical change)

Convert demos in numeric order, grid family first (they share the exact `Env`):
replace the inline `Env`/`Plotting` with the shared imports, swap the metrics
bootstrap for `measure()`, run `pytest tests/` after each batch. The smoke test
already covers "imports cleanly + defines a class", so regressions surface
immediately.
