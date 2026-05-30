"""Shared pytest setup for the pathfinding demos.

The demos are designed to be run as standalone scripts. To import them inside a
test process we need to:

* force a non-interactive matplotlib backend so nothing tries to open a window;
* expose ``Search_2D/`` on ``sys.path`` so ``from metrics import ...`` resolves;
* expose the repo root so the few demos that do ``from Search_2D import ...``
  resolve as well.
"""

import os
import sys

# Headless backend must be selected before any demo imports matplotlib.pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEARCH_2D = os.path.join(_REPO_ROOT, "Search_2D")

for _path in (_REPO_ROOT, _SEARCH_2D):
    if _path not in sys.path:
        sys.path.insert(0, _path)
