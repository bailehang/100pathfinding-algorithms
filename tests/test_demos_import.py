"""Smoke test: every demo script must import cleanly.

Importing a demo executes all of its module-level code (the ``Env``/``Plotting``
definitions, the algorithm class, ``install_metrics()``) but *not* the
``if __name__ == "__main__"`` block, so no window opens and no GIF is written.

This is a deliberately lightweight safety net for the "keep flat, light
cleanup" direction: it catches syntax errors, bad imports, broken ``sys.path``
hacks and missing dependencies across all demos without trying to exercise each
algorithm's runtime behaviour.
"""

import importlib.util
import re
from pathlib import Path

import pytest

SEARCH_2D = Path(__file__).resolve().parent.parent / "Search_2D"

# Files named like ``001_bfs.py`` .. ``092_*.py``.
DEMO_FILES = sorted(
    p for p in SEARCH_2D.glob("*.py") if re.match(r"^\d{3}_", p.name)
)


def _import_path(path: Path):
    module_name = f"demo_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_files_discovered():
    assert DEMO_FILES, f"no demo files found under {SEARCH_2D}"


@pytest.mark.parametrize("demo_path", DEMO_FILES, ids=lambda p: p.stem)
def test_demo_imports(demo_path):
    module = _import_path(demo_path)
    # A demo is expected to define at least one class (the algorithm and/or its
    # Env/Plotting helpers). An empty module is almost certainly a mistake.
    has_class = any(isinstance(v, type) for v in vars(module).values())
    assert has_class, f"{demo_path.name} defined no classes after import"
