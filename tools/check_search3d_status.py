"""Check Search_3D algorithm status documentation consistency.

This verifies the 37 3D aerial algorithm entries across:

* ``Search_3D/README.md`` status table;
* ``Search_3D/src/main.js`` algorithm detail metadata;
* the root ``README.md`` 3D headline and implemented / approximate counts.

The root README reports exact ``已实现`` and ``近似实现`` counts for the 3D aerial algorithms.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_README = REPO_ROOT / "README.md"
SEARCH3D_README = REPO_ROOT / "Search_3D" / "README.md"
SEARCH3D_MAIN = REPO_ROOT / "Search_3D" / "src" / "main.js"

EXPECTED_IDS = [
    *(f"A{i:02d}" for i in range(1, 9)),
    *(f"B{i:02d}" for i in range(1, 7)),
    *(f"C{i:02d}" for i in range(1, 9)),
    *(f"D{i:02d}" for i in range(1, 6)),
    *(f"E{i:02d}" for i in range(1, 6)),
    *(f"F{i:02d}" for i in range(1, 6)),
]
EXPECTED_SET = set(EXPECTED_IDS)
IMPLEMENTED_STATUSES = {"已实现", "近似实现", "部分实现", "参数展示"}

README_ROW_RE = re.compile(r"^\|\s*([A-F]\d{2})\s*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|", re.M)
JS_STATUS_RE = re.compile(r"^\s{2}([A-F]\d{2}):\s*\{\r?\n\s*status:\s*\"([^\"]+)\"", re.M)
ROOT_PROGRESS_RE = re.compile(
    r"当前进度：37 个 3D 空中算法条目已建档，其中\s*(\d+)\s*个已实现、\s*(\d+)\s*个近似实现"
)
ROOT_IMPLEMENTED_LINE_RE = re.compile(r"3D IMPLEMENTED\s+\[[#\-]+\]\s+\d+%\s+(\d+)\s*/\s*37")
ROOT_APPROXIMATE_LINE_RE = re.compile(r"3D APPROXIMATE\s+\[[#\-]+\]\s+\d+%\s+(\d+)\s*/\s*37")
ROOT_TABLE_RE = re.compile(r"\|\s*(已实现|近似实现)\s*\|\s*(\d+)\s*\|")


def parse_search3d_readme() -> dict[str, str]:
    text = SEARCH3D_README.read_text(encoding="utf-8")
    return {match.group(1): match.group(2).strip() for match in README_ROW_RE.finditer(text)}


def parse_main_js() -> dict[str, str]:
    text = SEARCH3D_MAIN.read_text(encoding="utf-8")
    return {match.group(1): match.group(2).strip() for match in JS_STATUS_RE.finditer(text)}


def parse_root_counts() -> dict[str, int | None]:
    text = ROOT_README.read_text(encoding="utf-8")
    counts: dict[str, int | None] = {
        "headline_done": None,
        "headline_approx": None,
        "bar_done": None,
        "bar_approx": None,
        "table_done": None,
        "table_approx": None,
    }
    if match := ROOT_PROGRESS_RE.search(text):
        counts["headline_done"] = int(match.group(1))
        counts["headline_approx"] = int(match.group(2))
    if match := ROOT_IMPLEMENTED_LINE_RE.search(text):
        counts["bar_done"] = int(match.group(1))
    if match := ROOT_APPROXIMATE_LINE_RE.search(text):
        counts["bar_approx"] = int(match.group(1))
    for label, value in ROOT_TABLE_RE.findall(text):
        if label == "已实现":
            counts["table_done"] = int(value)
        if label == "近似实现":
            counts["table_approx"] = int(value)
    return counts


def main() -> int:
    problems: list[str] = []
    readme_status = parse_search3d_readme()
    js_status = parse_main_js()

    for source_name, statuses in (("Search_3D README", readme_status), ("main.js", js_status)):
        missing = sorted(EXPECTED_SET - set(statuses))
        extra = sorted(set(statuses) - EXPECTED_SET)
        if missing:
            problems.append(f"{source_name} missing ids: {', '.join(missing)}")
        if extra:
            problems.append(f"{source_name} has unexpected ids: {', '.join(extra)}")

    for algo_id in EXPECTED_IDS:
        if algo_id in readme_status and algo_id in js_status and readme_status[algo_id] != js_status[algo_id]:
            problems.append(
                f"{algo_id} status mismatch: Search_3D README={readme_status[algo_id]} main.js={js_status[algo_id]}"
            )

    status_counts = Counter(readme_status.get(algo_id, "缺失") for algo_id in EXPECTED_IDS)
    implemented = sum(1 for algo_id in EXPECTED_IDS if readme_status.get(algo_id) == "已实现")
    approximate = sum(1 for algo_id in EXPECTED_IDS if readme_status.get(algo_id) == "近似实现")
    not_implemented = sum(1 for algo_id in EXPECTED_IDS if readme_status.get(algo_id) == "未实现")

    root_counts = parse_root_counts()
    expected_root = {
        "headline_done": implemented,
        "headline_approx": approximate,
        "bar_done": implemented,
        "bar_approx": approximate,
        "table_done": implemented,
        "table_approx": approximate,
    }
    for key, expected in expected_root.items():
        actual = root_counts.get(key)
        if actual != expected:
            problems.append(f"root README {key} is {actual}, expected {expected}")

    print("Search_3D status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"implemented              : {implemented}/37")
    print(f"approximate              : {approximate}/37")
    print(f"not implemented          : {not_implemented}/37")
    print()

    if not problems:
        print("OK: Search_3D status metadata is consistent.")
        return 0

    print(f"Found {len(problems)} inconsistency(ies):")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

