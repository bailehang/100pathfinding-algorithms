"""Check Search_3D algorithm status documentation consistency.

This verifies the 37 3D aerial algorithm entries across:

* ``Search_3D/README.md`` status table;
* ``Search_3D/src/main.js`` algorithm detail metadata;
* the root ``README.md`` 3D headline and simplified implemented / not
  implemented counts.

The root README intentionally uses a simplified two-state summary: every status
except ``未实现`` counts as implemented.
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
    r"当前进度：37 个 3D 空中算法条目已建档，其中\s*(\d+)\s*个已实现、\s*(\d+)\s*个未实现"
)
ROOT_IMPLEMENTED_LINE_RE = re.compile(r"3D IMPLEMENTED\s+\[[#\-]+\]\s+\d+%\s+(\d+)\s*/\s*37")
ROOT_NOT_IMPLEMENTED_LINE_RE = re.compile(r"3D NOT IMPLEMENTED\s+\[[#\-]+\]\s+\d+%\s+(\d+)\s*/\s*37")
ROOT_TABLE_RE = re.compile(r"\|\s*(已实现|未实现)\s*\|\s*(\d+)\s*\|")


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
        "headline_todo": None,
        "bar_done": None,
        "bar_todo": None,
        "table_done": None,
        "table_todo": None,
    }
    if match := ROOT_PROGRESS_RE.search(text):
        counts["headline_done"] = int(match.group(1))
        counts["headline_todo"] = int(match.group(2))
    if match := ROOT_IMPLEMENTED_LINE_RE.search(text):
        counts["bar_done"] = int(match.group(1))
    if match := ROOT_NOT_IMPLEMENTED_LINE_RE.search(text):
        counts["bar_todo"] = int(match.group(1))
    for label, value in ROOT_TABLE_RE.findall(text):
        if label == "已实现":
            counts["table_done"] = int(value)
        if label == "未实现":
            counts["table_todo"] = int(value)
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
    implemented = sum(1 for algo_id in EXPECTED_IDS if readme_status.get(algo_id) in IMPLEMENTED_STATUSES)
    not_implemented = sum(1 for algo_id in EXPECTED_IDS if readme_status.get(algo_id) == "未实现")

    root_counts = parse_root_counts()
    expected_root = {
        "headline_done": implemented,
        "headline_todo": not_implemented,
        "bar_done": implemented,
        "bar_todo": not_implemented,
        "table_done": implemented,
        "table_todo": not_implemented,
    }
    for key, expected in expected_root.items():
        actual = root_counts.get(key)
        if actual != expected:
            problems.append(f"root README {key} is {actual}, expected {expected}")

    print("Search_3D status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"simplified implemented    : {implemented}/37")
    print(f"simplified not implemented: {not_implemented}/37")
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
