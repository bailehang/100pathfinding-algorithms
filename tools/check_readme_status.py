"""Check the README status tables against the actual files on disk.

The README maintains hand-written status tables (``✅`` / ``WIP`` / ``TODO``) and
progress counts for ~94 algorithms. Those are easy to let drift out of sync with
the real ``Search_2D/NNN_*.py`` demos and ``Search_2D/gif/NNN_*.gif`` previews.

This tool does *not* rewrite the curated table (``✅`` vs ``WIP`` is a human
judgement that cannot be inferred from file existence). It only reports
inconsistencies so they can be fixed deliberately:

In this README ``✅`` means "implemented *and* has a demo gif"; a file that
exists but has no gif is marked ``WIP``. The headline counts reflect that:
``IMPLEMENTED N/94`` counts files on disk, ``DEMO GIF M/94`` counts ``✅`` rows.

Reported inconsistencies:

* an algorithm marked ``✅`` with no demo gif (or no implementation file);
* an algorithm marked ``TODO`` that nevertheless has an implementation file;
* a duplicate algorithm number in the status tables;
* an implementation file whose number never appears in the tables;
* the headline progress counts disagreeing with what is on disk.

Exit code is non-zero when any inconsistency is found, so it can run in CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The report prints the ✅ emoji; force UTF-8 so it works on a GBK/cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
SEARCH_2D = REPO_ROOT / "Search_2D"
GIF_DIR = SEARCH_2D / "gif"

STATUS_DONE = "✅"
STATUS_TOKENS = (STATUS_DONE, "WIP", "TODO")

NUM_RE = re.compile(r"^\d{3}$")
FILE_NUM_RE = re.compile(r"^(\d{3})_")


def implemented_numbers() -> set[str]:
    """Three-digit prefixes of demo files that exist on disk."""
    nums = set()
    for path in SEARCH_2D.glob("*.py"):
        m = FILE_NUM_RE.match(path.name)
        if m:
            nums.add(m.group(1))
    return nums


def gif_numbers() -> set[str]:
    nums = set()
    if GIF_DIR.is_dir():
        for path in GIF_DIR.glob("*.gif"):
            m = FILE_NUM_RE.match(path.name)
            if m:
                nums.add(m.group(1))
    return nums


def parse_status_table() -> tuple[dict[str, str], list[str]]:
    """Return (number -> status) and a list of numbers seen (to spot dupes).

    Status table rows look like::

        | 001 | BFS | ✅ |  022  | D* Lite | WIP |

    so within a single row a three-digit cell is followed, a few cells later, by
    a status cell. We pair each number with the next status token after it.
    """
    statuses: dict[str, str] = {}
    order: list[str] = []

    for line in README.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        pending_num: str | None = None
        for cell in cells:
            if NUM_RE.match(cell):
                pending_num = cell
                continue
            if pending_num is None:
                continue
            token = next((t for t in STATUS_TOKENS if t == cell), None)
            if token is not None:
                order.append(pending_num)
                statuses[pending_num] = token
                pending_num = None
    return statuses, order


def main() -> int:
    impl = implemented_numbers()
    gifs = gif_numbers()
    statuses, order = parse_status_table()

    problems: list[str] = []

    # Duplicate numbers in the status tables.
    seen = set()
    for num in order:
        if num in seen:
            problems.append(f"duplicate number {num} in README status tables")
        seen.add(num)

    # Status vs files on disk. ✅ means "implemented and has a demo gif".
    for num, status in statuses.items():
        if status == STATUS_DONE:
            if num not in impl:
                problems.append(f"{num} marked {STATUS_DONE} but no Search_2D/{num}_*.py exists")
            elif num not in gifs:
                problems.append(f"{num} marked {STATUS_DONE} but no Search_2D/gif/{num}_*.gif exists")
        if status == "TODO" and num in impl:
            problems.append(f"{num} marked TODO but Search_2D/{num}_*.py exists (promote it?)")

    # Implementation files missing from the tables entirely.
    for num in sorted(impl):
        if num not in statuses:
            problems.append(f"Search_2D/{num}_*.py exists but {num} is absent from README tables")

    # Headline progress counts. IMPLEMENTED counts files on disk; DEMO GIF
    # counts ✅ rows (which should equal the gif files on disk).
    text = README.read_text(encoding="utf-8")
    done_count = sum(1 for s in statuses.values() if s == STATUS_DONE)
    if done_count != len(gifs):
        problems.append(
            f"{done_count} rows marked {STATUS_DONE} but {len(gifs)} gif files exist on disk"
        )
    claimed_impl = re.search(r"IMPLEMENTED.*?(\d+)\s*/\s*\d+", text, re.S)
    if claimed_impl and int(claimed_impl.group(1)) != len(impl):
        problems.append(
            f"README headline claims {claimed_impl.group(1)} implemented but "
            f"{len(impl)} implementation files exist on disk"
        )
    claimed_gif = re.search(r"DEMO GIF.*?(\d+)\s*/\s*\d+", text, re.S)
    if claimed_gif and int(claimed_gif.group(1)) != len(gifs):
        problems.append(
            f"README headline claims {claimed_gif.group(1)} demo gifs but "
            f"{len(gifs)} gif files exist on disk"
        )

    print(f"implementation files on disk : {len(impl)}")
    print(f"gif files on disk            : {len(gifs)}")
    print(f"rows marked {STATUS_DONE} in README     : {done_count}")
    print(f"status rows parsed           : {len(statuses)}")
    print()

    if not problems:
        print("OK: README status tables are consistent with the files on disk.")
        return 0

    print(f"Found {len(problems)} inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
