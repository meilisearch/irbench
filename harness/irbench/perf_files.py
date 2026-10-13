"""Where the perf stage writes its reports, and how every reader finds them.

One loadgen report per page size: perf-p<N>.json. Bundles from before page sizes hold a
single perf.json measured at top_k = 100, read as page size 100.
"""

import re
from pathlib import Path
from typing import Any

from irbench.io import read_json

# Legacy bundles were measured at the scenario's top_k, which is 100 for every scenario that
# produced one.
LEGACY_PAGE_SIZE = 100
LEGACY_NAME = "perf.json"
_NAME = re.compile(r"perf-p(\d+)\.json")


def perf_path(run_dir: Path, page_size: int) -> Path:
    return run_dir / f"perf-p{page_size}.json"


def perf_report_paths(run_dir: Path) -> dict[int, Path]:
    found = {
        int(m.group(1)): p for p in run_dir.glob("perf-p*.json") if (m := _NAME.fullmatch(p.name))
    }
    if found:
        return dict(sorted(found.items()))
    legacy = run_dir / LEGACY_NAME
    return {LEGACY_PAGE_SIZE: legacy} if legacy.exists() else {}


def read_perf_reports(run_dir: Path) -> dict[int, dict[str, Any]]:
    return {n: read_json(p) for n, p in perf_report_paths(run_dir).items()}
