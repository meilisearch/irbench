"""Final-run summary for one scenario: test-half nDCG@10 per base, indexing time, memory,
disk, and keyword/hybrid latency (p50/p99 ms) and max QPS at depth 100, for the tuned and
default tracks on aws-c7i-4xlarge. Run from the repository root on the bench machine.

  python3 scripts/finals_table.py knowledge-base__covid-papers
"""

import json
import os
import sys

scen = sys.argv[1] if len(sys.argv) > 1 else "ecommerce__product-search"
engs = ["meilisearch", "elasticsearch", "typesense", "qdrant", "postgres-paradedb", "postgres"]


def f(x):
    return f"{x['p50_ms']:5.1f}/{x['p99_ms']:6.1f} {x['max_qps'] or 0:6.0f}"


print(
    f"{'engine':18s}{'track':8s} {'kw':>6s}{'sem':>6s}{'hyb':>6s} | {'index_s':>8s}{'mem_GB':>7s}{'disk_GB':>8s} | kw p50/p99 maxqps | hyb p50/p99 maxqps"
)
for e in engs:
    for t in ("tuned", "default"):
        d = f".irbench/runs/{scen}__{e}__bge-small-en-v1.5__{t}__aws-c7i-4xlarge"
        if not os.path.exists(d + "/scores.json"):
            continue
        s = json.load(open(d + "/scores.json"))
        ix = json.load(open(d + "/index.json"))
        rel = {
            r["base"]: r["metrics"]["ndcg@10"]
            for r in s["relevance"]
            if r.get("half") == "test"
            and r.get("perturbation") == "all"
            and r.get("group") is None
            and r["mode"] == r["base"]
        }
        p = (s.get("perf") or {}).get("100", {})
        perf = f"{f(p['keyword/keyword'])} | {f(p['hybrid/hybrid'])}" if p else "-"
        print(
            f"{e:18s}{t:8s} {rel['keyword']:6.3f}{rel['semantic']:6.3f}{rel['hybrid']:6.3f} | {ix['seconds']:8.1f}{ix['memory_peak_bytes'] / 1e9:7.2f}{ix['disk_bytes'] / 1e9:8.2f} | {perf}"
        )
