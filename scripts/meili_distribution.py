"""Meilisearch semantic score distribution ((1 + cos) / 2) of the top-20 documents for the
tune-half original queries of each scenario. Feeds MEILI_DISTRIBUTION in make_sweeps.py.
Run: uv run python scripts/meili_distribution.py <scenario>..."""

import sys
from pathlib import Path

import numpy as np

from irbench.runs import RunSpec
from irbench.score.halves import task_halves
from irbench.stages.context import RunContext

root = Path.cwd()
for scenario in sys.argv[1:]:
    ctx = RunContext(
        root, RunSpec(scenario=scenario, engine="meilisearch", model="bge-small-en-v1.5")
    )
    halves = task_halves(scenario, ctx.all_tasks)
    qids = sorted(
        {
            t.qid
            for t in ctx.all_tasks
            if halves[t.task_id] == "tune" and t.perturbation == "original" and t.mode == "semantic"
        }
    )
    docs = ctx.doc_vectors
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.stack([ctx.query_vectors[x] for x in qids])
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    top = []
    for i in range(0, len(q), 256):
        sims = q[i : i + 256] @ docs.T
        top.append(-np.sort(-np.partition(sims, -20, axis=1)[:, -20:], axis=1))
    s = (1 + np.concatenate(top)) / 2
    print(
        f"{scenario}: queries={len(qids)} mean={s.mean():.4f} sigma={s.std():.4f} top1_mean={s[:, 0].mean():.4f}"
    )
