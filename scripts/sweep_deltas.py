"""Per-base nDCG@10 change of every sweep variant against `tuned+base` (tune half).

Input: the concatenated output of `irbench sweep-summary`, each block headed by
`### <engine> <scenario>`. Run from the repository root (reads engines/*/sweeps/).

  for e in ...; do for s in ...; do echo "### $e $s"; irbench sweep-summary ...; done; done > summary.tsv
  uv run python scripts/sweep_deltas.py summary.tsv
"""

import sys

import yaml

path = sys.argv[1]
blocks = open(path).read().split("### ")[1:]
for b in blocks:
    lines = b.strip().splitlines()
    eng, scen = lines[0].split()
    if len(lines) < 3:
        print(eng, scen, "NO DATA", lines[1:])
        continue
    hdr = lines[1].split("\t")
    rows = {
        row.split("\t")[0]: dict(zip(hdr[1:], map(float, row.split("\t")[1:]))) for row in lines[2:]
    }
    cols = [
        c for c in ("keyword/keyword:all", "semantic/semantic:all", "hybrid/hybrid:all") if c in hdr
    ]
    sw = yaml.safe_load(open(f"engines/{eng}/sweeps/{scen}.yaml"))["variants"]
    base = rows["tuned+base"]
    print(f"== {eng} {scen}  base: " + "  ".join(f"{c.split('/')[0]}={base[c]:.4f}" for c in cols))
    for t, r in sorted(rows.items()):
        if t == "tuned+base":
            continue
        v = t.split("+", 1)[1]
        d = "  ".join(f"{c.split('/')[0][:3]} {r[c] - base[c]:+.4f}" for c in cols)
        print(
            f"   {v:12s} {d}   {yaml.safe_dump(sw.get(v), default_flow_style=True, width=200).strip()}"
        )
