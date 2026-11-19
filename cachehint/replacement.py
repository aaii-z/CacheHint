"""Phase 3 stretch: a replacement hint. Loads whose lines are never reused get
a "non-temporal" hint: their fills go in at LRU and are not promoted on hits,
so they stop pushing reused data out of the cache.

Ground truth: profile each load with no prefetching; a load is non-temporal if
under NT_THRESHOLD of the lines it brings in are hit again more than
REUSE_WINDOW accesses later (i.e. beyond its own immediate spatial use).

Usage: python3 -m cachehint.replacement [OPT ...]   (default: O2 O3)
Writes results/replacement_<opt>.csv.
"""

import csv
import sys
from multiprocessing import Pool

from .config import RESULTS_DIR, WORKLOADS
from .evaluate import labels
from .policy import Static
from .sim import simulate
from .trace import Trace

NT_THRESHOLD = 0.05
MIN_FILLS = 1000


def run(job):
    opt, workload, oracle = job
    trace = Trace(workload, opt)
    _, base = simulate(trace, track_reuse=True)
    nt = {pc for pc, (fills, reused) in base["reuse"].items()
          if fills >= MIN_FILLS and reused / fills < NT_THRESHOLD}
    cyc = {"lru": base["cycles"], "lru+nt": simulate(trace, nontemporal=nt)[1]["cycles"]}
    cyc["pf"] = simulate(trace, policy=Static(oracle))[1]["cycles"]
    cyc["pf+nt"] = simulate(trace, policy=Static(oracle), nontemporal=nt)[1]["cycles"]
    reused_pcs = sum(1 for f, r in base["reuse"].values() if f >= MIN_FILLS and r / f >= NT_THRESHOLD)
    return workload, len(nt), reused_pcs, cyc


def evaluate(opt: str) -> None:
    lab = labels(opt)
    jobs = [(opt, w, lab.get(w, {})) for w in WORKLOADS]
    rows = []
    with Pool(3) as pool:
        for w, n_nt, n_reused, cyc in pool.imap_unordered(run, jobs):
            rows.append({"workload": w, "nt_loads": n_nt, "reused_loads": n_reused,
                         "speedup_no_pf": cyc["lru"] / cyc["lru+nt"],
                         "speedup_with_pf": cyc["pf"] / cyc["pf+nt"], **cyc})
    rows.sort(key=lambda r: r["workload"])
    with open(RESULTS_DIR / f"replacement_{opt}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n== {opt}: non-temporal hint speedup (vs plain LRU)")
    print(f"{'workload':14s} {'NT loads':>9s} {'reused':>7s} {'no prefetch':>12s} {'oracle prefetch':>16s}")
    for r in rows:
        print(f"{r['workload']:14s} {r['nt_loads']:9d} {r['reused_loads']:7d} "
              f"{r['speedup_no_pf']:12.3f} {r['speedup_with_pf']:16.3f}")


def main() -> None:
    for opt in sys.argv[1:] or ["O2", "O3"]:
        evaluate(opt)


if __name__ == "__main__":
    main()
