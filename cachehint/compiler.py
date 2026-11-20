"""Phase 3: compiler sensitivity. Does the best prefetcher for a source line
survive a change of optimisation level?

For each source line with a load that matters at 2+ opt levels, compares the
best prefetcher type per level, and whether one hint could serve all levels
(their near-best sets overlap). Also checks whether each LLM gives the same
hint for that line at every level.
"""

import csv
import json
from collections import defaultdict

from .config import OPT_LEVELS, RESULTS_DIR
from .hints import HINT_DIR, MODELS


def analyse():
    """Returns (per-line rows, summary {same_best, one_fits_all, n, consistent: {model: count}})."""
    by_line = defaultdict(dict)  # (workload, src) -> opt -> [rows that matter]
    for opt in OPT_LEVELS:
        with open(RESULTS_DIR / f"labels_{opt}.csv") as f:
            for r in csv.DictReader(f):
                if r["matters"] == "1":
                    by_line[(r["workload"], r["src"])].setdefault(opt, []).append(r)

    hints = {}  # model -> (opt, kernel, pc) -> prefetcher type
    for name in MODELS:
        path = HINT_DIR / f"{name}.jsonl"
        if path.exists():
            hints[name] = {(r["opt"], r["kernel"], r["pc"]): r["prefetcher"]
                           for r in map(json.loads, path.read_text().splitlines())}

    lines = {k: v for k, v in by_line.items() if len(v) >= 2}
    rows = []
    summary = {"n": len(lines), "same_best": 0, "one_fits_all": 0, "consistent": dict.fromkeys(hints, 0)}
    for (w, src), per_opt in sorted(lines.items()):
        best = {o: "/".join(sorted({r["label_pf"] for r in rs})) for o, rs in per_opt.items()}
        ok_sets = [set.union(*(set(r["acceptable_pf"].split()) for r in rs)) for rs in per_opt.values()]
        fits = bool(set.intersection(*ok_sets))
        summary["same_best"] += len(set(best.values())) == 1
        summary["one_fits_all"] += fits
        rows.append((w, src, best, fits))
        for name, h in hints.items():
            given = {h.get((o, r["kernel"], r["pc"])) for o, rs in per_opt.items() for r in rs}
            summary["consistent"][name] += len(given) == 1
    return rows, summary


def main() -> None:
    rows, s = analyse()
    print(f"{'workload':14s} {'source':20s}" + "".join(f"{o:>10s}" for o in OPT_LEVELS) + "  one hint fits all?")
    for w, src, best, fits in rows:
        print(f"{w:14s} {src:20s}" + "".join(f"{best.get(o, '-'):>10s}" for o in OPT_LEVELS)
              + f"  {'yes' if fits else 'NO'}")
    n = s["n"]
    print(f"\n{n} source lines with a load that matters at 2+ opt levels")
    print(f"  same best prefetcher at every level: {s['same_best']}/{n}")
    print(f"  one hint near-best at every level:   {s['one_fits_all']}/{n}")
    for name, c in s["consistent"].items():
        print(f"  {name:22s} same hint at every level: {c}/{n}")


if __name__ == "__main__":
    main()
