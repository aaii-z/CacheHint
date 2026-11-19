"""Phase 3: whole-program speedup of hinted prefetcher ensembles.

Usage: python3 -m cachehint.evaluate [OPT ...]   (default: all opt levels)
Compares, per workload: single prefetchers (from Phase 1), an online per-PC
selector, the oracle (Phase 1 labels as hints), each LLM's hints, and, for the
linked list, oracle hints taken from the *other* input.
Writes results/phase3_<opt>.csv (total cycles per workload x predictor).
"""

import csv
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

from .config import OPT_LEVELS, RESULTS_DIR, WORKLOADS
from .hints import HINT_DIR, MODELS
from .label import CONFIG_NAMES
from .policy import Online, Static
from .sim import GAP, config_name, simulate
from .trace import Trace

OTHER_INPUT = {"list_seq": "list_shuf", "list_shuf": "list_seq"}


def labels(opt: str) -> dict[str, dict[int, str]]:
    """workload -> {pc: label config}."""
    out = {}
    with open(RESULTS_DIR / f"labels_{opt}.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["workload"], {})[int(r["pc"], 16)] = r["label"]
    return out


def llm_hints(opt: str) -> dict[str, dict[str, dict[int, str]]]:
    """model -> kernel -> {pc: config}; invalid answers become none."""
    out = {}
    for name in MODELS:
        path = HINT_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["opt"] != opt:
                continue
            pf = r["prefetcher"]
            cfg = "none" if pf == "invalid" else config_name(pf, r["degree"])
            out.setdefault(name, {}).setdefault(r["kernel"], {})[int(r["pc"], 16)] = cfg
    return out


def run(job):
    """Simulate one workload under every hint-driven policy. Returns {predictor: cycles}."""
    opt, workload, tables = job
    trace = Trace(workload, opt)
    cycles = {}
    for name, table in tables.items():
        policy = Online() if table is None else Static(table)
        cycles[name] = simulate(trace, policy=policy)[1]["cycles"]
    return opt, workload, cycles, len(trace), int(np.count_nonzero(trace.kind == 1))


def single_cycles(opt: str, workload: str, reads_plus_stores: int, stores: int) -> dict[str, float]:
    """Cycles for each single-prefetcher config, from Phase 1 AMATs."""
    with open(RESULTS_DIR / f"workload_amat_{opt}.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["workload"] == workload)
    reads = reads_plus_stores - stores
    return {n: reads * float(row[n]) + reads_plus_stores * GAP for n in CONFIG_NAMES}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def evaluate(opt: str) -> None:
    lab, hints = labels(opt), llm_hints(opt)
    jobs = []
    for w, (kernel, _) in WORKLOADS.items():
        tables = {"online": None, "oracle": lab.get(w, {})}
        if w in OTHER_INPUT:
            tables["oracle-other-input"] = lab.get(OTHER_INPUT[w], {})
        for model, by_kernel in hints.items():
            tables[model] = by_kernel.get(kernel, {})
        jobs.append((opt, w, tables))

    rows = []
    with Pool(3) as pool:
        for _, w, cyc, n, stores in pool.imap_unordered(run, jobs):
            rows.append({"workload": w, **single_cycles(opt, w, n, stores), **cyc})
            print(f"  {opt}/{w} done", flush=True)
    rows.sort(key=lambda r: r["workload"])

    path = RESULTS_DIR / f"phase3_{opt}.csv"
    cols = ["workload"] + [k for k in rows[0] if k != "workload"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols + ["oracle-other-input"], restval="")
        wr.writeheader()
        wr.writerows({k: (round(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    speed = {r["workload"]: {k: r["none"] / v for k, v in r.items() if k != "workload"} for r in rows}
    preds = [k for k in rows[0] if k not in ("workload", "oracle-other-input")]
    gm = {p: geomean([speed[w][p] for w in speed]) for p in preds}
    best_single = max(CONFIG_NAMES, key=gm.get)
    show = [best_single, "online", "oracle", *hints]

    print(f"\n== {opt}: speedup over no prefetching")
    print(f"{'workload':14s}" + "".join(f"{p[:13]:>14s}" for p in show))
    for w in speed:
        print(f"{w:14s}" + "".join(f"{speed[w][p]:14.2f}" for p in show))
    print(f"{'geomean':14s}" + "".join(f"{gm[p]:14.3f}" for p in show))
    span = gm["oracle"] - gm[best_single]
    print(f"{'gap closed':14s}" + "".join(
        f"{(gm[p] - gm[best_single]) / span:14.0%}" if span > 0 else f"{'-':>14s}" for p in show))
    print(f"(best single prefetcher = {best_single}; gap closed = share of best-single -> oracle gain)")

    print("\nInput dependence (speedup with oracle hints from each input):")
    for w in OTHER_INPUT:
        print(f"  {w:10s} own-input hints {speed[w]['oracle']:.2f}x, other-input hints {speed[w]['oracle-other-input']:.2f}x")


def main() -> None:
    for opt in sys.argv[1:] or OPT_LEVELS:
        evaluate(opt)


if __name__ == "__main__":
    main()
