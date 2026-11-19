"""Build ground-truth labels: best prefetcher config per load (PF-LLM Table 1 method).

Usage: python3 -m cachehint.label [--relabel] [OPT ...]   (default: all opt levels)
  --relabel  recompute labels from existing CSV AMATs (no simulation)
Outputs per opt level: results/labels_<opt>.csv, results/workload_amat_<opt>.csv
"""

import csv
import subprocess
import sys
from multiprocessing import Pool

from .config import BUILD_DIR, OPT_LEVELS, RESULTS_DIR, WORKLOADS
from .sim import PREFETCHERS, all_configs, config_name, simulate
from .trace import Trace, kernel_ranges

MIN_READS = 1000              # skip rarely executed loads
MATTERS_CYCLES = 2.0          # best-vs-worst AMAT gap for a load to "matter"
TIE_REL, TIE_ABS = 0.05, 0.5  # near-best tolerance (no bandwidth model, so small gains are unreliable)

# Tie-break: lowest degree, then this order.
TIE_ORDER = {"none": 0, "stride": 1, "stream": 2, "nextline": 3}
CONFIGS = all_configs()
CONFIG_NAMES = [config_name(p, d) for p, d in CONFIGS]
CONFIG_KEY = {config_name(p, d): (d, TIE_ORDER[p]) for p, d in CONFIGS}


def run_workload(job: tuple[str, str]):
    """Simulate one trace under every config."""
    workload, opt = job
    trace = Trace(workload, opt)
    per_cfg, summaries = {}, {}
    for (pf, deg), name in zip(CONFIGS, CONFIG_NAMES):
        per_cfg[name], summaries[name] = simulate(trace, PREFETCHERS[pf](deg))
    return workload, opt, per_cfg, summaries


def source_lines(binary, pcs: list[int]) -> dict[int, str]:
    """PC -> 'file.c:line'."""
    if not pcs:
        return {}
    out = subprocess.run(
        ["addr2line", "-e", str(binary), *[hex(p) for p in pcs]],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {pc: line.rsplit("/", 1)[-1].split(" ")[0] for pc, line in zip(pcs, out)}


def label_pc(amats: dict[str, float]):
    """Return (label, near-best configs, best-vs-worst spread)."""
    best = min(amats.values())
    tol = max(TIE_ABS, TIE_REL * best)
    acceptable = [n for n in CONFIG_NAMES if amats[n] <= best + tol]
    return min(acceptable, key=CONFIG_KEY.get), acceptable, max(amats.values()) - best


def pf_of(cfg: str) -> str:
    return cfg.rstrip("0123456789")


def label_fields(amats: dict[str, float]) -> dict:
    label, acceptable, spread = label_pc(amats)
    return {
        "label": label,
        "label_pf": pf_of(label),
        "acceptable": " ".join(acceptable),
        "acceptable_pf": " ".join(sorted({pf_of(a) for a in acceptable})),
        "spread": round(spread, 2),
        "matters": int(spread >= MATTERS_CYCLES),
    }


def relabel(opt: str) -> None:
    path = RESULTS_DIR / f"labels_{opt}.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(label_fields({n: float(r[f"amat_{n}"]) for n in CONFIG_NAMES}))
    write_csv(path, rows)
    print(f"{opt}: relabelled {len(rows)} loads")


def write_csv(path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = sys.argv[1:]
    if "--relabel" in args:
        args.remove("--relabel")
        for opt in args or OPT_LEVELS:
            relabel(opt)
        return
    for opt in args or OPT_LEVELS:
        label_opt(opt)


def label_opt(opt: str) -> None:
    jobs = [(w, opt) for w in WORKLOADS]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label_rows, amat_rows = [], []
    with Pool(3) as pool:
        for workload, opt, per_cfg, summaries in pool.imap_unordered(run_workload, jobs):
            kernel = WORKLOADS[workload][0]
            binary = BUILD_DIR / opt / kernel
            ranges = kernel_ranges(binary)
            pcs = sorted(pc for pc, s in per_cfg["none"].items() if s[0] >= MIN_READS)
            src = source_lines(binary, pcs)
            n_matter = 0
            for pc in pcs:
                amats = {n: per_cfg[n][pc][1] / per_cfg[n][pc][0] for n in CONFIG_NAMES}
                fields = label_fields(amats)
                n_matter += fields["matters"]
                none = per_cfg["none"][pc]
                label_rows.append({
                    "opt": opt, "workload": workload, "kernel": kernel,
                    "pc": hex(pc),
                    "func": next(name for lo, hi, name in ranges if lo <= pc < hi),
                    "src": src.get(pc, "?"),
                    "reads": none[0],
                    "miss_rate_none": round(none[2] / none[0], 4),
                    **{f"amat_{n}": round(amats[n], 2) for n in CONFIG_NAMES},
                    **fields,
                })
            amat_rows.append({"opt": opt, "workload": workload,
                              **{n: round(summaries[n]["amat"], 2) for n in CONFIG_NAMES}})
            print(f"{opt}/{workload}: {len(pcs)} loads, {n_matter} matter", flush=True)

    label_rows.sort(key=lambda r: (r["workload"], r["pc"]))
    amat_rows.sort(key=lambda r: r["workload"])
    write_csv(RESULTS_DIR / f"labels_{opt}.csv", label_rows)
    write_csv(RESULTS_DIR / f"workload_amat_{opt}.csv", amat_rows)
    print(f"{opt}: wrote {len(label_rows)} labels", flush=True)


if __name__ == "__main__":
    main()
