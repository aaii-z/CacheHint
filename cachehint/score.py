"""Score LLM hints against Phase 1 labels, with simple baselines.

Metrics (per model, per opt level):
  exact      predicted config == label
  type       predicted prefetcher == label prefetcher
  near_best  predicted config is within tolerance of best (in `acceptable`)
  regret     mean AMAT of predicted config minus best AMAT (cycles); invalid counts as none
Reported on all loads and on loads where the choice matters.
Writes results/scores.csv.
"""

import csv
import json
from collections import Counter

from .config import OPT_LEVELS, RESULTS_DIR
from .hints import HINT_DIR
from .label import CONFIG_NAMES
from .sim import config_name

TYPES = ["none", "nextline", "stride", "stream"]


def load_labels(opt: str) -> list[dict]:
    with open(RESULTS_DIR / f"labels_{opt}.csv") as f:
        return list(csv.DictReader(f))


def load_hints() -> dict[str, dict[tuple, str]]:
    """model -> {(opt, kernel, pc): config name}."""
    out = {}
    for path in sorted(HINT_DIR.glob("*.jsonl")):
        hints = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            pf = r["prefetcher"]
            hints[(r["opt"], r["kernel"], r["pc"])] = "invalid" if pf == "invalid" else config_name(pf, r["degree"])
        out[path.stem] = hints
    return out


def metrics(rows: list[dict], preds: list[str]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    exact = typ = near = 0
    regret = 0.0
    for r, p in zip(rows, preds):
        cfg = "none" if p == "invalid" else p
        exact += cfg == r["label"]
        typ += cfg.rstrip("124") == r["label_pf"]
        near += cfg in r["acceptable"].split()
        regret += float(r[f"amat_{cfg}"]) - float(r[f"amat_{r['label']}"])
    return {"n": n, "exact": exact / n, "type": typ / n, "near_best": near / n, "regret": regret / n,
            "invalid": preds.count("invalid")}


def predictors(hints: dict, opt: str, rows: list[dict]) -> dict[str, list[str]]:
    matter = [r for r in rows if r["matters"] == "1"]
    majority = Counter(r["label"] for r in matter).most_common(1)[0][0] if matter else "none"
    preds = {
        "always-none": ["none"] * len(rows),
        "always-stream2": ["stream2"] * len(rows),
        f"majority ({majority})": [majority] * len(rows),
    }
    for model, h in hints.items():
        preds[model] = [h.get((opt, r["kernel"], r["pc"]), "invalid") for r in rows]
    return preds


def main() -> None:
    hints = load_hints()
    out_rows = []
    confusion = {m: Counter() for m in hints}
    pooled: dict[str, tuple[list, list]] = {}  # loads that matter, all opt levels
    for opt in OPT_LEVELS:
        rows = load_labels(opt)
        idx_matter = [i for i, r in enumerate(rows) if r["matters"] == "1"]
        print(f"\n== {opt}: {len(rows)} loads, {len(idx_matter)} matter")
        print(f"{'predictor':24s} {'exact':>6s} {'near':>6s} | {'exact*':>6s} {'type*':>6s} {'near*':>6s} {'regret*':>8s} {'inv':>4s}")
        for name, preds in predictors(hints, opt, rows).items():
            a = metrics(rows, preds)
            m = metrics([rows[i] for i in idx_matter], [preds[i] for i in idx_matter])
            print(f"{name:24s} {a['exact']:6.0%} {a['near_best']:6.0%} | {m.get('exact', 0):6.0%} "
                  f"{m.get('type', 0):6.0%} {m.get('near_best', 0):6.0%} {m.get('regret', 0):8.1f} {a['invalid']:4d}")
            out_rows.append({"opt": opt, "predictor": name,
                             **{f"all_{k}": v for k, v in a.items()},
                             **{f"matter_{k}": v for k, v in m.items()}})
            key = name.split(" ")[0]
            pr, pp = pooled.setdefault(key, ([], []))
            pr += [rows[i] for i in idx_matter]
            pp += [preds[i] for i in idx_matter]
            if name in confusion:
                for i in idx_matter:
                    p = "none" if preds[i] == "invalid" else preds[i]
                    confusion[name][(rows[i]["label_pf"], p.rstrip("124"))] += 1
    print("\n* = loads where the choice matters. regret = extra cycles per access vs best.")

    print(f"\n== Pooled, loads that matter ({len(next(iter(pooled.values()))[0])})")
    print(f"{'predictor':24s} {'exact':>6s} {'type':>6s} {'near':>6s} {'regret':>7s}")
    for name, (pr, pp) in sorted(pooled.items(), key=lambda kv: metrics(*kv[1])["regret"]):
        m = metrics(pr, pp)
        print(f"{name:24s} {m['exact']:6.0%} {m['type']:6.0%} {m['near_best']:6.0%} {m['regret']:7.1f}")
        out_rows.append({"opt": "pooled", "predictor": name, **{f"matter_{k}": v for k, v in m.items()}})

    for model, cm in confusion.items():
        print(f"\n{model}: label (rows) vs prediction (cols), loads that matter")
        print(f"{'':10s}" + "".join(f"{t:>10s}" for t in TYPES))
        for t in TYPES:
            print(f"{t:10s}" + "".join(f"{cm[(t, p)]:10d}" for p in TYPES))

    print("\nInput dependence (same PC, two inputs):")
    for opt in OPT_LEVELS:
        rows = {(r["workload"], r["pc"]): r for r in load_labels(opt) if r["kernel"] == "linked_list"}
        for (w, pc), r in sorted(rows.items()):
            if w == "list_seq" and r["matters"] == "1":
                shuf = rows.get(("list_shuf", pc))
                hint_str = ", ".join(f"{m}={h.get((opt, 'linked_list', pc), '-')}" for m, h in hints.items())
                print(f"  {opt} {pc}: seq best={r['label']}, shuf best={shuf['label'] if shuf else '?'} | {hint_str}")

    if out_rows:
        with open(RESULTS_DIR / "scores.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in out_rows for k in r}, key=lambda k: (k != "opt", k != "predictor", k)))
            w.writeheader()
            w.writerows(out_rows)


if __name__ == "__main__":
    main()
