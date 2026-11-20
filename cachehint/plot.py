"""Plots from results/*.csv into results/plots/.

Usage: python3 -m cachehint.plot   (run after score, evaluate, compiler data, replacement)
"""

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .compiler import analyse
from .config import OPT_LEVELS, RESULTS_DIR
from .label import CONFIG_NAMES

PLOT_DIR = RESULTS_DIR / "plots"
NAMES = {"qwen25-coder-32b": "Qwen2.5-Coder-32B", "claude-sonnet-5.5": "Claude Sonnet 5.5",
         "glm-5.3": "GLM-5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
LLMS = ["claude-sonnet-5.5", "qwen25-coder-32b", "deepseek-v4.1-flash", "glm-5.3"]
COLOR = {"base": "#9aa1ad", "online": "#1baf7a", "oracle": "#2a78d6", "llm": "#eb6834"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8ec", "axes.axisbelow": True})


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def speedups(opt):
    """workload -> {predictor: speedup over none}."""
    with open(RESULTS_DIR / f"phase3_{opt}.csv") as f:
        return {r["workload"]: {k: float(r["none"]) / float(v) for k, v in r.items() if k != "workload" and v}
                for r in csv.DictReader(f)}


def hbar(ax, labels, values, colors, fmt, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt(v), va="center", fontsize=9)
    if ref is not None:
        ax.axvline(ref, color="#555", ls="--", lw=1)


def plot_speedup():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, opt in zip(axes, OPT_LEVELS):
        sp = speedups(opt)
        gm = {p: geomean([sp[w][p] for w in sp]) for p in next(iter(sp.values()))}
        best = max(CONFIG_NAMES, key=gm.get)
        preds = [(f"Best single ({best})", best, "base"), ("Online per-PC", "online", "online"),
                 ("Oracle hints", "oracle", "oracle")] + [(NAMES[m], m, "llm") for m in LLMS]
        vals = [gm[k] for _, k, _ in preds]
        hbar(ax, [l for l, _, _ in preds], vals, [COLOR[c] for _, _, c in preds], lambda v: f"{v:.2f}x", ref=gm[best])
        ax.set_xlim(1, max(vals) * 1.15)
        ax.set_title(f"-{opt}")
        ax.set_xlabel("speedup over no prefetching (geomean)")
    fig.suptitle("Whole-program speedup (dashed: best single prefetcher)")
    return fig


def plot_accuracy():
    with open(RESULTS_DIR / "scores.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["opt"] == "pooled"]
    rows.sort(key=lambda r: float(r["matter_regret"]))
    labels = [NAMES.get(r["predictor"], r["predictor"]) for r in rows]
    colors = [COLOR["llm"] if r["predictor"] in NAMES else COLOR["base"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    hbar(a, labels, [100 * float(r["matter_near_best"]) for r in rows], colors, lambda v: f"{v:.0f}%")
    a.set_xlim(0, 100)
    a.set_title("Near-best picks (higher is better)")
    a.set_xlabel("% of loads that matter")
    hbar(b, labels, [float(r["matter_regret"]) for r in rows], colors, lambda v: f"{v:.1f}")
    b.set_title("Regret (lower is better)")
    b.set_xlabel("extra cycles per access vs best config")
    fig.suptitle(f"Per-load hint quality ({rows[0]['matter_n']} loads that matter, all opt levels)")
    return fig


def plot_heatmap(opt):
    sp = speedups(opt)
    gm = {p: geomean([sp[w][p] for w in sp]) for p in next(iter(sp.values()))}
    best = max(CONFIG_NAMES, key=gm.get)
    cols = [best, "online", "oracle"] + LLMS
    heads = [best, "online", "oracle"] + [NAMES[m].split()[0].split("-")[0] for m in LLMS]
    workloads = sorted(sp)
    data = np.array([[sp[w][c] for c in cols] for w in workloads] + [[gm[c] for c in cols]])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(np.log2(data), cmap="RdBu", norm=TwoSlopeNorm(0, -0.5, 3), aspect="auto")
    ax.set_xticks(range(len(cols)), heads)
    ax.set_yticks(range(len(workloads) + 1), workloads + ["geomean"])
    ax.grid(False)
    ax.axhline(len(workloads) - 0.5, color="white", lw=3)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(math.log2(v)) > 1.6 else "black")
    ax.set_title(f"Speedup over no prefetching per workload (-{opt})")
    return fig


def plot_sensitivity():
    sp = speedups("O2")["list_seq"]
    _, s = analyse()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    hbar(a, ["own-input hints", "other-input hints"], [sp["oracle"], sp["oracle-other-input"]],
         [COLOR["oracle"], COLOR["base"]], lambda v: f"{v:.2f}x")
    a.set_xlim(0, sp["oracle"] * 1.2)
    a.set_title("Input: list_seq (-O2), oracle hints from each input")
    a.set_xlabel("speedup over no prefetching")
    labels = ["Truth: one hint fits all", "Truth: same best"] + [f"{NAMES[m]} consistent" for m in LLMS]
    vals = [s["one_fits_all"], s["same_best"]] + [s["consistent"].get(m, 0) for m in LLMS]
    hbar(b, labels, vals, [COLOR["oracle"]] * 2 + [COLOR["llm"]] * len(LLMS), lambda v: f"{v:.0f}/{s['n']}")
    b.set_xlim(0, s["n"] * 1.15)
    b.set_xticks(range(0, s["n"] + 1, 5))
    b.set_title(f"Compiler: {s['n']} source lines at 2+ opt levels")
    b.set_xlabel("source lines")
    return fig


def plot_replacement(opt="O2"):
    with open(RESULTS_DIR / f"replacement_{opt}.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["workload"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in zip(axes, ["speedup_no_pf", "speedup_with_pf"], ["no prefetching", "oracle prefetch hints"]):
        v = np.array([float(r[col]) for r in rows]) - 1
        y = np.arange(len(rows))[::-1]
        ax.barh(y, v, color=[COLOR["oracle"] if x >= 0 else "#e34948" for x in v], height=0.6)
        ax.set_yticks(y, [r["workload"] for r in rows])
        ax.axvline(0, color="#555", lw=1)
        ax.set_xlim(-0.07, 0.07)
        ax.xaxis.set_major_formatter(lambda x, _: f"{1 + x:.2f}")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"With {title}")
        ax.set_xlabel("speedup vs plain LRU")
    fig.suptitle(f"Non-temporal replacement hint (-{opt})")
    return fig


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figs = {"speedup": plot_speedup(), "accuracy": plot_accuracy(), "sensitivity": plot_sensitivity(),
            "replacement": plot_replacement(), **{f"heatmap_{o}": plot_heatmap(o) for o in OPT_LEVELS}}
    for name, fig in figs.items():
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{name}.png", dpi=150)
        plt.close(fig)
        print(PLOT_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
