"""Compare our no-prefetch L1 read misses in k_* with cachegrind's (O2).

Small differences expected: cachegrind starts warm (runs setup code) and
counts line-straddling accesses twice.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from .config import BUILD_DIR, KERNEL_PREFIX, WORKLOADS
from .sim import L1_SIZE, L1_WAYS, L2_SIZE, L2_WAYS, simulate
from .trace import Trace


def cachegrind_kernel_reads(binary: Path, args: list[str]) -> tuple[int, int]:
    """(reads, L1 read misses) in k_* functions."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cg.out"
        subprocess.run(
            ["valgrind", "--tool=cachegrind", "--cache-sim=yes",
             f"--D1={L1_SIZE},{L1_WAYS},64", f"--LL={L2_SIZE},{L2_WAYS},64",
             f"--cachegrind-out-file={out}", str(binary), *args],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        events, fn = None, ""
        reads = misses = 0
        for line in out.read_text().splitlines():
            if line.startswith("events:"):
                events = line.split()[1:]
            elif line.startswith("fn="):
                fn = line[3:]
            elif line[:1].isdigit() and fn.startswith(KERNEL_PREFIX):
                vals = dict(zip(events, map(int, line.split()[1:])))
                reads += vals.get("Dr", 0)
                misses += vals.get("D1mr", 0)
    return reads, misses


def main() -> None:
    opt = "O2"
    workloads = sys.argv[1:] or list(WORKLOADS)
    print(f"{'workload':14s} {'reads ours':>11s} {'reads cg':>11s} "
          f"{'L1 miss ours':>13s} {'L1 miss cg':>11s} {'diff':>7s}")
    for w in workloads:
        kernel, args = WORKLOADS[w]
        cg_reads, cg_miss = cachegrind_kernel_reads(BUILD_DIR / opt / kernel, args)
        per_pc, s = simulate(Trace(w, opt))
        our_miss = sum(v[2] for v in per_pc.values())
        diff = (our_miss - cg_miss) / cg_miss * 100 if cg_miss else 0.0
        print(f"{w:14s} {s['reads']:11,d} {cg_reads:11,d} {our_miss:13,d} {cg_miss:11,d} {diff:6.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
