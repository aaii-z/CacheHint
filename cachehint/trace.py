"""Record memory traces with Valgrind lackey, keeping only k_* accesses.

Output: data/traces/<opt>/<workload>.npz with pc, addr, kind (0 load, 1 store, 2 modify).
"""

import os
import subprocess
import sys
import time
from array import array
from multiprocessing import Pool

import numpy as np

from .config import BUILD_DIR, KERNEL_PREFIX, OPT_LEVELS, TRACE_DIR, WORKLOADS

KIND = {"L": 0, "S": 1, "M": 2}
CHUNK = 1 << 20


class Trace:
    """Re-iterable (pc, addr, kind) stream; converts to Python ints chunk by chunk."""

    def __init__(self, workload: str, opt: str):
        d = np.load(TRACE_DIR / opt / f"{workload}.npz")
        self.pc, self.addr, self.kind = d["pc"], d["addr"], d["kind"]

    def __len__(self):
        return len(self.kind)

    def __iter__(self):
        for i in range(0, len(self.kind), CHUNK):
            j = i + CHUNK
            yield from zip(self.pc[i:j].tolist(), self.addr[i:j].tolist(), self.kind[i:j].tolist())


def kernel_ranges(binary) -> list[tuple[int, int, str]]:
    """[lo, hi) address range of each k_* function."""
    out = subprocess.run(
        ["nm", "-S", "--defined-only", str(binary)],
        check=True, capture_output=True, text=True,
    ).stdout
    ranges = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "tT" and parts[3].startswith(KERNEL_PREFIX):
            lo, size = int(parts[0], 16), int(parts[1], 16)
            ranges.append((lo, lo + size, parts[3]))
    if not ranges:
        raise RuntimeError(f"no {KERNEL_PREFIX}* functions in {binary}")
    return sorted(ranges)


def trace_one(job: tuple[str, str]) -> str:
    workload, opt = job
    kernel, args = WORKLOADS[workload]
    binary = BUILD_DIR / opt / kernel
    ranges = kernel_ranges(binary)
    lo_all, hi_all = ranges[0][0], ranges[-1][1]

    # Stream lackey's log through a pipe; full logs would be GBs.
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        ["valgrind", "--tool=lackey", "--trace-mem=yes", f"--log-fd={write_fd}",
         str(binary), *args],
        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.close(write_fd)

    pcs, addrs, kinds = array("Q"), array("Q"), array("B")
    t0 = time.time()
    pc, in_kernel = 0, False
    with os.fdopen(read_fd, "r", buffering=1 << 20) as log:
        for line in log:
            c = line[0]
            if c == "I":  # "I  <pc>,<size>"
                pc = int(line[3:line.index(",", 3)], 16)
                in_kernel = lo_all <= pc < hi_all and any(lo <= pc < hi for lo, hi, _ in ranges)
            elif c == " " and in_kernel:  # " L|S|M <addr>,<size>"
                pcs.append(pc)
                addrs.append(int(line[3:line.index(",", 3)], 16))
                kinds.append(KIND[line[1]])
    if proc.wait() != 0:
        raise RuntimeError(f"{binary} {args} exited with {proc.returncode}")

    out = TRACE_DIR / opt / f"{workload}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pc=np.frombuffer(pcs, dtype=np.uint64),
        addr=np.frombuffer(addrs, dtype=np.uint64),
        kind=np.frombuffer(kinds, dtype=np.uint8),
    )
    n_loads = sum(1 for k in kinds if k != 1)
    return f"{opt}/{workload}: {len(kinds):,} accesses ({n_loads:,} reads) in {time.time() - t0:.0f}s"


def main() -> None:
    workloads = sys.argv[1:] or list(WORKLOADS)
    jobs = [(w, opt) for opt in OPT_LEVELS for w in workloads]
    with Pool(4) as pool:
        for msg in pool.imap_unordered(trace_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
