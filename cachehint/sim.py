"""Trace-driven L1/L2 cache simulator with pluggable L1D prefetchers.

In-order blocking core: each read waits for its data, then GAP cycles pass.
Stores update the cache but never stall. Prefetch timeliness is modelled
(a prefetched line is usable only after its fetch latency); bandwidth is not.
"""

from collections import OrderedDict

LINE_BITS = 6  # 64 B lines
L1_SIZE, L1_WAYS, L1_LAT = 32 * 1024, 8, 4
L2_SIZE, L2_WAYS, L2_LAT = 512 * 1024, 8, 14
DRAM_LAT = 200
GAP = 4  # non-memory cycles between accesses


class Cache:
    """Set-associative LRU cache of line numbers."""

    def __init__(self, size: int, ways: int):
        self.nsets = size // (ways << LINE_BITS)
        self.ways = ways
        self.sets = [[] for _ in range(self.nsets)]  # LRU first, MRU last

    def contains(self, line: int) -> bool:
        return line in self.sets[line % self.nsets]

    def touch(self, line: int) -> bool:
        """Lookup; on hit, move to MRU."""
        s = self.sets[line % self.nsets]
        if line in s:
            if s[-1] != line:
                s.remove(line)
                s.append(line)
            return True
        return False

    def insert(self, line: int):
        """Insert as MRU; return evicted line or None."""
        s = self.sets[line % self.nsets]
        if line in s:
            s.remove(line)
            s.append(line)
            return None
        victim = s.pop(0) if len(s) >= self.ways else None
        s.append(line)
        return victim


# Prefetchers: train() sees every demand read, returns lines to prefetch.

class NoPrefetcher:
    name = "none"

    def __init__(self, degree: int = 0):
        self.degree = 0

    def train(self, pc: int, addr: int, line: int) -> list[int]:
        return []


class NextLine:
    """Fetch the next `degree` lines on each new line."""

    name = "nextline"

    def __init__(self, degree: int):
        self.degree = degree
        self.last_line = -1

    def train(self, pc, addr, line):
        if line == self.last_line:
            return []
        self.last_line = line
        return [line + k for k in range(1, self.degree + 1)]


class IPStride:
    """Per-PC line-stride detector, like ChampSim's ip_stride."""

    name = "stride"
    TABLE_SIZE = 256

    def __init__(self, degree: int):
        self.degree = degree
        self.table: OrderedDict[int, list[int]] = OrderedDict()  # pc -> [last_line, stride, conf]

    def train(self, pc, addr, line):
        e = self.table.get(pc)
        if e is None:
            if len(self.table) >= self.TABLE_SIZE:
                self.table.popitem(last=False)
            self.table[pc] = [line, 0, 0]
            return []
        self.table.move_to_end(pc)
        stride = line - e[0]
        if stride == 0:
            return []
        if stride == e[1]:
            e[2] = min(e[2] + 1, 3)
        else:
            e[2] = max(e[2] - 1, 0)
            if e[2] == 0:
                e[1] = stride
        e[0] = line
        if e[2] >= 2:
            return [line + e[1] * k for k in range(1, self.degree + 1)]
        return []


class Stream:
    """PC-agnostic stream detector per 4 KB page; runs 2x degree lines ahead."""

    name = "stream"
    TABLE_SIZE = 64
    PAGE_LINES = 64

    def __init__(self, degree: int):
        self.degree = degree
        self.table: OrderedDict[int, list[int]] = OrderedDict()  # page -> [last_line, dir, conf]

    def train(self, pc, addr, line):
        page = line // self.PAGE_LINES
        e = self.table.get(page)
        if e is None:
            e = [line, 0, 0]
            for nb, d in ((page - 1, 1), (page + 1, -1)):  # stream entering from a neighbour page
                n = self.table.get(nb)
                if n is not None and n[1] == d and n[2] >= 2:
                    e = [line, d, n[2]]
                    break
            if len(self.table) >= self.TABLE_SIZE:
                self.table.popitem(last=False)
            self.table[page] = e
            if e[2] < 2:
                return []
        else:
            self.table.move_to_end(page)
            delta = line - e[0]
            if delta == 0:
                return []
            d = 1 if delta > 0 else -1
            if d == e[1] and abs(delta) <= 4:
                e[2] = min(e[2] + 1, 3)
            else:
                e[1], e[2] = d, 0
            e[0] = line
            if e[2] < 2:
                return []
        d = e[1]
        return [line + d * k for k in range(1, 2 * self.degree + 1)]


PREFETCHERS = {"none": NoPrefetcher, "nextline": NextLine, "stride": IPStride, "stream": Stream}
DEGREES = [1, 2, 4]


def all_configs() -> list[tuple[str, int]]:
    return [("none", 0)] + [(p, d) for p in ("nextline", "stride", "stream") for d in DEGREES]


def config_name(pf: str, degree: int) -> str:
    return pf if pf == "none" else f"{pf}{degree}"


def simulate(trace, prefetcher=None, policy=None):
    """Simulate (pc, addr, kind) tuples with one `prefetcher`, or a per-PC `policy` {pc: prefetcher}.

    Returns (per_pc {pc: [reads, wait_cycles, l1_misses]}, summary dict).
    """
    if prefetcher is None and policy is None:
        prefetcher = NoPrefetcher()

    l1 = Cache(L1_SIZE, L1_WAYS)
    l2 = Cache(L2_SIZE, L2_WAYS)
    ready = {}         # line -> cycle its in-flight fetch completes
    pf_unused = set()  # prefetched lines not yet used
    per_pc = {}
    now = 0
    issued = useful = late = 0

    for pc, addr, kind in trace:
        line = addr >> LINE_BITS
        if kind == 1:  # store
            if not l1.touch(line):
                if not l2.touch(line):
                    l2.insert(line)
                victim = l1.insert(line)
                if victim is not None:
                    ready.pop(victim, None)
                    pf_unused.discard(victim)
            now += GAP
            continue

        # Demand read (load or modify)
        if l1.touch(line):
            wait = L1_LAT
            r = ready.get(line)
            if r is not None:
                if r > now:  # prefetch still in flight
                    wait += r - now
                    late += line in pf_unused
                del ready[line]
            if line in pf_unused:
                useful += 1
                pf_unused.discard(line)
            miss = 0
        else:
            if l2.touch(line):
                wait = L2_LAT
            else:
                wait = DRAM_LAT
                l2.insert(line)
            victim = l1.insert(line)
            if victim is not None:
                ready.pop(victim, None)
                pf_unused.discard(victim)
            miss = 1

        s = per_pc.get(pc)
        if s is None:
            per_pc[pc] = [1, wait, miss]
        else:
            s[0] += 1
            s[1] += wait
            s[2] += miss
        now += wait + GAP

        pf = prefetcher if policy is None else policy.get(pc)
        if pf is None:
            continue
        for pl in pf.train(pc, addr, line):
            if pl < 0 or l1.contains(pl):
                continue
            if l2.touch(pl):
                ready[pl] = now + L2_LAT
            else:
                ready[pl] = now + DRAM_LAT
                l2.insert(pl)
            victim = l1.insert(pl)
            if victim is not None:
                ready.pop(victim, None)
                pf_unused.discard(victim)
            pf_unused.add(pl)
            issued += 1

    reads = sum(s[0] for s in per_pc.values())
    total_wait = sum(s[1] for s in per_pc.values())
    misses = sum(s[2] for s in per_pc.values())
    summary = {
        "reads": reads,
        "amat": total_wait / reads if reads else 0.0,
        "l1_miss_rate": misses / reads if reads else 0.0,
        "pf_issued": issued,
        "pf_useful": useful,
        "pf_late": late,
        "pf_accuracy": useful / issued if issued else 0.0,
        "cycles": now,
    }
    return per_pc, summary
