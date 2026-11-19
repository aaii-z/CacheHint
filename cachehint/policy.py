"""Per-PC prefetcher ensembles: one shared instance per prefetcher type."""

from .sim import IPStride, NextLine, Stream, all_configs, config_name

CONFIGS = all_configs()


def split(cfg: str) -> tuple[str, int]:
    """'stride4' -> ('stride', 4); 'none' -> ('none', 0)."""
    pf = cfg.rstrip("0123456789")
    return pf, int(cfg[len(pf):] or 0)


def ensemble():
    return {"nextline": NextLine(1), "stride": IPStride(1), "stream": Stream(1)}


class Static:
    """Route each PC per a fixed hint table {pc: config}. Unhinted PCs: no prefetch."""

    def __init__(self, hints: dict[int, str]):
        pfs = ensemble()
        self.table = {}
        for pc, cfg in hints.items():
            pf, deg = split(cfg)
            if pf != "none":
                self.table[pc] = (pfs[pf], deg)

    def route(self, pc):
        return self.table.get(pc)


class Online:
    """Per-PC trial and error, no hints: try each config for TRIAL reads,
    keep the one with the least wait for EXPLOIT reads, then re-explore."""

    TRIAL, EXPLOIT = 128, 16384

    def __init__(self):
        pfs = ensemble()
        self.choices = [None if pf == "none" else (pfs[pf], d) for pf, d in CONFIGS]
        self.state = {}  # pc -> [config idx, reads in phase, waits per config, exploiting]

    def _get(self, pc):
        s = self.state.get(pc)
        if s is None:
            s = self.state[pc] = [0, 0, [0] * len(self.choices), False]
        return s

    def observe(self, pc, wait):
        s = self._get(pc)
        s[1] += 1
        if s[3]:
            if s[1] >= self.EXPLOIT:
                s[0], s[1], s[2], s[3] = 0, 0, [0] * len(self.choices), False
            return
        s[2][s[0]] += wait
        if s[1] >= self.TRIAL:
            s[1] = 0
            if s[0] + 1 < len(self.choices):
                s[0] += 1
            else:
                s[0] = min(range(len(s[2])), key=s[2].__getitem__)
                s[3] = True

    def route(self, pc):
        s = self._get(pc)
        return self.choices[s[0]]


__all__ = ["Static", "Online", "split", "config_name"]
