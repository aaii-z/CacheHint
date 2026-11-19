"""Assembly context around a load: +-WINDOW .text lines, local symbol names anonymised."""

import re
from functools import lru_cache

from .config import BUILD_DIR

WINDOW = 48
INSN_RE = re.compile(r"^\s+([0-9a-f]+):\t(.*)$")
FUNC_RE = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
SYM_RE = re.compile(r"<([^>+@]+)((?:\+0x[0-9a-f]+)?)>")


@lru_cache(maxsize=None)
def text_lines(opt: str, kernel: str) -> tuple[list[str], dict[int, int]]:
    """(.text lines with anonymised names, pc -> line index)."""
    asm = (BUILD_DIR / opt / f"{kernel}.asm").read_text().splitlines()
    names: dict[str, str] = {}

    def anon(name: str) -> str:
        return names.setdefault(name, f"fn{len(names)}")

    lines, index, in_text = [], {}, False
    for raw in asm:
        if raw.startswith("Disassembly of section"):
            in_text = raw.endswith(".text:")
            continue
        if not in_text or not raw.strip():
            continue
        if m := FUNC_RE.match(raw):
            lines.append(f"<{anon(m.group(1))}>:")
            continue
        if m := INSN_RE.match(raw):
            insn = SYM_RE.sub(lambda s: f"<{anon(s.group(1))}{s.group(2)}>", m.group(2))
            index[int(m.group(1), 16)] = len(lines)
            lines.append(f"{m.group(1)}: {insn}")
    return lines, index


def load_context(opt: str, kernel: str, pc: int, window: int = WINDOW) -> str:
    lines, index = text_lines(opt, kernel)
    i = index[pc]
    out = lines[max(0, i - window): i + window + 1]
    j = i - max(0, i - window)
    out[j] = f"<load>{out[j]}</load>"
    return "\n".join(out)
