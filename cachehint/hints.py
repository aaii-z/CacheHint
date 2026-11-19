"""Zero-shot prefetch hints from LLMs via OpenRouter.

Usage: python3 -m cachehint.hints [--dry-run] [MODEL ...]   (default: all MODELS)
Key: OPENROUTER_API_KEY or ~/.config/openrouter/key. Responses are cached in results/hints/<model>.jsonl
(resumable). One query per (opt, kernel, pc): workloads sharing a binary share the hint.
"""

import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .config import OPT_LEVELS, RESULTS_DIR, ROOT
from .context import load_context

# name -> (OpenRouter id, extra request params). Reasoning off where allowed,
# else minimal (Sonnet and GLM don't allow disabling it).
MODELS = {
    "qwen25-coder-32b": ("qwen/qwen-2.5-coder-32b-instruct", {}),
    "claude-sonnet-5.5": ("anthropic/claude-sonnet-5.5", {"reasoning": {"effort": "minimal"}}),
    "glm-5.3": ("z-ai/glm-5.3", {"reasoning": {"effort": "minimal"}}),
    "deepseek-v4.1-flash": ("deepseek/deepseek-v4.1-flash", {"reasoning": {"enabled": False}}),
}
PROMPT = "system_v2.txt"
MAX_TOKENS = 2048
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path.home() / ".config" / "openrouter" / "key"
HINT_DIR = RESULTS_DIR / "hints"
PREFETCHERS = {"none", "nextline", "stride", "stream"}
DEGREES = {1, 2, 4}


def queries() -> list[tuple[str, str, str]]:
    """Unique (opt, kernel, pc) across all label files."""
    seen = {}
    for opt in OPT_LEVELS:
        path = RESULTS_DIR / f"labels_{opt}.csv"
        if path.exists():
            with open(path) as f:
                for r in csv.DictReader(f):
                    seen[(opt, r["kernel"], r["pc"])] = None
    return list(seen)


def messages(opt: str, kernel: str, pc: str) -> list[dict]:
    system = (ROOT / "prompts" / PROMPT).read_text()
    asm = load_context(opt, kernel, int(pc, 16))
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"```asm\n{asm}\n```"}]


def parse(text: str) -> tuple[str, int]:
    """(prefetcher, degree) from the last valid JSON object; ('invalid', 0) if none."""
    for blob in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            obj = json.loads(blob)
            pf = str(obj["prefetcher"]).lower()
            deg = 0 if pf == "none" else int(obj.get("degree", 1))
        except (KeyError, TypeError, ValueError):
            continue
        if pf in PREFETCHERS and (pf == "none" or deg in DEGREES):
            return pf, deg
    return "invalid", 0


def ask(name: str, msgs: list[dict], key: str) -> dict:
    model_id, extra = MODELS[name]
    body = {"model": model_id, "messages": msgs, "temperature": 0, "max_tokens": MAX_TOKENS, **extra}
    for attempt in range(5):
        r = requests.post(URL, json=body, timeout=300,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"{model_id}: HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"{model_id}: gave up after retries")


def run_model(name: str, key: str) -> None:
    out = HINT_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["opt"], r["kernel"], r["pc"]))
    todo = [q for q in queries() if q not in done]
    print(f"{name}: {len(todo)} to query ({len(done)} cached)", flush=True)

    lock = threading.Lock()

    def work(q):
        opt, kernel, pc = q
        resp = ask(name, messages(opt, kernel, pc), key)
        text = resp["choices"][0]["message"].get("content") or ""
        pf, deg = parse(text)
        rec = {"opt": opt, "kernel": kernel, "pc": pc, "prefetcher": pf, "degree": deg,
               "prompt": PROMPT, "raw": text, "usage": resp.get("usage", {})}
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return pf

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, todo))
    print(f"{name}: done, {results.count('invalid')} invalid", flush=True)


def main() -> None:
    args = sys.argv[1:]
    dry = "--dry-run" in args
    names = [a for a in args if a != "--dry-run"] or list(MODELS)
    qs = queries()
    if dry:
        msgs = messages(*qs[0])
        print(msgs[0]["content"], msgs[1]["content"], sep="\n---\n")
        chars = sum(len(m["content"]) for q in qs for m in messages(*q))
        print(f"\n{len(qs)} queries per model, ~{chars // 4:,} input tokens per model")
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"set OPENROUTER_API_KEY or put the key in {KEY_FILE}")
    for name in names:
        run_model(name, key)


if __name__ == "__main__":
    main()
