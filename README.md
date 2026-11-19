# CacheHint

Mini replication of PF-LLM (Xu et al., ASPLOS '26): can an LLM pick the best
prefetcher for a load instruction from its code alone?

## Phases

1. **Ground truth**: trace kernels, simulate each prefetcher, label each load with its best config.
2. **LLM hints**: zero-shot hints from assembly context, scored against labels.
3. **Evaluation**: hinted prefetcher ensemble, input and compiler sensitivity, replacement hint.

## Setup

- **Kernels**: 13 C programs (`kernels/`), 14 workloads (`linked_list` runs `seq` and `shuf`), built at `-O0/-O2/-O3`.
- **Traces**: Valgrind `lackey`, `k_*` functions only.
- **Simulator**: L1D 32 KB 8-way (4 cyc), L2 512 KB 8-way (14 cyc), DRAM 200 cyc; in-order blocking core, no bandwidth model.
- **Prefetchers**: none, next-line, per-PC stride, stream; degree 1/2/4.
- **Labels** (`results/labels_<opt>.csv`): lowest-AMAT config per load, plus `acceptable` (configs within 5% of best) and `matters` (config choice changes AMAT by ≥2 cycles).
- **Hints**: ±48 lines of assembly around the load, local function names anonymised, prompt in `prompts/`. OpenRouter, temperature 0, reasoning off or minimal.
- **Models**: Qwen2.5-Coder-32B (PF-LLM's family, untuned), Claude Sonnet 5.5, GLM-5.3, DeepSeek V4.1 Flash.
- **Ensemble**: one shared instance per prefetcher; each load trains only the one its hint names. Unhinted loads: no prefetch.
- **Online baseline**: per-PC trial and error over all 10 configs (128 reads each), exploit best for 16K reads.

## Results

**Per-load accuracy** (64 loads where the choice matters, all opt levels):

| Predictor | Near-best | Regret (cycles/access) |
|---|---|---|
| Claude Sonnet 5.5 | 81% | 7.8 |
| always stream2 | 56% | 13.1 |
| DeepSeek V4.1 Flash | 58% | 24.6 |
| GLM-5.3 | 59% | 26.6 |
| Qwen2.5-Coder-32B | 58% | 29.7 |

**Whole-program speedup** over no prefetching (geomean, 14 workloads):

| | O0 | O2 | O3 |
|---|---|---|---|
| Best single (stream4) | 1.36 | 1.80 | 2.06 |
| Online per-PC | 1.40 | 1.84 | 2.07 |
| **Oracle hints** | **1.43** | **1.96** | **2.25** |
| Claude Sonnet 5.5 | 1.26 | 1.81 | 1.87 |
| Qwen2.5-Coder-32B | 1.33 | 1.15 | 1.16 |
| DeepSeek V4.1 Flash | 1.15 | 1.36 | 1.26 |
| GLM-5.3 | 1.15 | 1.18 | 1.28 |

Absolute speedups are inflated by the blocking core; compare rows, not to real CPUs.

**Findings**
- Perfect static hints beat the best single prefetcher by 6-9%, close to PF-LLM's reported 9.8%.
- No zero-shot LLM beats the best single prefetcher. A wrong hint switches prefetching off for that load, so bad hints cost more than no hints.
- Exact-match accuracy misleads: "always none" scores 93% on all O0 loads, 11% on loads that matter.
- Input dependence: `linked_list` hints from the shuffled input drop the sequential input from 8.15x to 1.00x (same binary, same PCs).
- Compiler sensitivity: best prefetcher per source line is the same at all opt levels for 15/19 lines; LLM hints for the same line agree for only 10-12/19.
- Replacement (non-temporal) hint: within ±5%; hurts `binsearch`, where one instruction touches both hot and cold lines.

## Run

```sh
python3 -m cachehint.build
python3 -m cachehint.trace
python3 -m cachehint.label            # simulate + label (all opt levels)
python3 -m cachehint.label --relabel  # re-derive labels from saved AMATs
python3 -m cachehint.validate         # vs cachegrind
python3 -m cachehint.hints            # needs OPENROUTER_API_KEY
python3 -m cachehint.score
python3 -m cachehint.evaluate         # whole-program speedups
python3 -m cachehint.compiler         # compiler sensitivity
python3 -m cachehint.replacement      # non-temporal hint
```

Validation (O2): L1 read misses match cachegrind within 0.8% on all workloads.
