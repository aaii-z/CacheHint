# CacheHint

Mini replication of PF-LLM (Xu et al., ASPLOS '26): can an LLM pick the best
prefetcher for a load instruction from its code alone?

## Phases

1. **Ground truth**: trace kernels, simulate each prefetcher, label each load with its best config.
2. **LLM hints**: zero-shot hints from assembly context, scored against labels.
3. **Evaluation**: hinted prefetcher ensemble, input and compiler sensitivity. *(todo)*

## Setup

- **Kernels**: 13 C programs (`kernels/`), 14 workloads (`linked_list` runs `seq` and `shuf`), built at `-O0/-O2/-O3`.
- **Traces**: Valgrind `lackey`, `k_*` functions only.
- **Simulator**: L1D 32 KB 8-way (4 cyc), L2 512 KB 8-way (14 cyc), DRAM 200 cyc; in-order blocking core.
- **Prefetchers**: none, next-line, per-PC stride, stream; degree 1/2/4.
- **Labels** (`results/labels_<opt>.csv`): lowest-AMAT config per load, plus `acceptable` (configs within 5% of best) and `matters` (config choice changes AMAT by ≥2 cycles).
- **Hints**: ±48 lines of assembly around the load, local function names anonymised, prompt in `prompts/`. OpenRouter, temperature 0, reasoning off or minimal.
- **Models**: Qwen2.5-Coder-32B (PF-LLM's family, untuned), Claude Sonnet 5.5, GLM-5.3, DeepSeek V4.1 Flash.
- **Scoring**: exact / prefetcher-type / near-best accuracy and AMAT regret, vs. always-none, always-stream2 and majority-label baselines.

## Run

```sh
python3 -m cachehint.build
python3 -m cachehint.trace
python3 -m cachehint.label            # simulate + label (all opt levels)
python3 -m cachehint.label --relabel  # re-derive labels from saved AMATs
python3 -m cachehint.validate         # vs cachegrind
python3 -m cachehint.hints            # needs OPENROUTER_API_KEY
python3 -m cachehint.score
```

Validation (O2): L1 read misses match cachegrind within 0.8% on all workloads.
