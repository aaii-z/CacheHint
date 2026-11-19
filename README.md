# CacheHint

Mini replication of PF-LLM (Xu et al., ASPLOS '26): can an LLM pick the best
prefetcher for a load instruction from its code alone?

## Phases

1. **Ground truth**: trace kernels, simulate each prefetcher, label each load with its best config.
2. **LLM hints**: zero-shot hints from assembly context, scored against labels. *(todo)*
3. **Evaluation**: hinted prefetcher ensemble, input and compiler sensitivity. *(todo)*

## Setup

- **Kernels**: 13 C programs (`kernels/`), 14 workloads (`linked_list` runs `seq` and `shuf`), built at `-O0/-O2/-O3`.
- **Traces**: Valgrind `lackey`, `k_*` functions only.
- **Simulator**: L1D 32 KB 8-way (4 cyc), L2 512 KB 8-way (14 cyc), DRAM 200 cyc; in-order blocking core.
- **Prefetchers**: none, next-line, per-PC stride, stream; degree 1/2/4.
- **Labels** (`results/labels.csv`): lowest-AMAT config per load, plus `acceptable` (all near-best configs) and `matters` (config choice changes AMAT by ≥2 cycles).

## Run

```sh
python3 -m cachehint.build
python3 -m cachehint.trace
python3 -m cachehint.label            # simulate + label (all opt levels)
python3 -m cachehint.label --relabel  # re-derive labels from saved AMATs
python3 -m cachehint.validate         # vs cachegrind
```

Validation (O2): L1 read misses match cachegrind within 0.8% on all workloads.
